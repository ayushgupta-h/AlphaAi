"""
MongoDB persistence layer for Bitcoin Prediction API.

This module implements the MongoDB persistence layer using Motor (async MongoDB driver)
with proper schema validation, connection pooling, and retry logic for database operations.
This replaces the MockPersistenceLayer in the PredictionEngineService.

Requirements: 3.1, 12.1, 12.2
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import math

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo.errors import (
    ConnectionFailure, 
    ServerSelectionTimeoutError, 
    DuplicateKeyError,
    WriteError,
    BulkWriteError
)
from pymongo import DESCENDING, ASCENDING
import pymongo

from bitcoin_prediction_api.models.prediction import PredictionRecord
from bitcoin_prediction_api.config import settings

logger = logging.getLogger(__name__)


class MongoDBPersistenceLayer:
    """
    MongoDB persistence layer for prediction storage and retrieval.
    
    This class provides async MongoDB operations using Motor driver with:
    - Connection pooling and retry logic for database operations
    - Schema validation for predictions collection
    - Efficient indexing for date range queries
    - Proper error handling and logging
    
    Requirements: 3.1, 12.1, 12.2
    """
    
    def __init__(self, mongodb_url: Optional[str] = None, database_name: Optional[str] = None):
        """
        Initialize MongoDB persistence layer.
        
        Args:
            mongodb_url: MongoDB connection URL (defaults to settings.mongodb_url)
            database_name: Database name (defaults to settings.mongodb_database)
        """
        self.mongodb_url = mongodb_url or settings.mongodb_url
        self.database_name = database_name or settings.mongodb_database
        
        # Connection configuration with pooling and timeouts
        self.client_options = {
            'maxPoolSize': 10,  # Maximum connections in pool
            'minPoolSize': 2,   # Minimum connections in pool
            'maxIdleTimeMS': 30000,  # 30 seconds idle timeout
            'serverSelectionTimeoutMS': 5000,  # 5 seconds server selection timeout
            'connectTimeoutMS': 10000,  # 10 seconds connection timeout
            'socketTimeoutMS': 20000,  # 20 seconds socket timeout
            'retryWrites': True,  # Enable retryable writes
            'retryReads': True,   # Enable retryable reads
            'w': 'majority',      # Write concern: majority
            'readPreference': 'primary'  # Read from primary
        }
        
        self._client: Optional[AsyncIOMotorClient] = None
        self._database: Optional[AsyncIOMotorDatabase] = None
        self._predictions_collection: Optional[AsyncIOMotorCollection] = None
        self._connected = False
        
        logger.info(
            "MongoDBPersistenceLayer initialized - url=%s, database=%s",
            self._mask_connection_string(self.mongodb_url),
            self.database_name
        )
    
    async def connect(self) -> None:
        """
        Establish connection to MongoDB with retry logic.
        
        Raises:
            ConnectionFailure: If connection cannot be established after retries
        """
        if self._connected:
            logger.debug("Already connected to MongoDB")
            return
        
        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                logger.info(
                    "Connecting to MongoDB - attempt %d/%d, url=%s",
                    attempt + 1,
                    max_retries,
                    self._mask_connection_string(self.mongodb_url)
                )
                
                # Create client with connection options
                self._client = AsyncIOMotorClient(self.mongodb_url, **self.client_options)
                
                # Test connection with ping
                await self._client.admin.command('ping')
                
                # Get database and collection references
                self._database = self._client[self.database_name]
                self._predictions_collection = self._database.predictions
                
                # Verify collection exists and has proper schema
                await self._verify_collection_schema()
                
                self._connected = True
                logger.info("Successfully connected to MongoDB - database=%s", self.database_name)
                return
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(
                    "MongoDB connection attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    str(e)
                )
                
                if attempt < max_retries - 1:
                    # Exponential backoff with jitter
                    delay = retry_delay * (2 ** attempt) + (0.1 * attempt)
                    logger.info("Retrying MongoDB connection in %.1f seconds", delay)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Failed to connect to MongoDB after %d attempts", max_retries)
                    raise ConnectionFailure(f"Could not connect to MongoDB: {e}") from e
            
            except Exception as e:
                logger.error("Unexpected error during MongoDB connection: %s", str(e))
                raise
    
    async def disconnect(self) -> None:
        """Close MongoDB connection and cleanup resources."""
        if self._client:
            try:
                self._client.close()
                logger.info("MongoDB connection closed")
            except Exception as e:
                logger.error("Error closing MongoDB connection: %s", str(e))
            finally:
                self._client = None
                self._database = None
                self._predictions_collection = None
                self._connected = False
    
    async def store_prediction(self, prediction: PredictionRecord) -> str:
        """
        Store prediction record in MongoDB with validation and retry logic.
        
        Args:
            prediction: PredictionRecord to store
            
        Returns:
            str: The ObjectId of the stored document as string
            
        Raises:
            ValueError: If prediction data is invalid
            ConnectionFailure: If database operation fails after retries
        """
        await self._ensure_connected()
        
        # Validate prediction data before storage
        self._validate_prediction_record(prediction)
        
        # Convert to MongoDB document
        document = self._prediction_to_document(prediction)
        
        max_retries = 3
        retry_delay = 0.1  # Start with 100ms delay
        
        for attempt in range(max_retries):
            try:
                logger.debug(
                    "Storing prediction - attempt %d/%d, timestamp=%s, confidence_level=%.3f",
                    attempt + 1,
                    max_retries,
                    document['timestamp'],
                    document['confidence_level']
                )
                
                # Insert document with retry
                result = await self._predictions_collection.insert_one(document)
                
                if not result.acknowledged:
                    raise WriteError("Insert operation was not acknowledged by MongoDB")
                
                document_id = str(result.inserted_id)
                logger.info(
                    "Prediction stored successfully - id=%s, timestamp=%s, interval_width=%.2f",
                    document_id,
                    document['timestamp'],
                    document['upper_bound'] - document['lower_bound']
                )
                
                return document_id
                
            except (ConnectionFailure, ServerSelectionTimeoutError) as e:
                logger.warning(
                    "MongoDB store attempt %d/%d failed: %s",
                    attempt + 1,
                    max_retries,
                    str(e)
                )
                
                if attempt < max_retries - 1:
                    delay = retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                else:
                    logger.error("Failed to store prediction after %d attempts", max_retries)
                    raise ConnectionFailure(f"Could not store prediction: {e}") from e
            
            except (WriteError, BulkWriteError) as e:
                logger.error("MongoDB write error: %s", str(e))
                raise ValueError(f"Invalid prediction data: {e}") from e
            
            except Exception as e:
                logger.error("Unexpected error storing prediction: %s", str(e))
                raise
    
    async def query_predictions_by_date_range(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 1000
    ) -> List[PredictionRecord]:
        """
        Query historical predictions by date range with efficient indexing.
        
        Args:
            start_date: Start date for query range
            end_date: End date for query range  
            limit: Maximum number of records to return (max 1000)
            
        Returns:
            List[PredictionRecord]: Predictions sorted by timestamp descending
            
        Raises:
            ValueError: If date range or limit is invalid
            ConnectionFailure: If database query fails
        """
        await self._ensure_connected()
        
        # Validate parameters
        if start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        if limit <= 0 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
        
        try:
            logger.debug(
                "Querying predictions - start_date=%s, end_date=%s, limit=%d",
                start_date,
                end_date,
                limit
            )
            
            # Build query with date range filter
            query = {
                'timestamp': {
                    '$gte': start_date,
                    '$lte': end_date
                }
            }
            
            # Execute query with efficient indexing (uses timestamp_confidence_idx)
            cursor = self._predictions_collection.find(query).sort('timestamp', DESCENDING).limit(limit)
            
            # Convert documents to PredictionRecord objects
            predictions = []
            async for document in cursor:
                prediction = self._document_to_prediction(document)
                predictions.append(prediction)
            
            logger.info(
                "Query completed - found %d predictions in range %s to %s",
                len(predictions),
                start_date,
                end_date
            )
            
            return predictions
            
        except Exception as e:
            logger.error("Failed to query predictions by date range: %s", str(e))
            raise ConnectionFailure(f"Database query failed: {e}") from e
    
    async def get_latest_prediction(self) -> Optional[PredictionRecord]:
        """
        Retrieve the most recent prediction.
        
        Returns:
            Optional[PredictionRecord]: Latest prediction or None if no predictions exist
            
        Raises:
            ConnectionFailure: If database query fails
        """
        await self._ensure_connected()
        
        try:
            logger.debug("Querying latest prediction")
            
            # Find most recent prediction (uses timestamp_desc_idx)
            document = await self._predictions_collection.find_one(
                {},
                sort=[('timestamp', DESCENDING)]
            )
            
            if document:
                prediction = self._document_to_prediction(document)
                logger.info("Latest prediction retrieved - timestamp=%s", prediction.timestamp)
                return prediction
            else:
                logger.info("No predictions found in database")
                return None
                
        except Exception as e:
            logger.error("Failed to get latest prediction: %s", str(e))
            raise ConnectionFailure(f"Database query failed: {e}") from e
    
    async def get_prediction_count(self) -> int:
        """
        Get total count of predictions in database.
        
        Returns:
            int: Total number of predictions
            
        Raises:
            ConnectionFailure: If database query fails
        """
        await self._ensure_connected()
        
        try:
            count = await self._predictions_collection.count_documents({})
            logger.debug("Prediction count retrieved - total=%d", count)
            return count
            
        except Exception as e:
            logger.error("Failed to get prediction count: %s", str(e))
            raise ConnectionFailure(f"Database query failed: {e}") from e
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on MongoDB connection and collection.
        
        Returns:
            Dict[str, Any]: Health status information
        """
        try:
            await self._ensure_connected()
            
            # Test basic operations
            start_time = datetime.utcnow()
            
            # Ping database
            await self._client.admin.command('ping')
            
            # Get collection stats
            stats = await self._database.command('collStats', 'predictions')
            
            # Calculate response time
            response_time_ms = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            return {
                'status': 'healthy',
                'response_time_ms': round(response_time_ms, 2),
                'database': self.database_name,
                'collection_stats': {
                    'document_count': stats.get('count', 0),
                    'size_bytes': stats.get('size', 0),
                    'index_count': stats.get('nindexes', 0)
                },
                'connection_pool': {
                    'max_pool_size': self.client_options['maxPoolSize'],
                    'min_pool_size': self.client_options['minPoolSize']
                }
            }
            
        except Exception as e:
            logger.error("MongoDB health check failed: %s", str(e))
            return {
                'status': 'unhealthy',
                'error': str(e),
                'database': self.database_name
            }
    
    async def _ensure_connected(self) -> None:
        """Ensure MongoDB connection is established."""
        if not self._connected:
            await self.connect()
    
    async def _verify_collection_schema(self) -> None:
        """Verify that predictions collection exists with proper schema validation."""
        try:
            # Check if collection exists
            collections = await self._database.list_collection_names()
            if 'predictions' not in collections:
                logger.warning("Predictions collection does not exist - it will be created on first insert")
                return
            
            # Verify indexes exist
            indexes = await self._predictions_collection.list_indexes().to_list(length=None)
            index_names = [idx['name'] for idx in indexes]
            
            expected_indexes = ['timestamp_confidence_idx', 'timestamp_desc_idx']
            missing_indexes = [idx for idx in expected_indexes if idx not in index_names]
            
            if missing_indexes:
                logger.warning("Missing expected indexes: %s", missing_indexes)
            else:
                logger.info("All expected indexes are present")
            
        except Exception as e:
            logger.warning("Could not verify collection schema: %s", str(e))
    
    def _validate_prediction_record(self, prediction: PredictionRecord) -> None:
        """
        Validate prediction record before storage.
        
        Args:
            prediction: PredictionRecord to validate
            
        Raises:
            ValueError: If prediction data is invalid
        """
        # Validate required fields
        if not prediction.timestamp:
            raise ValueError("timestamp is required")
        
        if prediction.current_price is None or prediction.current_price <= 0:
            raise ValueError("current_price must be positive")
        
        if prediction.lower_bound is None or prediction.lower_bound <= 0:
            raise ValueError("lower_bound must be positive")
        
        if prediction.upper_bound is None or prediction.upper_bound <= 0:
            raise ValueError("upper_bound must be positive")
        
        if prediction.lower_bound >= prediction.upper_bound:
            raise ValueError("lower_bound must be less than upper_bound")
        
        if prediction.confidence_level is None or not (0.5 <= prediction.confidence_level <= 0.999):
            raise ValueError("confidence_level must be between 0.5 and 0.999")
        
        # Validate numeric fields are finite
        numeric_fields = [
            ('current_price', prediction.current_price),
            ('lower_bound', prediction.lower_bound),
            ('upper_bound', prediction.upper_bound),
            ('confidence_level', prediction.confidence_level),
            ('volatility', prediction.volatility),
            ('drift', prediction.drift),
            ('prediction_horizon', prediction.prediction_horizon)
        ]
        
        for field_name, value in numeric_fields:
            if value is not None and not math.isfinite(value):
                raise ValueError(f"{field_name} must be finite, got {value}")
    
    def _prediction_to_document(self, prediction: PredictionRecord) -> Dict[str, Any]:
        """
        Convert PredictionRecord to MongoDB document.
        
        Args:
            prediction: PredictionRecord to convert
            
        Returns:
            Dict[str, Any]: MongoDB document
        """
        document = {
            'timestamp': prediction.timestamp,
            'current_price': prediction.current_price,
            'lower_bound': prediction.lower_bound,
            'upper_bound': prediction.upper_bound,
            'confidence_level': prediction.confidence_level,
            'prediction_horizon': prediction.prediction_horizon or 1.0,
            'volatility': prediction.volatility,
            'drift': prediction.drift,
            'model_version': prediction.model_version or "gbm-ewma-v1.0",
            'created_at': prediction.created_at or datetime.utcnow()
        }
        
        # Remove None values to keep document clean
        return {k: v for k, v in document.items() if v is not None}
    
    def _document_to_prediction(self, document: Dict[str, Any]) -> PredictionRecord:
        """
        Convert MongoDB document to PredictionRecord.
        
        Args:
            document: MongoDB document
            
        Returns:
            PredictionRecord: Converted prediction record
        """
        return PredictionRecord(
            id=str(document['_id']),
            timestamp=document['timestamp'],
            current_price=document['current_price'],
            lower_bound=document['lower_bound'],
            upper_bound=document['upper_bound'],
            confidence_level=document['confidence_level'],
            prediction_horizon=document.get('prediction_horizon', 1.0),
            volatility=document.get('volatility'),
            drift=document.get('drift'),
            model_version=document.get('model_version', "gbm-ewma-v1.0"),
            created_at=document.get('created_at', document['timestamp'])
        )
    
    def _mask_connection_string(self, connection_string: str) -> str:
        """
        Mask sensitive information in connection string for logging.
        
        Args:
            connection_string: MongoDB connection string
            
        Returns:
            str: Masked connection string
        """
        try:
            # Simple masking - replace password with asterisks
            if '@' in connection_string and '://' in connection_string:
                parts = connection_string.split('://', 1)
                if len(parts) == 2:
                    protocol = parts[0]
                    rest = parts[1]
                    if '@' in rest:
                        auth_part, host_part = rest.split('@', 1)
                        if ':' in auth_part:
                            username, _ = auth_part.split(':', 1)
                            return f"{protocol}://{username}:***@{host_part}"
            
            return connection_string
            
        except Exception:
            return "***masked***"


# Global persistence layer instance
_persistence_layer: Optional[MongoDBPersistenceLayer] = None


async def get_persistence_layer() -> MongoDBPersistenceLayer:
    """
    Get or create global MongoDB persistence layer instance.
    
    Returns:
        MongoDBPersistenceLayer: Global persistence layer instance
    """
    global _persistence_layer
    
    if _persistence_layer is None:
        _persistence_layer = MongoDBPersistenceLayer()
        await _persistence_layer.connect()
    
    return _persistence_layer


async def close_persistence_layer() -> None:
    """Close global persistence layer connection."""
    global _persistence_layer
    
    if _persistence_layer:
        await _persistence_layer.disconnect()
        _persistence_layer = None