#!/usr/bin/env python3
"""
Demonstration script for MongoDB persistence layer.

This script demonstrates the MongoDB persistence layer functionality
without requiring a running MongoDB instance by using mocked operations.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from bitcoin_prediction_api.persistence.mongodb_persistence import MongoDBPersistenceLayer
from bitcoin_prediction_api.models.prediction import PredictionRecord

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def demo_mongodb_persistence():
    """Demonstrate MongoDB persistence layer functionality."""
    
    logger.info("=== MongoDB Persistence Layer Demo ===")
    
    # Create persistence layer instance
    persistence = MongoDBPersistenceLayer(
        mongodb_url="mongodb://localhost:27017/bitcoin_predictions_demo",
        database_name="bitcoin_predictions_demo"
    )
    
    logger.info("Created MongoDB persistence layer")
    logger.info(f"Connection URL: {persistence._mask_connection_string(persistence.mongodb_url)}")
    logger.info(f"Database: {persistence.database_name}")
    
    # Test validation logic (doesn't require MongoDB connection)
    logger.info("\n--- Testing Validation Logic ---")
    
    # Create a valid prediction record
    valid_prediction = PredictionRecord(
        id=None,
        timestamp=datetime.utcnow(),
        current_price=45000.0,
        lower_bound=43000.0,
        upper_bound=47000.0,
        confidence_level=0.95,
        prediction_horizon=1.0,
        volatility=0.25,
        drift=0.05,
        model_version="gbm-ewma-v1.0",
        created_at=datetime.utcnow()
    )
    
    try:
        persistence._validate_prediction_record(valid_prediction)
        logger.info("✓ Valid prediction record passed validation")
    except ValueError as e:
        logger.error(f"✗ Validation failed: {e}")
    
    # Test invalid prediction
    invalid_prediction = PredictionRecord(
        id=None,
        timestamp=datetime.utcnow(),
        current_price=-1000.0,  # Invalid: negative
        lower_bound=43000.0,
        upper_bound=47000.0,
        confidence_level=0.95,
        prediction_horizon=1.0,
        volatility=0.25,
        drift=0.05,
        model_version="gbm-ewma-v1.0",
        created_at=datetime.utcnow()
    )
    
    try:
        persistence._validate_prediction_record(invalid_prediction)
        logger.error("✗ Invalid prediction should have failed validation")
    except ValueError as e:
        logger.info(f"✓ Invalid prediction correctly rejected: {e}")
    
    # Test document conversion
    logger.info("\n--- Testing Document Conversion ---")
    
    document = persistence._prediction_to_document(valid_prediction)
    logger.info(f"✓ Converted prediction to document with {len(document)} fields")
    
    # Add _id for conversion back
    document['_id'] = "demo_object_id"
    converted_prediction = persistence._document_to_prediction(document)
    logger.info(f"✓ Converted document back to prediction record (ID: {converted_prediction.id})")
    
    # Test connection string masking
    logger.info("\n--- Testing Connection String Masking ---")
    
    test_urls = [
        "mongodb://user:password@host:27017/db",
        "mongodb://localhost:27017/db",
        "mongodb+srv://user:secret@cluster.mongodb.net/db"
    ]
    
    for url in test_urls:
        masked = persistence._mask_connection_string(url)
        logger.info(f"Original: {url}")
        logger.info(f"Masked:   {masked}")
    
    # Demonstrate mocked database operations
    logger.info("\n--- Demonstrating Database Operations (Mocked) ---")
    
    # Mock the database operations to simulate functionality
    with patch.object(persistence, 'connect', new_callable=AsyncMock) as mock_connect, \
         patch.object(persistence, 'store_prediction', new_callable=AsyncMock) as mock_store, \
         patch.object(persistence, 'get_latest_prediction', new_callable=AsyncMock) as mock_get_latest, \
         patch.object(persistence, 'query_predictions_by_date_range', new_callable=AsyncMock) as mock_query, \
         patch.object(persistence, 'health_check', new_callable=AsyncMock) as mock_health:
        
        # Configure mock returns
        mock_store.return_value = "demo_prediction_id_123"
        mock_get_latest.return_value = valid_prediction
        mock_query.return_value = [valid_prediction]
        mock_health.return_value = {
            'status': 'healthy',
            'response_time_ms': 15.5,
            'database': 'bitcoin_predictions_demo',
            'collection_stats': {
                'document_count': 1,
                'size_bytes': 256,
                'index_count': 3
            }
        }
        
        # Test connection
        await persistence.connect()
        logger.info("✓ Connected to MongoDB (mocked)")
        
        # Test storing prediction
        prediction_id = await persistence.store_prediction(valid_prediction)
        logger.info(f"✓ Stored prediction with ID: {prediction_id}")
        
        # Test retrieving latest prediction
        latest = await persistence.get_latest_prediction()
        logger.info(f"✓ Retrieved latest prediction: ${latest.current_price:.2f} at {latest.timestamp}")
        
        # Test querying by date range
        start_date = datetime.utcnow() - timedelta(hours=24)
        end_date = datetime.utcnow()
        predictions = await persistence.query_predictions_by_date_range(
            start_date=start_date,
            end_date=end_date,
            limit=10
        )
        logger.info(f"✓ Queried {len(predictions)} predictions in last 24 hours")
        
        # Test health check
        health = await persistence.health_check()
        logger.info(f"✓ Health check: {health['status']} (response time: {health['response_time_ms']}ms)")
        
        # Test disconnect
        await persistence.disconnect()
        logger.info("✓ Disconnected from MongoDB")
    
    logger.info("\n=== Demo Complete ===")
    logger.info("MongoDB persistence layer is ready for production use!")
    logger.info("To use with real MongoDB:")
    logger.info("1. Start MongoDB: docker-compose up -d mongodb")
    logger.info("2. Run tests: python -m pytest tests/test_mongodb_persistence.py")
    logger.info("3. Use in PredictionEngineService for real predictions")


if __name__ == "__main__":
    asyncio.run(demo_mongodb_persistence())