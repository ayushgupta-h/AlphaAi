"""
Prediction Engine Service for Bitcoin Prediction API.

This module implements the main orchestrator service that coordinates the GBM engine,
EWMA calculator, caching, and external data integration for complete prediction workflow.
This service serves as the high-level coordinator as specified in Task 2.3.
"""

import logging
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import numpy as np

from bitcoin_prediction_api.models.prediction import (
    PredictionResponse, 
    PredictionRecord
)
from bitcoin_prediction_api.services.math_engine_service import MathEngineService, PredictionResult
from bitcoin_prediction_api.services.real_time_data_fetcher import RealTimeDataFetcher
from bitcoin_prediction_api.persistence.mongodb_persistence import MongoDBPersistenceLayer
from bitcoin_prediction_api.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Simple in-memory cache manager for prediction results."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            entry = self._cache[key]
            if datetime.utcnow() < entry['expires_at']:
                logger.debug(f"Cache hit for key: {key}")
                return entry['value']
            else:
                # Remove expired entry
                del self._cache[key]
                logger.debug(f"Cache expired for key: {key}")
        
        logger.debug(f"Cache miss for key: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: int) -> None:
        """Set cached value with TTL in seconds."""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        self._cache[key] = {
            'value': value,
            'expires_at': expires_at
        }
        logger.debug(f"Cached value for key: {key}, TTL: {ttl}s")
    
    async def delete(self, key: str) -> None:
        """Delete cached value."""
        if key in self._cache:
            del self._cache[key]
            logger.debug(f"Deleted cache key: {key}")
    
    async def delete_pattern(self, pattern: str) -> None:
        """Delete all keys matching pattern (simple prefix matching)."""
        prefix = pattern.replace('*', '')
        keys_to_delete = [key for key in self._cache.keys() if key.startswith(prefix)]
        for key in keys_to_delete:
            del self._cache[key]
        logger.debug(f"Deleted {len(keys_to_delete)} keys matching pattern: {pattern}")


class PredictionEngineService:
    """
    Main prediction engine service orchestrator.
    
    This service coordinates the complete prediction workflow including:
    - GBM engine and EWMA calculator integration
    - Real-time data fetching and caching
    - Volatility computation with configurable lookback window
    - Drift estimation from recent price returns
    - Prediction result caching and persistence
    
    Requirements: 4.5, 4.6, 4.7
    """
    
    def __init__(self):
        """Initialize the prediction engine service with all components."""
        # Initialize mathematical engine
        self.math_engine = MathEngineService()
        
        # Initialize cache manager
        self.cache = CacheManager()
        
        # Initialize real-time data fetcher
        self.data_fetcher = RealTimeDataFetcher()
        
        # Initialize MongoDB persistence layer
        self.persistence = MongoDBPersistenceLayer()
        
        logger.info(
            "PredictionEngineService initialized - lookback_window=%d, cache_ttl=%d, n_simulations=%d",
            settings.lookback_window,
            settings.cache_ttl_seconds,
            settings.n_simulations
        )
    
    async def initialize(self) -> None:
        """Initialize all service components including database connection."""
        try:
            # Connect to MongoDB
            await self.persistence.connect()
            logger.info("PredictionEngineService initialization completed successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PredictionEngineService: {e}")
            raise
    
    async def generate_prediction(self, confidence_level: float = 0.95) -> PredictionResponse:
        """
        Generate a complete Bitcoin price prediction with caching and persistence.
        
        This method orchestrates the complete prediction workflow:
        1. Check cache for recent prediction
        2. Fetch current price and historical data
        3. Compute volatility using EWMA with configurable lookback window
        4. Estimate drift from recent price returns
        5. Generate prediction using GBM engine
        6. Cache and persist results
        
        Args:
            confidence_level: Confidence level for prediction interval (0.5-0.999)
        
        Returns:
            PredictionResponse containing prediction data and metadata
        
        Raises:
            ValueError: If confidence_level is invalid
            Exception: If prediction generation fails
        """
        # Validate confidence level
        if not (0.5 <= confidence_level <= 0.999):
            raise ValueError(f"Invalid confidence_level: {confidence_level}. Must be between 0.5 and 0.999")
        
        # Check cache first (cache key includes confidence level and minute timestamp)
        cache_key = f"prediction:{confidence_level}:{int(datetime.utcnow().timestamp() // 60)}"
        cached_prediction = await self.cache.get(cache_key)
        if cached_prediction:
            logger.info("Returning cached prediction - cache_key=%s", cache_key)
            return cached_prediction
        
        logger.info("Generating new prediction - confidence_level=%.3f", confidence_level)
        
        try:
            # Step 1: Fetch current price and historical data
            price_data = await self.data_fetcher.fetch_current_price()
            current_price = price_data.price
            historical_prices = await self.data_fetcher.fetch_historical_prices(
                limit=settings.lookback_window
            )
            
            logger.info(
                "Data fetched successfully - current_price=%.2f, source=%s, historical_count=%d, lookback_window=%d",
                current_price,
                price_data.source,
                len(historical_prices),
                settings.lookback_window
            )
            
            # Step 2: Generate prediction using mathematical engine
            # (This internally computes EWMA volatility and drift estimation)
            prediction_result = await self.math_engine.generate_prediction(
                current_price=current_price,
                historical_prices=historical_prices,
                confidence_level=confidence_level
            )
            
            # Step 3: Create response object with all metadata
            prediction_response = PredictionResponse(
                timestamp=datetime.utcnow(),
                current_price=current_price,
                lower_bound=prediction_result.lower_bound,
                upper_bound=prediction_result.upper_bound,
                confidence_level=confidence_level,
                volatility=prediction_result.volatility,
                drift=prediction_result.drift,
                interval_width=prediction_result.upper_bound - prediction_result.lower_bound
            )
            
            # Step 4: Store prediction asynchronously (don't block response)
            asyncio.create_task(self._store_prediction_async(prediction_response))
            
            # Step 5: Cache the result
            await self.cache.set(cache_key, prediction_response, ttl=settings.cache_ttl_seconds)
            
            logger.info(
                "Prediction generated successfully - interval_width=%.2f, volatility=%.4f, drift=%.6f, cache_key=%s",
                prediction_response.interval_width,
                prediction_response.volatility,
                prediction_response.drift,
                cache_key
            )
            
            return prediction_response
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}", exc_info=True)
            raise
    
    async def get_prediction_history(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ) -> List[PredictionRecord]:
        """
        Retrieve historical prediction data with caching.
        
        Args:
            start_date: Start date for query range (optional)
            end_date: End date for query range (optional)
            limit: Maximum number of records to return (max 1000)
        
        Returns:
            List of PredictionRecord objects
        
        Raises:
            ValueError: If date range is invalid or limit exceeds maximum
            Exception: If database query fails
        """
        # Validate parameters
        if limit > 1000:
            raise ValueError(f"Limit cannot exceed 1000, got {limit}")
        
        if start_date and end_date and start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)  # Default to last 7 days
        
        logger.info(
            "Querying prediction history - start_date=%s, end_date=%s, limit=%d",
            start_date,
            end_date,
            limit
        )
        
        try:
            # Check cache first (longer TTL for historical data)
            cache_key = f"history:{start_date.isoformat()}:{end_date.isoformat()}:{limit}"
            cached_history = await self.cache.get(cache_key)
            if cached_history:
                logger.info("Returning cached history", cache_key=cache_key)
                return cached_history
            
            # Query database
            history = await self.persistence.query_predictions_by_date_range(
                start_date=start_date,
                end_date=end_date,
                limit=limit
            )
            
            # Cache the result (longer TTL for historical data - 5 minutes)
            await self.cache.set(cache_key, history, ttl=300)
            
            logger.info(f"Retrieved {len(history)} historical predictions")
            return history
            
        except Exception as e:
            logger.error(f"History query failed: {e}", exc_info=True)
            raise
    
    async def get_latest_prediction(self) -> Optional[PredictionRecord]:
        """
        Get the most recent prediction with caching.
        
        Returns:
            Latest PredictionRecord or None if no predictions exist
        """
        try:
            # Check cache first
            cached_latest = await self.cache.get("latest_prediction")
            if cached_latest:
                logger.debug("Returning cached latest prediction")
                return cached_latest
            
            latest = await self.persistence.get_latest_prediction()
            
            if latest:
                # Cache for short duration (1 minute)
                await self.cache.set("latest_prediction", latest, ttl=60)
                logger.info(f"Retrieved latest prediction from {latest.timestamp}")
            else:
                logger.info("No predictions found in database")
            
            return latest
            
        except Exception as e:
            logger.error(f"Failed to get latest prediction: {e}", exc_info=True)
            return None
    
    async def get_engine_status(self) -> Dict[str, Any]:
        """
        Get prediction engine status and configuration.
        
        Returns:
            Dictionary containing engine status and configuration
        """
        try:
            # Get mathematical engine configuration
            math_config = self.math_engine.get_config_summary()
            
            # Get cache statistics (simple implementation)
            cache_size = len(self.cache._cache)
            
            # Get latest prediction info
            latest_prediction = await self.get_latest_prediction()
            
            status = {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "mathematical_engine": math_config,
                "cache": {
                    "size": cache_size,
                    "ttl_seconds": settings.cache_ttl_seconds
                },
                "data_fetcher": {
                    "type": "mock",  # Will be updated when real implementation is added
                    "lookback_window": settings.lookback_window
                },
                "persistence": {
                    "type": "mongodb",
                    "database": settings.mongodb_database,
                    "latest_prediction": latest_prediction.timestamp.isoformat() if latest_prediction else None
                }
            }
            
            logger.debug("Engine status retrieved successfully")
            return status
            
        except Exception as e:
            logger.error(f"Failed to get engine status: {e}", exc_info=True)
            return {
                "status": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
    
    async def _store_prediction_async(self, prediction: PredictionResponse) -> None:
        """
        Store prediction in database asynchronously.
        
        This method runs in the background and logs errors without
        failing the main prediction generation workflow.
        
        Args:
            prediction: PredictionResponse to store
        """
        try:
            # Convert to database record (id will be set by persistence layer)
            record = PredictionRecord(
                id=None,  # Will be set by persistence layer
                timestamp=prediction.timestamp,
                current_price=prediction.current_price,
                lower_bound=prediction.lower_bound,
                upper_bound=prediction.upper_bound,
                confidence_level=prediction.confidence_level,
                prediction_horizon=prediction.prediction_horizon,
                volatility=prediction.volatility,
                drift=prediction.drift,
                model_version=prediction.model_version,
                created_at=datetime.utcnow()
            )
            
            await self.persistence.store_prediction(record)
            
            # Invalidate related caches
            await self.cache.delete("latest_prediction")
            await self.cache.delete_pattern("history:")
            
            logger.info("Prediction stored successfully")
            
        except Exception as e:
            # Log error but don't fail the main workflow
            logger.error(f"Failed to store prediction: {e}", exc_info=True)
    
    async def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache._cache.clear()
        logger.info("Cache cleared successfully")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_entries = len(self.cache._cache)
        expired_entries = 0
        
        # Count expired entries
        now = datetime.utcnow()
        for entry in self.cache._cache.values():
            if now >= entry['expires_at']:
                expired_entries += 1
        
        return {
            "total_entries": total_entries,
            "active_entries": total_entries - expired_entries,
            "expired_entries": expired_entries,
            "ttl_seconds": settings.cache_ttl_seconds
        }
    
    async def cleanup(self) -> None:
        """Clean up resources when shutting down the service."""
        try:
            await self.data_fetcher.close()
            await self.persistence.disconnect()
            logger.info("PredictionEngineService cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}", exc_info=True)