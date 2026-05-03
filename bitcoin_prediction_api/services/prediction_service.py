"""
Prediction service for Bitcoin Prediction API.

This module contains the main business logic for prediction generation,
orchestrating data fetching, mathematical engine execution, and persistence.
"""

from typing import List, Optional
from datetime import datetime, timedelta
import logging
import asyncio

from bitcoin_prediction_api.models.prediction import (
    PredictionResponse, 
    PredictionRecord, 
    PriceData
)
from bitcoin_prediction_api.services.math_engine_service import MathEngineService
from bitcoin_prediction_api.config import settings

logger = logging.getLogger(__name__)

class PredictionService:
    """
    Main service for Bitcoin prediction generation and management.
    
    This service orchestrates the complete prediction workflow:
    1. Fetch current Bitcoin price and historical data
    2. Generate prediction using mathematical engine
    3. Store prediction in database
    4. Return formatted response
    """
    
    def __init__(self, data_fetcher, persistence_layer, cache_manager):
        """
        Initialize prediction service with dependencies.
        
        Args:
            data_fetcher: Service for fetching Bitcoin price data
            persistence_layer: Database persistence service
            cache_manager: Caching service for performance optimization
        """
        self.data_fetcher = data_fetcher
        self.math_engine = MathEngineService()  # Initialize integrated math engine
        self.persistence = persistence_layer
        self.cache = cache_manager
    
    async def generate_prediction(self, confidence_level: float = 0.95) -> PredictionResponse:
        """
        Generate a new Bitcoin price prediction.
        
        This method implements the complete prediction workflow with caching
        and error handling. It fetches current price data, generates a prediction
        using the mathematical engine, stores the result, and returns a formatted response.
        
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
            raise ValueError(f"Invalid confidence_level: {confidence_level}")
        
        # Check cache first
        cache_key = f"prediction:{confidence_level}:{int(datetime.utcnow().timestamp() // 60)}"
        cached_prediction = await self.cache.get(cache_key)
        if cached_prediction:
            logger.info("Returning cached prediction", cache_key=cache_key)
            return cached_prediction
        
        logger.info("Generating new prediction", confidence_level=confidence_level)
        
        try:
            # Fetch current price and historical data
            current_price_data = await self.data_fetcher.fetch_current_price()
            historical_prices = await self.data_fetcher.fetch_historical_prices(
                limit=settings.lookback_window
            )
            
            logger.info(
                "Data fetched successfully",
                current_price=current_price_data.price,
                historical_count=len(historical_prices)
            )
            
            # Generate prediction using mathematical engine
            prediction_result = await self.math_engine.generate_prediction(
                current_price=current_price_data.price,
                historical_prices=historical_prices,
                confidence_level=confidence_level
            )
            
            # Create response object
            prediction_response = PredictionResponse(
                timestamp=datetime.utcnow(),
                current_price=current_price_data.price,
                lower_bound=prediction_result.lower_bound,
                upper_bound=prediction_result.upper_bound,
                confidence_level=confidence_level,
                volatility=prediction_result.volatility,
                drift=prediction_result.drift,
                interval_width=prediction_result.upper_bound - prediction_result.lower_bound
            )
            
            # Store prediction asynchronously (don't wait for completion)
            asyncio.create_task(self._store_prediction_async(prediction_response))
            
            # Cache the result
            await self.cache.set(cache_key, prediction_response, ttl=settings.cache_ttl_seconds)
            
            logger.info(
                "Prediction generated successfully",
                interval_width=prediction_response.interval_width,
                volatility=prediction_response.volatility
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
        Retrieve historical prediction data.
        
        Args:
            start_date: Start date for query range (optional)
            end_date: End date for query range (optional)
            limit: Maximum number of records to return
        
        Returns:
            List of PredictionRecord objects
        
        Raises:
            ValueError: If date range is invalid
            Exception: If database query fails
        """
        # Validate parameters
        if start_date and end_date and start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        # Set default date range if not provided
        if not end_date:
            end_date = datetime.utcnow()
        if not start_date:
            start_date = end_date - timedelta(days=7)  # Default to last 7 days
        
        logger.info(
            "Querying prediction history",
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        try:
            # Check cache first
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
            
            # Cache the result (longer TTL for historical data)
            await self.cache.set(cache_key, history, ttl=300)  # 5 minutes
            
            logger.info(f"Retrieved {len(history)} historical predictions")
            return history
            
        except Exception as e:
            logger.error(f"History query failed: {e}", exc_info=True)
            raise
    
    async def get_latest_prediction(self) -> Optional[PredictionRecord]:
        """
        Get the most recent prediction from the database.
        
        Returns:
            Latest PredictionRecord or None if no predictions exist
        """
        try:
            # Check cache first
            cached_latest = await self.cache.get("latest_prediction")
            if cached_latest:
                return cached_latest
            
            latest = await self.persistence.get_latest_prediction()
            
            if latest:
                # Cache for short duration
                await self.cache.set("latest_prediction", latest, ttl=60)
            
            return latest
            
        except Exception as e:
            logger.error(f"Failed to get latest prediction: {e}", exc_info=True)
            return None
    
    async def _store_prediction_async(self, prediction: PredictionResponse) -> None:
        """
        Store prediction in database asynchronously.
        
        This method runs in the background and logs errors without
        failing the main prediction generation workflow.
        
        Args:
            prediction: PredictionResponse to store
        """
        try:
            # Convert to database record
            record = PredictionRecord(
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
            await self.cache.delete_pattern("history:*")
            
            logger.info("Prediction stored successfully")
            
        except Exception as e:
            # Log error but don't fail the main workflow
            logger.error(f"Failed to store prediction: {e}", exc_info=True)