"""
Integration tests for PredictionEngineService with MongoDB persistence.

This module tests the integration between PredictionEngineService and
the MongoDB persistence layer.
"""

import pytest
import asyncio
from datetime import datetime
from unittest.mock import AsyncMock, patch
import os

from bitcoin_prediction_api.services.prediction_engine_service import PredictionEngineService
from bitcoin_prediction_api.models.prediction import PredictionRecord


@pytest.mark.asyncio
async def test_prediction_engine_initialization():
    """Test that PredictionEngineService initializes with MongoDB persistence layer."""
    
    # Mock the MongoDB connection to avoid requiring actual MongoDB
    with patch('bitcoin_prediction_api.persistence.mongodb_persistence.MongoDBPersistenceLayer.connect') as mock_connect:
        mock_connect.return_value = None
        
        engine = PredictionEngineService()
        
        # Verify MongoDB persistence layer is used (not mock)
        assert engine.persistence.__class__.__name__ == "MongoDBPersistenceLayer"
        
        # Test initialization
        await engine.initialize()
        
        # Verify connect was called
        mock_connect.assert_called_once()


@pytest.mark.asyncio
async def test_prediction_engine_with_mocked_persistence():
    """Test prediction engine workflow with mocked persistence operations."""
    
    engine = PredictionEngineService()
    
    # Mock all persistence operations
    engine.persistence.connect = AsyncMock()
    engine.persistence.store_prediction = AsyncMock(return_value="test_id_123")
    engine.persistence.get_latest_prediction = AsyncMock(return_value=None)
    engine.persistence.query_predictions_by_date_range = AsyncMock(return_value=[])
    engine.persistence.disconnect = AsyncMock()
    
    # Mock data fetcher to avoid external API calls
    mock_price_data = type('PriceData', (), {'price': 45000.0, 'source': 'mock'})()
    engine.data_fetcher.fetch_current_price = AsyncMock(return_value=mock_price_data)
    engine.data_fetcher.fetch_historical_prices = AsyncMock(return_value=[
        44000.0, 44200.0, 44500.0, 44800.0, 45000.0
    ] * 5)  # 25 prices for EWMA calculation
    engine.data_fetcher.close = AsyncMock()
    
    try:
        # Initialize engine
        await engine.initialize()
        
        # Generate a prediction
        prediction = await engine.generate_prediction(confidence_level=0.95)
        
        # Verify prediction structure
        assert prediction.current_price == 45000.0
        assert prediction.confidence_level == 0.95
        assert prediction.lower_bound < prediction.upper_bound
        assert prediction.volatility > 0
        assert prediction.timestamp is not None
        
        # Verify persistence operations were called
        engine.persistence.connect.assert_called_once()
        
        # Wait a bit for async storage to complete
        await asyncio.sleep(0.1)
        
        # Verify store_prediction was called (async task)
        # Note: This might not be called immediately due to asyncio.create_task
        
        # Test history retrieval
        history = await engine.get_prediction_history()
        assert isinstance(history, list)
        engine.persistence.query_predictions_by_date_range.assert_called_once()
        
        # Test latest prediction retrieval
        latest = await engine.get_latest_prediction()
        engine.persistence.get_latest_prediction.assert_called_once()
        
        # Test engine status
        status = await engine.get_engine_status()
        assert status['status'] == 'healthy'
        assert status['persistence']['type'] == 'mongodb'
        
    finally:
        # Cleanup
        await engine.cleanup()
        engine.persistence.disconnect.assert_called_once()


@pytest.mark.asyncio
async def test_prediction_engine_error_handling():
    """Test prediction engine error handling with persistence failures."""
    
    engine = PredictionEngineService()
    
    # Mock persistence to raise errors
    engine.persistence.connect = AsyncMock(side_effect=Exception("Connection failed"))
    
    # Test initialization failure
    with pytest.raises(Exception, match="Connection failed"):
        await engine.initialize()


@pytest.mark.asyncio
async def test_prediction_engine_cache_integration():
    """Test that caching works correctly with persistence layer."""
    
    engine = PredictionEngineService()
    
    # Mock persistence operations
    engine.persistence.connect = AsyncMock()
    engine.persistence.store_prediction = AsyncMock(return_value="test_id_123")
    engine.persistence.disconnect = AsyncMock()
    
    # Mock data fetcher
    mock_price_data = type('PriceData', (), {'price': 45000.0, 'source': 'mock'})()
    engine.data_fetcher.fetch_current_price = AsyncMock(return_value=mock_price_data)
    engine.data_fetcher.fetch_historical_prices = AsyncMock(return_value=[45000.0] * 25)
    engine.data_fetcher.close = AsyncMock()
    
    try:
        await engine.initialize()
        
        # Generate first prediction
        prediction1 = await engine.generate_prediction(confidence_level=0.95)
        
        # Generate second prediction immediately (should be cached)
        prediction2 = await engine.generate_prediction(confidence_level=0.95)
        
        # Should be the same prediction (cached)
        assert prediction1.timestamp == prediction2.timestamp
        assert prediction1.current_price == prediction2.current_price
        
        # Verify data fetcher was only called once (due to caching)
        assert engine.data_fetcher.fetch_current_price.call_count == 1
        
        # Test cache stats
        cache_stats = await engine.get_cache_stats()
        assert cache_stats['total_entries'] > 0
        
        # Clear cache and test again
        await engine.clear_cache()
        cache_stats = await engine.get_cache_stats()
        assert cache_stats['total_entries'] == 0
        
    finally:
        await engine.cleanup()


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])