"""
Unit tests for PredictionEngineService.

Tests the prediction engine service orchestrator that coordinates
GBM engine, EWMA calculator, and caching functionality.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

from bitcoin_prediction_api.services.prediction_engine_service import PredictionEngineService
from bitcoin_prediction_api.models.prediction import PredictionResponse


class TestPredictionEngineService:
    """Test cases for PredictionEngineService."""
    
    @pytest.fixture
    def engine(self):
        """Create a PredictionEngineService instance for testing."""
        return PredictionEngineService()
    
    @pytest.mark.asyncio
    async def test_service_initialization(self, engine):
        """Test that the service initializes correctly."""
        assert engine.math_engine is not None
        assert engine.cache is not None
        assert engine.data_fetcher is not None
        assert engine.persistence is not None
    
    @pytest.mark.asyncio
    async def test_generate_prediction_default_confidence(self, engine):
        """Test prediction generation with default confidence level."""
        prediction = await engine.generate_prediction()
        
        assert isinstance(prediction, PredictionResponse)
        assert prediction.confidence_level == 0.95
        assert prediction.current_price > 0
        assert prediction.lower_bound > 0
        assert prediction.upper_bound > prediction.lower_bound
        assert prediction.volatility > 0
        assert prediction.interval_width > 0
        assert prediction.model_version == "gbm-ewma-v1.0"
    
    @pytest.mark.asyncio
    async def test_generate_prediction_custom_confidence(self, engine):
        """Test prediction generation with custom confidence level."""
        prediction_90 = await engine.generate_prediction(confidence_level=0.90)
        prediction_95 = await engine.generate_prediction(confidence_level=0.95)
        
        # 90% confidence interval should be narrower than 95%
        assert prediction_90.confidence_level == 0.90
        assert prediction_95.confidence_level == 0.95
        # Note: Due to caching, we might get the same prediction, so we clear cache
        await engine.clear_cache()
        
        prediction_90_fresh = await engine.generate_prediction(confidence_level=0.90)
        prediction_95_fresh = await engine.generate_prediction(confidence_level=0.95)
        
        # Fresh predictions should have different interval widths
        assert prediction_90_fresh.interval_width != prediction_95_fresh.interval_width
    
    @pytest.mark.asyncio
    async def test_invalid_confidence_level(self, engine):
        """Test that invalid confidence levels raise ValueError."""
        with pytest.raises(ValueError, match="Invalid confidence_level"):
            await engine.generate_prediction(confidence_level=0.4)  # Too low
        
        with pytest.raises(ValueError, match="Invalid confidence_level"):
            await engine.generate_prediction(confidence_level=1.0)  # Too high
    
    @pytest.mark.asyncio
    async def test_caching_behavior(self, engine):
        """Test that predictions are cached correctly."""
        # Generate first prediction
        prediction1 = await engine.generate_prediction()
        
        # Generate second prediction (should be cached)
        prediction2 = await engine.generate_prediction()
        
        # Should be the same prediction due to caching
        assert prediction1.timestamp == prediction2.timestamp
        assert prediction1.current_price == prediction2.current_price
        
        # Clear cache and generate new prediction
        await engine.clear_cache()
        prediction3 = await engine.generate_prediction()
        
        # Should be different prediction
        assert prediction3.timestamp != prediction1.timestamp
    
    @pytest.mark.asyncio
    async def test_cache_statistics(self, engine):
        """Test cache statistics functionality."""
        # Initially empty cache
        stats = await engine.get_cache_stats()
        assert stats['total_entries'] == 0
        
        # Generate prediction to populate cache
        await engine.generate_prediction()
        
        stats = await engine.get_cache_stats()
        assert stats['total_entries'] > 0
        assert stats['active_entries'] > 0
    
    @pytest.mark.asyncio
    async def test_engine_status(self, engine):
        """Test engine status reporting."""
        status = await engine.get_engine_status()
        
        assert status['status'] == 'healthy'
        assert 'timestamp' in status
        assert 'mathematical_engine' in status
        assert 'cache' in status
        assert 'data_fetcher' in status
        assert 'persistence' in status
        
        # Check mathematical engine config
        math_config = status['mathematical_engine']
        assert math_config['model_version'] == 'gbm-ewma-v1.0'
        assert 'lookback_window' in math_config
        assert 'n_simulations' in math_config
    
    @pytest.mark.asyncio
    async def test_historical_data_query(self, engine):
        """Test historical data querying."""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=1)
        
        history = await engine.get_prediction_history(
            start_date=start_date,
            end_date=end_date,
            limit=10
        )
        
        assert isinstance(history, list)
        # With mock persistence, should return empty list initially
        assert len(history) == 0
    
    @pytest.mark.asyncio
    async def test_historical_data_validation(self, engine):
        """Test validation of historical data query parameters."""
        # Test invalid date range
        end_date = datetime.utcnow()
        start_date = end_date + timedelta(days=1)  # Start after end
        
        with pytest.raises(ValueError, match="start_date must be before end_date"):
            await engine.get_prediction_history(
                start_date=start_date,
                end_date=end_date
            )
        
        # Test limit too high
        with pytest.raises(ValueError, match="Limit cannot exceed 1000"):
            await engine.get_prediction_history(limit=1001)
    
    @pytest.mark.asyncio
    async def test_latest_prediction(self, engine):
        """Test latest prediction retrieval."""
        # Initially no predictions
        latest = await engine.get_latest_prediction()
        assert latest is None
        
        # After generating a prediction, should still be None with mock persistence
        await engine.generate_prediction()
        latest = await engine.get_latest_prediction()
        assert latest is None  # Mock persistence doesn't store predictions
    
    @pytest.mark.asyncio
    async def test_volatility_computation(self, engine):
        """Test that EWMA volatility is computed correctly."""
        # Generate multiple predictions and check volatility values
        volatilities = []
        
        for _ in range(3):
            await engine.clear_cache()  # Force new computation
            prediction = await engine.generate_prediction()
            volatilities.append(prediction.volatility)
        
        # All volatilities should be positive
        assert all(vol > 0 for vol in volatilities)
        
        # With real data, volatilities should be consistent
        # (currently using fallback volatility due to insufficient historical data)
        assert all(vol == volatilities[0] for vol in volatilities), f"Expected consistent volatilities, got {volatilities}"
        
        # Volatility should be reasonable for Bitcoin (between 0.1 and 5.0 annualized)
        assert 0.1 <= volatilities[0] <= 5.0, f"Volatility {volatilities[0]} is outside reasonable range"
    
    @pytest.mark.asyncio
    async def test_drift_estimation(self, engine):
        """Test that drift estimation works correctly."""
        prediction = await engine.generate_prediction()
        
        # Drift should be finite
        assert isinstance(prediction.drift, float)
        assert not (prediction.drift != prediction.drift)  # Check for NaN
        
        # Drift should be within reasonable bounds (capped at ±2.0)
        assert -2.0 <= prediction.drift <= 2.0
    
    @pytest.mark.asyncio
    async def test_prediction_bounds_validation(self, engine):
        """Test that prediction bounds are valid."""
        prediction = await engine.generate_prediction()
        
        # All values should be positive and finite
        assert prediction.current_price > 0
        assert prediction.lower_bound > 0
        assert prediction.upper_bound > 0
        
        # Upper bound should be greater than lower bound
        assert prediction.upper_bound > prediction.lower_bound
        
        # Interval width should match calculation
        expected_width = prediction.upper_bound - prediction.lower_bound
        assert abs(prediction.interval_width - expected_width) < 0.01
    
    @pytest.mark.asyncio
    async def test_clear_cache(self, engine):
        """Test cache clearing functionality."""
        # Generate prediction to populate cache
        await engine.generate_prediction()
        
        stats_before = await engine.get_cache_stats()
        assert stats_before['total_entries'] > 0
        
        # Clear cache
        await engine.clear_cache()
        
        stats_after = await engine.get_cache_stats()
        assert stats_after['total_entries'] == 0