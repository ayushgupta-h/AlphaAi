"""
Integration tests for mathematical engine service.

This module tests the integration of existing GBM engine components
with the API service, ensuring mathematical consistency with the
backtesting system.
"""

import pytest
import numpy as np
from unittest.mock import patch

from bitcoin_prediction_api.services.math_engine_service import MathEngineService
from bitcoin_forecasting.config import ForecastConfig


class TestMathEngineIntegration:
    """Test mathematical engine integration with existing components."""
    
    @pytest.fixture
    def math_engine(self):
        """Create MathEngineService instance for testing."""
        return MathEngineService()
    
    @pytest.fixture
    def sample_historical_prices(self):
        """Generate sample historical price data."""
        # Generate realistic Bitcoin price series
        base_price = 50000.0
        prices = []
        for i in range(25):  # 25 prices for 24 returns
            # Add some realistic price movement
            change = np.random.normal(0, 0.02)  # 2% hourly volatility
            if i == 0:
                prices.append(base_price)
            else:
                new_price = prices[-1] * (1 + change)
                prices.append(max(new_price, 1000.0))  # Ensure positive prices
        return prices
    
    def test_forecast_config_creation(self, math_engine):
        """Test that ForecastConfig is created correctly from API settings."""
        config = math_engine.config
        
        assert isinstance(config, ForecastConfig)
        assert config.lookback_window == 24  # From settings
        assert config.degrees_of_freedom == 5.0  # From settings
        assert config.n_simulations == 10000  # From settings
        assert config.confidence_level == 0.95  # From settings
    
    @pytest.mark.asyncio
    async def test_prediction_generation_success(self, math_engine, sample_historical_prices):
        """Test successful prediction generation with valid inputs."""
        current_price = 52000.0
        confidence_level = 0.95
        
        result = await math_engine.generate_prediction(
            current_price=current_price,
            historical_prices=sample_historical_prices,
            confidence_level=confidence_level
        )
        
        # Validate result structure
        assert hasattr(result, 'lower_bound')
        assert hasattr(result, 'upper_bound')
        assert hasattr(result, 'volatility')
        assert hasattr(result, 'drift')
        assert hasattr(result, 'terminal_prices')
        
        # Validate mathematical properties
        assert result.lower_bound > 0
        assert result.upper_bound > 0
        assert result.lower_bound < result.upper_bound
        assert result.volatility > 0
        assert np.isfinite(result.drift)
        
        # Validate simulation results
        assert len(result.terminal_prices) == 10000
        assert np.all(result.terminal_prices > 0)
        assert np.all(np.isfinite(result.terminal_prices))
    
    @pytest.mark.asyncio
    async def test_mathematical_consistency(self, math_engine, sample_historical_prices):
        """Test mathematical consistency with backtesting system."""
        current_price = 50000.0
        confidence_level = 0.95
        
        # Generate multiple predictions with same inputs
        results = []
        for _ in range(3):
            result = await math_engine.generate_prediction(
                current_price=current_price,
                historical_prices=sample_historical_prices,
                confidence_level=confidence_level
            )
            results.append(result)
        
        # Volatility should be identical (deterministic EWMA calculation)
        volatilities = [r.volatility for r in results]
        assert len(set(volatilities)) == 1, "EWMA volatility should be deterministic"
        
        # Drift should be identical (deterministic calculation)
        drifts = [r.drift for r in results]
        assert len(set(drifts)) == 1, "Drift estimation should be deterministic"
        
        # Bounds should vary (stochastic simulation) but be in reasonable range
        lower_bounds = [r.lower_bound for r in results]
        upper_bounds = [r.upper_bound for r in results]
        
        # Check that bounds vary (not identical due to randomness)
        assert len(set(lower_bounds)) > 1, "Lower bounds should vary due to randomness"
        assert len(set(upper_bounds)) > 1, "Upper bounds should vary due to randomness"
        
        # Check bounds are in reasonable range around current price
        for result in results:
            assert 0.5 * current_price < result.lower_bound < current_price
            assert current_price < result.upper_bound < 2.0 * current_price
    
    @pytest.mark.asyncio
    async def test_volatility_computation(self, math_engine):
        """Test EWMA volatility computation integration."""
        # Create price series with known volatility characteristics
        prices = [50000.0]
        for i in range(24):
            # Add 1% hourly moves (should result in high volatility)
            change = 0.01 if i % 2 == 0 else -0.01
            new_price = prices[-1] * (1 + change)
            prices.append(new_price)
        
        volatility = math_engine._compute_volatility(prices)
        
        # Validate volatility properties
        assert volatility > 0
        assert np.isfinite(volatility)
        assert 0.1 < volatility < 5.0  # Reasonable range for Bitcoin
    
    @pytest.mark.asyncio
    async def test_drift_estimation(self, math_engine):
        """Test drift estimation from price returns."""
        # Create trending price series
        prices = [50000.0]
        for i in range(24):
            # Add consistent 0.5% hourly upward trend
            new_price = prices[-1] * 1.005
            prices.append(new_price)
        
        drift = math_engine._estimate_drift(prices)
        
        # Should detect positive drift
        assert drift > 0
        assert np.isfinite(drift)
        assert -2.0 <= drift <= 2.0  # Within reasonable bounds
    
    @pytest.mark.asyncio
    async def test_input_validation(self, math_engine, sample_historical_prices):
        """Test input validation for prediction generation."""
        
        # Test invalid current price
        with pytest.raises(ValueError, match="current_price must be positive"):
            await math_engine.generate_prediction(
                current_price=-1000.0,
                historical_prices=sample_historical_prices,
                confidence_level=0.95
            )
        
        # Test insufficient historical data
        with pytest.raises(ValueError, match="Need at least 10 historical prices"):
            await math_engine.generate_prediction(
                current_price=50000.0,
                historical_prices=[50000.0, 51000.0],  # Only 2 prices
                confidence_level=0.95
            )
        
        # Test invalid confidence level
        with pytest.raises(ValueError, match="confidence_level must be between"):
            await math_engine.generate_prediction(
                current_price=50000.0,
                historical_prices=sample_historical_prices,
                confidence_level=1.5  # Invalid confidence level
            )
    
    @pytest.mark.asyncio
    async def test_output_validation(self, math_engine):
        """Test output validation catches invalid mathematical results."""
        # Create price series that might cause numerical issues
        prices = [50000.0] * 25  # Constant prices (zero volatility)
        
        # Should still produce valid output (fallback volatility)
        result = await math_engine.generate_prediction(
            current_price=50000.0,
            historical_prices=prices,
            confidence_level=0.95
        )
        
        # Validate all outputs are valid
        assert result.lower_bound > 0
        assert result.upper_bound > 0
        assert result.lower_bound < result.upper_bound
        assert result.volatility > 0
        assert np.isfinite(result.drift)
    
    def test_config_summary(self, math_engine):
        """Test configuration summary generation."""
        summary = math_engine.get_config_summary()
        
        expected_keys = {
            "lookback_window",
            "degrees_of_freedom", 
            "n_simulations",
            "default_confidence_level",
            "model_version"
        }
        
        assert set(summary.keys()) == expected_keys
        assert summary["model_version"] == "gbm-ewma-v1.0"
        assert summary["n_simulations"] == 10000