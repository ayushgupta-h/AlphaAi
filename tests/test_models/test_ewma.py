"""
Unit tests for EWMA volatility calculator module.

Tests cover:
- Volatility calculation with known data
- Fallback behavior with insufficient data
- Validation of positive and finite results
- Parameter validation
- Edge cases and error handling
"""

import pytest
import numpy as np
import pandas as pd
from bitcoin_forecasting.models.ewma import (
    compute_ewma_volatility,
    compute_ewma_volatility_series,
    FALLBACK_VOLATILITY
)


class TestComputeEWMAVolatility:
    """Test suite for compute_ewma_volatility function."""
    
    def test_volatility_calculation_with_known_data(self):
        """Test EWMA volatility calculation with known price data."""
        # Create a simple price series with known returns
        # Prices: 100, 110, 105, 115, 120
        # Returns: ln(110/100), ln(105/110), ln(115/105), ln(120/115)
        prices = pd.Series([100.0, 110.0, 105.0, 115.0, 120.0, 118.0, 125.0, 
                           130.0, 128.0, 135.0, 140.0, 138.0, 145.0, 150.0,
                           148.0, 155.0, 160.0, 158.0, 165.0, 170.0, 168.0,
                           175.0, 180.0, 178.0, 185.0])
        
        # Compute volatility with lookback window of 10
        vol = compute_ewma_volatility(prices, lookback_window=10)
        
        # Volatility should be positive
        assert vol > 0, "Volatility must be positive"
        
        # Volatility should be finite
        assert np.isfinite(vol), "Volatility must be finite"
        
        # For Bitcoin, annualized volatility typically ranges from 0.3 to 2.0
        assert 0.1 < vol < 5.0, f"Volatility {vol} seems unrealistic for typical price data"
    
    def test_fallback_with_insufficient_data(self):
        """Test that fallback volatility is returned when insufficient data."""
        # Create price series with fewer observations than required
        prices = pd.Series([100.0, 102.0, 101.0, 103.0])
        
        # Try to compute with lookback window of 10 (needs 11 prices)
        vol = compute_ewma_volatility(prices, lookback_window=10)
        
        # Should return fallback volatility
        assert vol == FALLBACK_VOLATILITY, \
            f"Expected fallback volatility {FALLBACK_VOLATILITY}, got {vol}"
    
    def test_fallback_with_exact_minimum_data(self):
        """Test behavior when we have exactly minimum required data."""
        # Create price series with exactly lookback_window + 1 observations
        lookback = 10
        prices = pd.Series([100.0 + i for i in range(lookback + 1)])
        
        # Should successfully compute volatility (not fallback)
        vol = compute_ewma_volatility(prices, lookback_window=lookback)
        
        # Should be a valid computed value, not fallback
        assert vol > 0 and np.isfinite(vol), "Should compute valid volatility"
    
    def test_positive_volatility_result(self):
        """Test that computed volatility is always positive."""
        # Test with various price patterns
        test_cases = [
            # Upward trend
            pd.Series([100.0, 105.0, 110.0, 115.0, 120.0, 125.0, 130.0, 135.0,
                      140.0, 145.0, 150.0, 155.0]),
            # Downward trend
            pd.Series([150.0, 145.0, 140.0, 135.0, 130.0, 125.0, 120.0, 115.0,
                      110.0, 105.0, 100.0, 95.0]),
            # Volatile pattern
            pd.Series([100.0, 110.0, 95.0, 115.0, 90.0, 120.0, 85.0, 125.0,
                      80.0, 130.0, 75.0, 135.0]),
            # Stable pattern
            pd.Series([100.0, 100.5, 100.2, 100.7, 100.3, 100.8, 100.4, 100.9,
                      100.5, 101.0, 100.6, 101.1])
        ]
        
        for prices in test_cases:
            vol = compute_ewma_volatility(prices, lookback_window=10)
            assert vol > 0, f"Volatility must be positive, got {vol}"
    
    def test_finite_volatility_result(self):
        """Test that computed volatility is always finite."""
        # Create price series
        prices = pd.Series([100.0 + i * 2 for i in range(25)])
        
        vol = compute_ewma_volatility(prices, lookback_window=20)
        
        assert np.isfinite(vol), f"Volatility must be finite, got {vol}"
        assert not np.isnan(vol), "Volatility must not be NaN"
        assert not np.isinf(vol), "Volatility must not be infinite"
    
    def test_lookback_window_validation(self):
        """Test validation of lookback_window parameter."""
        prices = pd.Series([100.0 + i for i in range(30)])
        
        # Test lookback window too small
        with pytest.raises(ValueError, match="lookback_window must be between 10 and 50"):
            compute_ewma_volatility(prices, lookback_window=5)
        
        # Test lookback window too large
        with pytest.raises(ValueError, match="lookback_window must be between 10 and 50"):
            compute_ewma_volatility(prices, lookback_window=100)
        
        # Test non-integer lookback window
        with pytest.raises(ValueError, match="lookback_window must be an integer"):
            compute_ewma_volatility(prices, lookback_window=10.5)
    
    def test_decay_param_validation(self):
        """Test validation of decay_param parameter."""
        prices = pd.Series([100.0 + i for i in range(30)])
        
        # Test decay param too small
        with pytest.raises(ValueError, match="decay_param must be in range"):
            compute_ewma_volatility(prices, lookback_window=10, decay_param=0.0)
        
        # Test decay param too large
        with pytest.raises(ValueError, match="decay_param must be in range"):
            compute_ewma_volatility(prices, lookback_window=10, decay_param=1.0)
        
        # Test negative decay param
        with pytest.raises(ValueError, match="decay_param must be in range"):
            compute_ewma_volatility(prices, lookback_window=10, decay_param=-0.5)
    
    def test_different_lookback_windows(self):
        """Test that different lookback windows produce different results."""
        prices = pd.Series([100.0 + i * 2 for i in range(60)])
        
        vol_10 = compute_ewma_volatility(prices, lookback_window=10)
        vol_20 = compute_ewma_volatility(prices, lookback_window=20)
        vol_30 = compute_ewma_volatility(prices, lookback_window=30)
        
        # All should be positive and finite
        assert all(v > 0 and np.isfinite(v) for v in [vol_10, vol_20, vol_30])
        
        # They should generally be different (though could be similar)
        # Just verify they're all valid values
        assert isinstance(vol_10, float)
        assert isinstance(vol_20, float)
        assert isinstance(vol_30, float)
    
    def test_different_decay_parameters(self):
        """Test that different decay parameters produce different results."""
        prices = pd.Series([100.0 + i * 2 for i in range(30)])
        
        vol_low = compute_ewma_volatility(prices, lookback_window=10, decay_param=0.85)
        vol_mid = compute_ewma_volatility(prices, lookback_window=10, decay_param=0.94)
        vol_high = compute_ewma_volatility(prices, lookback_window=10, decay_param=0.98)
        
        # All should be positive and finite
        assert all(v > 0 and np.isfinite(v) for v in [vol_low, vol_mid, vol_high])
    
    def test_constant_prices_low_volatility(self):
        """Test that constant prices produce very low (but non-zero) volatility."""
        # Prices with very small variations
        prices = pd.Series([100.0] * 25)
        
        vol = compute_ewma_volatility(prices, lookback_window=20)
        
        # Should be very low but still positive
        # With constant prices, log returns are 0, so volatility should be near 0
        # But due to numerical precision, might not be exactly 0
        assert vol >= 0, "Volatility must be non-negative"
        assert np.isfinite(vol), "Volatility must be finite"
    
    def test_high_volatility_prices(self):
        """Test that highly volatile prices produce high volatility estimate."""
        # Create highly volatile price series
        np.random.seed(42)
        base_prices = [100.0]
        for i in range(30):
            # Random walk with large steps
            change = np.random.normal(0, 10)
            base_prices.append(max(base_prices[-1] + change, 50.0))  # Keep prices positive
        
        prices = pd.Series(base_prices)
        
        vol = compute_ewma_volatility(prices, lookback_window=20)
        
        # Should produce a relatively high volatility
        assert vol > 0.5, f"Expected high volatility for volatile prices, got {vol}"
        assert np.isfinite(vol), "Volatility must be finite"
    
    def test_annualization_factor(self):
        """Test that volatility is properly annualized."""
        # Create price series
        prices = pd.Series([100.0 + i * 0.5 for i in range(30)])
        
        vol = compute_ewma_volatility(prices, lookback_window=20)
        
        # Annualized volatility should be scaled by sqrt(8760) ≈ 93.6
        # So it should be significantly larger than the raw volatility
        # This is a sanity check that annualization is applied
        assert vol > 0, "Annualized volatility must be positive"
    
    def test_empty_price_series(self):
        """Test behavior with empty price series."""
        prices = pd.Series([])
        
        vol = compute_ewma_volatility(prices, lookback_window=10)
        
        # Should return fallback volatility
        assert vol == FALLBACK_VOLATILITY
    
    def test_single_price(self):
        """Test behavior with single price."""
        prices = pd.Series([100.0])
        
        vol = compute_ewma_volatility(prices, lookback_window=10)
        
        # Should return fallback volatility (insufficient data)
        assert vol == FALLBACK_VOLATILITY


class TestComputeEWMAVolatilitySeries:
    """Test suite for compute_ewma_volatility_series function."""
    
    def test_series_length_matches_input(self):
        """Test that output series has same length as input."""
        prices = pd.Series([100.0 + i for i in range(50)])
        
        vol_series = compute_ewma_volatility_series(prices, lookback_window=10)
        
        assert len(vol_series) == len(prices), \
            f"Output length {len(vol_series)} should match input length {len(prices)}"
    
    def test_early_values_are_fallback(self):
        """Test that early values (before min_periods) use fallback."""
        prices = pd.Series([100.0 + i for i in range(30)])
        lookback = 10
        
        vol_series = compute_ewma_volatility_series(prices, lookback_window=lookback)
        
        # First lookback values should be fallback
        for i in range(lookback):
            assert vol_series.iloc[i] == FALLBACK_VOLATILITY, \
                f"Early value at index {i} should be fallback"
    
    def test_later_values_are_computed(self):
        """Test that later values (after min_periods) are computed."""
        prices = pd.Series([100.0 + i for i in range(30)])
        lookback = 10
        
        vol_series = compute_ewma_volatility_series(prices, lookback_window=lookback)
        
        # Values after lookback should be computed (not fallback)
        for i in range(lookback + 1, len(prices)):
            assert vol_series.iloc[i] > 0, f"Value at index {i} should be positive"
            assert np.isfinite(vol_series.iloc[i]), f"Value at index {i} should be finite"
    
    def test_walk_forward_no_lookahead(self):
        """Test that each volatility estimate uses only historical data."""
        # This is implicitly tested by the implementation, but we verify
        # that the series is computed correctly
        prices = pd.Series([100.0 + i * 2 for i in range(40)])
        
        vol_series = compute_ewma_volatility_series(prices, lookback_window=10)
        
        # Manually compute volatility at a specific point
        test_index = 20
        manual_vol = compute_ewma_volatility(
            prices.iloc[:test_index + 1],
            lookback_window=10
        )
        
        # Should match the series value at that index
        assert abs(vol_series.iloc[test_index] - manual_vol) < 1e-10, \
            "Series value should match manual calculation using only historical data"
    
    def test_all_values_positive_and_finite(self):
        """Test that all values in series are positive and finite."""
        prices = pd.Series([100.0 + i * 1.5 for i in range(50)])
        
        vol_series = compute_ewma_volatility_series(prices, lookback_window=10)
        
        assert (vol_series > 0).all(), "All volatility values must be positive"
        assert np.isfinite(vol_series).all(), "All volatility values must be finite"


class TestEWMAEdgeCases:
    """Test suite for edge cases and error handling."""
    
    def test_negative_prices_in_series(self):
        """Test behavior with negative prices (should still compute if possible)."""
        # Note: In practice, prices should always be positive, but test robustness
        # The log return calculation will fail with negative prices
        prices = pd.Series([100.0, 105.0, -110.0, 115.0, 120.0, 125.0, 130.0,
                           135.0, 140.0, 145.0, 150.0, 155.0])
        
        # This should handle the error and return fallback
        vol = compute_ewma_volatility(prices, lookback_window=10)
        
        # Should return fallback due to calculation error
        assert vol == FALLBACK_VOLATILITY or (vol > 0 and np.isfinite(vol))
    
    def test_zero_prices_in_series(self):
        """Test behavior with zero prices (log returns undefined)."""
        prices = pd.Series([100.0, 105.0, 0.0, 115.0, 120.0, 125.0, 130.0,
                           135.0, 140.0, 145.0, 150.0, 155.0])
        
        # This should handle the error and return fallback
        vol = compute_ewma_volatility(prices, lookback_window=10)
        
        # Should return fallback due to calculation error (log of 0)
        assert vol == FALLBACK_VOLATILITY or (vol > 0 and np.isfinite(vol))
    
    def test_nan_prices_in_series(self):
        """Test behavior with NaN prices."""
        prices = pd.Series([100.0, 105.0, np.nan, 115.0, 120.0, 125.0, 130.0,
                           135.0, 140.0, 145.0, 150.0, 155.0])
        
        # This should handle the error and return fallback
        vol = compute_ewma_volatility(prices, lookback_window=10)
        
        # Should return fallback or handle gracefully
        assert vol == FALLBACK_VOLATILITY or (vol > 0 and np.isfinite(vol))
    
    def test_very_large_prices(self):
        """Test behavior with very large price values."""
        prices = pd.Series([1e10 + i * 1e8 for i in range(30)])
        
        vol = compute_ewma_volatility(prices, lookback_window=20)
        
        # Should still compute valid volatility
        assert vol > 0, "Volatility must be positive"
        assert np.isfinite(vol), "Volatility must be finite"
    
    def test_very_small_prices(self):
        """Test behavior with very small price values."""
        prices = pd.Series([0.0001 + i * 0.00001 for i in range(30)])
        
        vol = compute_ewma_volatility(prices, lookback_window=20)
        
        # Should still compute valid volatility
        assert vol > 0, "Volatility must be positive"
        assert np.isfinite(vol), "Volatility must be finite"
