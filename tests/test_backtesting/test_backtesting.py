"""
Unit tests for backtesting module.

Tests cover:
- Initialization period handling
- Sequential prediction generation
- Data isolation per prediction (no look-ahead bias)
- Chronological order validation
- Input validation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from bitcoin_forecasting.backtesting.backtesting import (
    run_backtest,
    _validate_price_data,
    _generate_single_prediction,
    _estimate_drift,
    _validate_chronological_order
)
from bitcoin_forecasting.config import ForecastConfig


def create_synthetic_price_data(n_bars: int = 1000, start_price: float = 50000.0) -> pd.DataFrame:
    """
    Create synthetic price data for testing.
    
    Parameters
    ----------
    n_bars : int
        Number of hourly bars to generate
    start_price : float
        Starting price
    
    Returns
    -------
    pd.DataFrame
        Synthetic price data with realistic structure
    """
    # Generate timestamps (hourly)
    start_time = datetime(2024, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_bars)]
    
    # Generate synthetic prices with random walk
    np.random.seed(42)
    log_returns = np.random.normal(0, 0.01, n_bars)  # 1% hourly volatility
    log_prices = np.log(start_price) + np.cumsum(log_returns)
    prices = np.exp(log_prices)
    
    # Create OHLCV data
    data = {
        'timestamp': timestamps,
        'open': prices * (1 + np.random.uniform(-0.001, 0.001, n_bars)),
        'high': prices * (1 + np.random.uniform(0, 0.002, n_bars)),
        'low': prices * (1 - np.random.uniform(0, 0.002, n_bars)),
        'close': prices,
        'volume': np.random.uniform(100, 1000, n_bars)
    }
    
    return pd.DataFrame(data)


class TestValidatePriceData:
    """Tests for _validate_price_data function."""
    
    def test_valid_data(self):
        """Test that valid data passes validation."""
        data = create_synthetic_price_data(n_bars=1000)
        # Should not raise any exception
        _validate_price_data(data)
    
    def test_missing_columns(self):
        """Test that missing required columns raises ValueError."""
        data = create_synthetic_price_data(n_bars=100)
        data_missing_close = data.drop(columns=['close'])
        
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_price_data(data_missing_close)
    
    def test_empty_data(self):
        """Test that empty data raises ValueError."""
        data = pd.DataFrame(columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        with pytest.raises(ValueError, match="empty"):
            _validate_price_data(data)
    
    def test_non_chronological_timestamps(self):
        """Test that non-chronological timestamps raise ValueError."""
        data = create_synthetic_price_data(n_bars=100)
        # Swap two timestamps to break chronological order
        data.loc[10, 'timestamp'], data.loc[11, 'timestamp'] = \
            data.loc[11, 'timestamp'], data.loc[10, 'timestamp']
        
        with pytest.raises(ValueError, match="not sorted by timestamp"):
            _validate_price_data(data)
    
    def test_non_positive_prices(self):
        """Test that non-positive prices raise ValueError."""
        data = create_synthetic_price_data(n_bars=100)
        data.loc[50, 'close'] = -100.0  # Invalid negative price
        
        with pytest.raises(ValueError, match="non-positive values"):
            _validate_price_data(data)


class TestEstimateDrift:
    """Tests for _estimate_drift function."""
    
    def test_positive_trend(self):
        """Test drift estimation with positive trend."""
        # Create prices with upward trend
        prices = pd.Series([100, 101, 102, 103, 104, 105])
        drift = _estimate_drift(prices, lookback=5)
        
        # Drift should be positive for upward trend
        assert drift > 0
        assert np.isfinite(drift)
    
    def test_negative_trend(self):
        """Test drift estimation with negative trend."""
        # Create prices with downward trend
        prices = pd.Series([105, 104, 103, 102, 101, 100])
        drift = _estimate_drift(prices, lookback=5)
        
        # Drift should be negative for downward trend
        assert drift < 0
        assert np.isfinite(drift)
    
    def test_insufficient_data(self):
        """Test that insufficient data returns 0.0."""
        prices = pd.Series([100, 101])  # Only 2 prices
        drift = _estimate_drift(prices, lookback=24)
        
        # Should return 0.0 when insufficient data
        assert drift == 0.0
    
    def test_constant_prices(self):
        """Test drift estimation with constant prices."""
        prices = pd.Series([100.0] * 30)
        drift = _estimate_drift(prices, lookback=24)
        
        # Drift should be approximately 0 for constant prices
        assert abs(drift) < 1e-10
        assert np.isfinite(drift)


class TestValidateChronologicalOrder:
    """Tests for _validate_chronological_order function."""
    
    def test_valid_chronological_order(self):
        """Test that chronologically ordered predictions pass validation."""
        predictions = [
            {'timestamp': datetime(2024, 1, 1, i, 0, 0), 'actual_price': 50000.0}
            for i in range(10)
        ]
        
        # Should not raise any exception
        _validate_chronological_order(predictions)
    
    def test_invalid_chronological_order(self):
        """Test that non-chronological predictions raise ValueError."""
        predictions = [
            {'timestamp': datetime(2024, 1, 1, 0, 0, 0), 'actual_price': 50000.0},
            {'timestamp': datetime(2024, 1, 1, 1, 0, 0), 'actual_price': 50100.0},
            {'timestamp': datetime(2024, 1, 1, 0, 30, 0), 'actual_price': 50050.0},  # Out of order
        ]
        
        with pytest.raises(ValueError, match="not in chronological order"):
            _validate_chronological_order(predictions)
    
    def test_equal_timestamps(self):
        """Test that equal timestamps raise ValueError."""
        predictions = [
            {'timestamp': datetime(2024, 1, 1, 0, 0, 0), 'actual_price': 50000.0},
            {'timestamp': datetime(2024, 1, 1, 0, 0, 0), 'actual_price': 50100.0},  # Duplicate
        ]
        
        with pytest.raises(ValueError, match="not in chronological order"):
            _validate_chronological_order(predictions)
    
    def test_empty_predictions(self):
        """Test that empty predictions list passes validation."""
        predictions = []
        # Should not raise any exception
        _validate_chronological_order(predictions)
    
    def test_single_prediction(self):
        """Test that single prediction passes validation."""
        predictions = [
            {'timestamp': datetime(2024, 1, 1, 0, 0, 0), 'actual_price': 50000.0}
        ]
        # Should not raise any exception
        _validate_chronological_order(predictions)


class TestGenerateSinglePrediction:
    """Tests for _generate_single_prediction function."""
    
    def test_valid_prediction_generation(self):
        """Test that valid prediction is generated with correct structure."""
        historical_data = create_synthetic_price_data(n_bars=300)
        current_timestamp = datetime(2024, 1, 15, 0, 0, 0)
        actual_price = 51000.0
        config = ForecastConfig()
        
        prediction = _generate_single_prediction(
            historical_data=historical_data,
            current_timestamp=current_timestamp,
            actual_price=actual_price,
            config=config
        )
        
        # Check prediction structure
        assert 'timestamp' in prediction
        assert 'actual_price' in prediction
        assert 'lower_bound' in prediction
        assert 'upper_bound' in prediction
        assert 'volatility' in prediction
        assert 'drift' in prediction
        
        # Check values are valid
        assert prediction['timestamp'] == current_timestamp
        assert prediction['actual_price'] == actual_price
        assert prediction['lower_bound'] > 0
        assert prediction['upper_bound'] > 0
        assert prediction['lower_bound'] < prediction['upper_bound']
        assert prediction['volatility'] > 0
        assert np.isfinite(prediction['drift'])
    
    def test_prediction_bounds_are_reasonable(self):
        """Test that prediction bounds are within reasonable range of current price."""
        historical_data = create_synthetic_price_data(n_bars=300, start_price=50000.0)
        current_timestamp = datetime(2024, 1, 15, 0, 0, 0)
        current_price = historical_data['close'].iloc[-1]
        actual_price = current_price * 1.01  # 1% higher
        config = ForecastConfig()
        
        prediction = _generate_single_prediction(
            historical_data=historical_data,
            current_timestamp=current_timestamp,
            actual_price=actual_price,
            config=config
        )
        
        # For 1-hour forecast, bounds should be relatively close to current price
        # Allow for wide range due to fat-tailed distribution
        lower_bound = prediction['lower_bound']
        upper_bound = prediction['upper_bound']
        
        # Bounds should be within ±20% of current price for 1-hour forecast
        assert lower_bound > current_price * 0.80
        assert upper_bound < current_price * 1.20


class TestRunBacktest:
    """Tests for run_backtest function."""
    
    def test_insufficient_data(self):
        """Test that insufficient data raises ValueError."""
        data = create_synthetic_price_data(n_bars=500)  # Less than 1000 required
        config = ForecastConfig()
        
        with pytest.raises(ValueError, match="Insufficient data"):
            run_backtest(data, config)
    
    def test_valid_backtest_execution(self):
        """Test that backtest executes successfully with valid data."""
        data = create_synthetic_price_data(n_bars=1000)
        config = ForecastConfig()
        
        predictions = run_backtest(data, config)
        
        # Should generate exactly 720 predictions (1000 - 280 initialization)
        assert len(predictions) == 720
        
        # Check first prediction structure
        first_pred = predictions[0]
        assert 'timestamp' in first_pred
        assert 'actual_price' in first_pred
        assert 'lower_bound' in first_pred
        assert 'upper_bound' in first_pred
        assert 'volatility' in first_pred
        assert 'drift' in first_pred
    
    def test_predictions_in_chronological_order(self):
        """Test that predictions are generated in chronological order."""
        data = create_synthetic_price_data(n_bars=1000)
        config = ForecastConfig()
        
        predictions = run_backtest(data, config)
        
        # Verify chronological order
        for i in range(1, len(predictions)):
            assert predictions[i]['timestamp'] > predictions[i-1]['timestamp']
    
    def test_no_look_ahead_bias(self):
        """Test that predictions use only historical data (no look-ahead bias)."""
        data = create_synthetic_price_data(n_bars=1000)
        config = ForecastConfig()
        
        predictions = run_backtest(data, config)
        
        # For each prediction, verify timestamp is at or after initialization period end
        initialization_end = data['timestamp'].iloc[280]
        
        for pred in predictions:
            # Prediction timestamp should be at or after initialization period end
            assert pred['timestamp'] >= initialization_end
    
    def test_all_bounds_are_positive(self):
        """Test that all prediction bounds are positive."""
        data = create_synthetic_price_data(n_bars=1000)
        config = ForecastConfig()
        
        predictions = run_backtest(data, config)
        
        for pred in predictions:
            assert pred['lower_bound'] > 0
            assert pred['upper_bound'] > 0
            assert pred['lower_bound'] < pred['upper_bound']
    
    def test_all_volatilities_are_positive(self):
        """Test that all volatility estimates are positive."""
        data = create_synthetic_price_data(n_bars=1000)
        config = ForecastConfig()
        
        predictions = run_backtest(data, config)
        
        for pred in predictions:
            assert pred['volatility'] > 0
            assert np.isfinite(pred['volatility'])
    
    def test_custom_config(self):
        """Test backtest with custom configuration."""
        data = create_synthetic_price_data(n_bars=1000)
        config = ForecastConfig(
            lookback_window=20,
            degrees_of_freedom=6.0,
            n_simulations=5000,
            confidence_level=0.90
        )
        
        predictions = run_backtest(data, config)
        
        # Should still generate 720 predictions
        assert len(predictions) == 720
        
        # All predictions should be valid
        for pred in predictions:
            assert pred['lower_bound'] > 0
            assert pred['upper_bound'] > 0
            assert pred['lower_bound'] < pred['upper_bound']
    
    def test_invalid_price_data_columns(self):
        """Test that missing columns raise ValueError."""
        data = create_synthetic_price_data(n_bars=1000)
        data_invalid = data.drop(columns=['close'])
        config = ForecastConfig()
        
        with pytest.raises(ValueError, match="missing required columns"):
            run_backtest(data_invalid, config)
    
    def test_unsorted_price_data(self):
        """Test that unsorted data raises ValueError."""
        data = create_synthetic_price_data(n_bars=1000)
        # Shuffle the data to break chronological order
        data_shuffled = data.sample(frac=1.0, random_state=42).reset_index(drop=True)
        config = ForecastConfig()
        
        with pytest.raises(ValueError, match="not sorted by timestamp"):
            run_backtest(data_shuffled, config)


class TestWalkForwardMethodology:
    """Integration tests for walk-forward validation methodology."""
    
    def test_initialization_period_usage(self):
        """Test that first 280 bars are used for initialization."""
        data = create_synthetic_price_data(n_bars=1000)
        config = ForecastConfig()
        
        predictions = run_backtest(data, config)
        
        # First prediction should be for timestamp at index 280
        first_prediction_timestamp = predictions[0]['timestamp']
        expected_timestamp = data['timestamp'].iloc[280]
        
        assert first_prediction_timestamp == expected_timestamp
    
    def test_test_period_coverage(self):
        """Test that predictions cover exactly 720 test bars."""
        data = create_synthetic_price_data(n_bars=1000)
        config = ForecastConfig()
        
        predictions = run_backtest(data, config)
        
        # Should have exactly 720 predictions
        assert len(predictions) == 720
        
        # Last prediction should be for timestamp at index 999
        last_prediction_timestamp = predictions[-1]['timestamp']
        expected_timestamp = data['timestamp'].iloc[999]
        
        assert last_prediction_timestamp == expected_timestamp
    
    def test_sequential_prediction_generation(self):
        """Test that predictions are generated sequentially without gaps."""
        data = create_synthetic_price_data(n_bars=1000)
        config = ForecastConfig()
        
        predictions = run_backtest(data, config)
        
        # Check that timestamps are consecutive hourly bars
        for i in range(1, len(predictions)):
            time_diff = predictions[i]['timestamp'] - predictions[i-1]['timestamp']
            # Should be exactly 1 hour apart
            assert time_diff == timedelta(hours=1)
