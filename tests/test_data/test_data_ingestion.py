"""
Unit tests for data ingestion module.

Tests cover:
- Successful API response parsing
- Retry logic with mock failures
- Exponential backoff timing
- Error handling after all retries fail
"""

import pytest
from unittest.mock import Mock, patch, call
import requests
import pandas as pd
from datetime import datetime
import time

from bitcoin_forecasting.data.data_ingestion import (
    fetch_binance_data,
    retry_with_exponential_backoff,
    _parse_binance_response,
    _validate_and_preprocess_data
)


class TestRetryLogic:
    """Test suite for retry logic with exponential backoff."""
    
    def test_retry_decorator_success_on_first_attempt(self):
        """Test that decorator returns immediately on successful first attempt."""
        mock_func = Mock(return_value="success")
        decorated = retry_with_exponential_backoff(max_attempts=3)(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 1
    
    def test_retry_decorator_success_on_second_attempt(self):
        """Test that decorator retries once and succeeds on second attempt."""
        mock_func = Mock(side_effect=[
            requests.exceptions.RequestException("First failure"),
            "success"
        ])
        decorated = retry_with_exponential_backoff(max_attempts=3, initial_delay=0.1)(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 2
    
    def test_retry_decorator_success_on_third_attempt(self):
        """Test that decorator retries twice and succeeds on third attempt."""
        mock_func = Mock(side_effect=[
            requests.exceptions.RequestException("First failure"),
            requests.exceptions.RequestException("Second failure"),
            "success"
        ])
        decorated = retry_with_exponential_backoff(max_attempts=3, initial_delay=0.1)(mock_func)
        
        result = decorated()
        
        assert result == "success"
        assert mock_func.call_count == 3
    
    def test_retry_decorator_fails_after_max_attempts(self):
        """Test that decorator raises error after all retry attempts fail."""
        mock_func = Mock(side_effect=requests.exceptions.RequestException("Persistent failure"))
        decorated = retry_with_exponential_backoff(max_attempts=3, initial_delay=0.1)(mock_func)
        
        with pytest.raises(requests.exceptions.RequestException) as exc_info:
            decorated()
        
        assert "Failed to fetch data after 3 attempts" in str(exc_info.value)
        assert "Persistent failure" in str(exc_info.value)
        assert mock_func.call_count == 3
    
    def test_retry_decorator_exponential_backoff_timing(self):
        """Test that decorator implements exponential backoff with correct delays."""
        mock_func = Mock(side_effect=[
            requests.exceptions.RequestException("Failure 1"),
            requests.exceptions.RequestException("Failure 2"),
            "success"
        ])
        decorated = retry_with_exponential_backoff(
            max_attempts=3,
            initial_delay=0.1,
            backoff_factor=2.0
        )(mock_func)
        
        start_time = time.time()
        result = decorated()
        elapsed_time = time.time() - start_time
        
        # Expected delays: 0.1s (after 1st failure) + 0.2s (after 2nd failure) = 0.3s
        # Allow some tolerance for execution time
        assert result == "success"
        assert elapsed_time >= 0.3
        assert elapsed_time < 0.5  # Should not take too long
    
    def test_retry_decorator_with_different_exception_types(self):
        """Test that decorator only retries on RequestException, not other exceptions."""
        mock_func = Mock(side_effect=ValueError("Not a request exception"))
        decorated = retry_with_exponential_backoff(max_attempts=3, initial_delay=0.1)(mock_func)
        
        # ValueError should not be caught by retry logic
        with pytest.raises(ValueError) as exc_info:
            decorated()
        
        assert "Not a request exception" in str(exc_info.value)
        assert mock_func.call_count == 1  # Should not retry


class TestFetchBinanceData:
    """Test suite for fetch_binance_data function."""
    
    @patch('bitcoin_forecasting.data.data_ingestion.requests.get')
    def test_successful_fetch(self, mock_get):
        """Test successful data fetch from Binance API."""
        # Mock successful API response with 1000 records
        mock_response = Mock()
        mock_response.status_code = 200
        
        # Generate 1000 mock records
        base_timestamp = 1609459200000  # 2021-01-01 00:00:00
        mock_data = []
        for i in range(1000):
            timestamp_ms = base_timestamp + i * 3600000  # Hourly intervals
            mock_data.append([
                timestamp_ms,
                f"{29000.00 + i}",
                f"{29500.00 + i}",
                f"{28800.00 + i}",
                f"{29200.00 + i}",
                "1000.5"
            ])
        
        mock_response.json.return_value = mock_data
        mock_get.return_value = mock_response
        
        df = fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        assert list(df.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert df.iloc[0]["close"] == 29200.00
        assert df.iloc[999]["close"] == 30199.00
    
    @patch('bitcoin_forecasting.data.data_ingestion.requests.get')
    def test_fetch_with_retry_on_transient_failure(self, mock_get):
        """Test that fetch retries on transient network failure."""
        # First call fails, second succeeds
        mock_response_success = Mock()
        mock_response_success.status_code = 200
        
        # Generate 1000 mock records
        base_timestamp = 1609459200000
        mock_data = []
        for i in range(1000):
            timestamp_ms = base_timestamp + i * 3600000
            mock_data.append([
                timestamp_ms,
                f"{29000.00 + i}",
                f"{29500.00 + i}",
                f"{28800.00 + i}",
                f"{29200.00 + i}",
                "1000.5"
            ])
        
        mock_response_success.json.return_value = mock_data
        
        mock_get.side_effect = [
            requests.exceptions.ConnectionError("Network error"),
            mock_response_success
        ]
        
        df = fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1000
        assert mock_get.call_count == 2
    
    @patch('bitcoin_forecasting.data.data_ingestion.requests.get')
    def test_fetch_fails_after_all_retries(self, mock_get):
        """Test that fetch raises error after all retry attempts fail."""
        mock_get.side_effect = requests.exceptions.ConnectionError("Persistent network error")
        
        with pytest.raises(requests.exceptions.RequestException) as exc_info:
            fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1)
        
        assert "Failed to fetch data after 3 attempts" in str(exc_info.value)
        assert mock_get.call_count == 3
    
    @patch('bitcoin_forecasting.data.data_ingestion.requests.get')
    def test_fetch_with_http_error(self, mock_get):
        """Test that fetch handles HTTP errors (4xx, 5xx) correctly."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_response
        
        with pytest.raises(requests.exceptions.RequestException):
            fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1)
        
        assert mock_get.call_count == 3  # Should retry 3 times
    
    @patch('bitcoin_forecasting.data.data_ingestion.requests.get')
    def test_fetch_with_invalid_json_response(self, mock_get):
        """Test that fetch handles invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"error": "Invalid response format"}
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError) as exc_info:
            fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1)
        
        assert "Expected list response from API" in str(exc_info.value)
    
    @patch('bitcoin_forecasting.data.data_ingestion.requests.get')
    def test_fetch_with_empty_response(self, mock_get):
        """Test that fetch handles empty API response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = []
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError) as exc_info:
            fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1)
        
        assert "API returned empty data" in str(exc_info.value)


class TestParseBinanceResponse:
    """Test suite for _parse_binance_response function."""
    
    def test_parse_valid_response(self):
        """Test parsing of valid Binance API response."""
        data = [
            [1609459200000, "29000.00", "29500.00", "28800.00", "29200.00", "1000.5"],
            [1609462800000, "29200.00", "29600.00", "29100.00", "29400.00", "1200.3"]
        ]
        
        df = _parse_binance_response(data)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert df.iloc[0]["open"] == 29000.00
        assert df.iloc[0]["high"] == 29500.00
        assert df.iloc[0]["low"] == 28800.00
        assert df.iloc[0]["close"] == 29200.00
        assert df.iloc[0]["volume"] == 1000.5
        assert isinstance(df.iloc[0]["timestamp"], datetime)
    
    def test_parse_with_invalid_record_format(self):
        """Test that parser raises error for records with insufficient fields."""
        data = [
            [1609459200000, "29000.00", "29500.00"]  # Only 3 fields, need at least 6
        ]
        
        with pytest.raises(ValueError) as exc_info:
            _parse_binance_response(data)
        
        assert "Invalid record format" in str(exc_info.value)
    
    def test_parse_with_invalid_price_values(self):
        """Test that parser skips records with invalid price values."""
        data = [
            [1609459200000, "invalid", "29500.00", "28800.00", "29200.00", "1000.5"],
            [1609462800000, "29200.00", "29600.00", "29100.00", "29400.00", "1200.3"]
        ]
        
        df = _parse_binance_response(data)
        
        # Should skip first record and only parse second
        assert len(df) == 1
        assert df.iloc[0]["close"] == 29400.00
    
    def test_parse_with_all_invalid_records(self):
        """Test that parser raises error when no valid records can be parsed."""
        data = [
            [1609459200000, "invalid", "invalid", "invalid", "invalid", "invalid"]
        ]
        
        with pytest.raises(ValueError) as exc_info:
            _parse_binance_response(data)
        
        assert "No valid records could be parsed" in str(exc_info.value)
    
    def test_parse_timestamp_conversion(self):
        """Test that timestamps are correctly converted from milliseconds to datetime."""
        data = [
            [1609459200000, "29000.00", "29500.00", "28800.00", "29200.00", "1000.5"]
        ]
        
        df = _parse_binance_response(data)
        
        timestamp = df.iloc[0]["timestamp"]
        assert isinstance(timestamp, datetime)
        # 1609459200000 ms = 2021-01-01 00:00:00 UTC
        assert timestamp.year == 2021
        assert timestamp.month == 1
        assert timestamp.day == 1


class TestValidateAndPreprocessData:
    """Test suite for _validate_and_preprocess_data function."""
    
    def test_validate_minimum_bars_success(self):
        """Test that validation passes with exactly 1000 bars."""
        # Create DataFrame with exactly 1000 records
        data = []
        base_timestamp = 1609459200000  # 2021-01-01 00:00:00
        for i in range(1000):
            timestamp = datetime.fromtimestamp((base_timestamp + i * 3600000) / 1000.0)
            data.append({
                "timestamp": timestamp,
                "open": 29000.0 + i,
                "high": 29500.0 + i,
                "low": 28800.0 + i,
                "close": 29200.0 + i,
                "volume": 1000.5
            })
        df = pd.DataFrame(data)
        
        result = _validate_and_preprocess_data(df)
        
        assert len(result) == 1000
        assert isinstance(result, pd.DataFrame)
    
    def test_validate_minimum_bars_failure(self):
        """Test that validation fails with fewer than 1000 bars."""
        # Create DataFrame with only 999 records
        data = []
        base_timestamp = 1609459200000
        for i in range(999):
            timestamp = datetime.fromtimestamp((base_timestamp + i * 3600000) / 1000.0)
            data.append({
                "timestamp": timestamp,
                "open": 29000.0,
                "high": 29500.0,
                "low": 28800.0,
                "close": 29200.0,
                "volume": 1000.5
            })
        df = pd.DataFrame(data)
        
        with pytest.raises(ValueError) as exc_info:
            _validate_and_preprocess_data(df)
        
        assert "Insufficient data" in str(exc_info.value)
        assert "expected at least 1000 hourly bars, got 999" in str(exc_info.value)
    
    def test_sort_by_timestamp_ascending(self):
        """Test that data is sorted by timestamp in ascending order."""
        # Create DataFrame with timestamps in random order
        data = []
        base_timestamp = 1609459200000  # 2021-01-01 00:00:00
        
        # Create 1000 records with timestamps in random order
        timestamps_ms = [base_timestamp + i * 3600000 for i in range(1000)]
        # Shuffle the first 3 to test sorting
        shuffled_indices = [2, 0, 1] + list(range(3, 1000))
        
        for idx in shuffled_indices:
            timestamp = datetime.fromtimestamp(timestamps_ms[idx] / 1000.0)
            data.append({
                "timestamp": timestamp,
                "open": 29000.0 + idx,
                "high": 29500.0 + idx,
                "low": 28800.0 + idx,
                "close": 29200.0 + idx,
                "volume": 1000.5
            })
        
        df = pd.DataFrame(data)
        
        result = _validate_and_preprocess_data(df)
        
        # Check that timestamps are sorted
        timestamps = result["timestamp"].tolist()
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1], f"Timestamp at index {i} is not >= timestamp at index {i-1}"
        
        # Verify the first three are now in correct order
        assert result.iloc[0]["close"] == 29200.0  # Was index 0
        assert result.iloc[1]["close"] == 29201.0  # Was index 1
        assert result.iloc[2]["close"] == 29202.0  # Was index 2
    
    def test_validate_positive_prices_success(self):
        """Test that validation passes when all prices are positive."""
        data = []
        base_timestamp = 1609459200000
        for i in range(1000):
            timestamp = datetime.fromtimestamp((base_timestamp + i * 3600000) / 1000.0)
            data.append({
                "timestamp": timestamp,
                "open": 29000.0,
                "high": 29500.0,
                "low": 28800.0,
                "close": 29200.0,
                "volume": 1000.5
            })
        df = pd.DataFrame(data)
        
        result = _validate_and_preprocess_data(df)
        
        assert (result["open"] > 0).all()
        assert (result["high"] > 0).all()
        assert (result["low"] > 0).all()
        assert (result["close"] > 0).all()
    
    def test_validate_positive_prices_failure_zero(self):
        """Test that validation fails when prices are zero."""
        data = []
        base_timestamp = 1609459200000
        for i in range(1000):
            timestamp = datetime.fromtimestamp((base_timestamp + i * 3600000) / 1000.0)
            data.append({
                "timestamp": timestamp,
                "open": 29000.0 if i != 500 else 0.0,  # One zero price
                "high": 29500.0,
                "low": 28800.0,
                "close": 29200.0,
                "volume": 1000.5
            })
        df = pd.DataFrame(data)
        
        with pytest.raises(ValueError) as exc_info:
            _validate_and_preprocess_data(df)
        
        assert "Invalid price data" in str(exc_info.value)
        assert "non-positive values in 'open' column" in str(exc_info.value)
    
    def test_validate_positive_prices_failure_negative(self):
        """Test that validation fails when prices are negative."""
        data = []
        base_timestamp = 1609459200000
        for i in range(1000):
            timestamp = datetime.fromtimestamp((base_timestamp + i * 3600000) / 1000.0)
            data.append({
                "timestamp": timestamp,
                "open": 29000.0,
                "high": 29500.0,
                "low": 28800.0,
                "close": 29200.0 if i != 100 else -100.0,  # One negative price
                "volume": 1000.5
            })
        df = pd.DataFrame(data)
        
        with pytest.raises(ValueError) as exc_info:
            _validate_and_preprocess_data(df)
        
        assert "Invalid price data" in str(exc_info.value)
        assert "non-positive values in 'close' column" in str(exc_info.value)
    
    def test_validate_chronological_order_success(self):
        """Test that validation passes when timestamps are in chronological order."""
        data = []
        base_timestamp = 1609459200000
        for i in range(1000):
            timestamp = datetime.fromtimestamp((base_timestamp + i * 3600000) / 1000.0)
            data.append({
                "timestamp": timestamp,
                "open": 29000.0,
                "high": 29500.0,
                "low": 28800.0,
                "close": 29200.0,
                "volume": 1000.5
            })
        df = pd.DataFrame(data)
        
        result = _validate_and_preprocess_data(df)
        
        # Verify timestamps are strictly increasing
        timestamps = result["timestamp"].values
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i-1]
    
    def test_validate_chronological_order_with_equal_timestamps(self):
        """Test that validation passes when some timestamps are equal (edge case)."""
        data = []
        base_timestamp = 1609459200000
        for i in range(1000):
            # Create some duplicate timestamps
            timestamp = datetime.fromtimestamp((base_timestamp + (i // 2) * 3600000) / 1000.0)
            data.append({
                "timestamp": timestamp,
                "open": 29000.0,
                "high": 29500.0,
                "low": 28800.0,
                "close": 29200.0,
                "volume": 1000.5
            })
        df = pd.DataFrame(data)
        
        # Should not raise error - equal timestamps are allowed
        result = _validate_and_preprocess_data(df)
        
        assert len(result) == 1000
