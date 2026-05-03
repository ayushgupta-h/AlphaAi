"""
Data ingestion module for fetching Bitcoin price data from Binance API.

This module provides functionality to fetch historical BTCUSDT price data
from the Binance public API and parse it into a structured format.
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Callable, TypeVar
from functools import wraps
import requests
import pandas as pd

logger = logging.getLogger(__name__)

# Type variable for generic function return type
T = TypeVar("T")


def retry_with_exponential_backoff(
    max_attempts: int = 3, initial_delay: float = 1.0, backoff_factor: float = 2.0
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """
    Decorator that implements retry logic with exponential backoff.

    This decorator will retry a function up to max_attempts times if it raises
    a requests.exceptions.RequestException. Between retries, it waits for an
    exponentially increasing delay: initial_delay, initial_delay * backoff_factor,
    initial_delay * backoff_factor^2, etc.

    Parameters
    ----------
    max_attempts : int, optional
        Maximum number of attempts (default: 3)
    initial_delay : float, optional
        Initial delay in seconds before first retry (default: 1.0)
    backoff_factor : float, optional
        Multiplier for delay between retries (default: 2.0)

    Returns
    -------
    Callable
        Decorated function with retry logic

    Raises
    ------
    requests.exceptions.RequestException
        If all retry attempts fail, raises the last exception with descriptive message

    Examples
    --------
    >>> @retry_with_exponential_backoff(max_attempts=3, initial_delay=1.0)
    ... def fetch_data():
    ...     response = requests.get("https://api.example.com/data")
    ...     response.raise_for_status()
    ...     return response.json()
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    last_exception = e

                    if attempt == max_attempts:
                        # All retries exhausted
                        error_msg = (
                            f"Failed to fetch data after {max_attempts} attempts. "
                            f"Last error: {type(e).__name__}: {str(e)}"
                        )
                        logger.error(error_msg)
                        raise requests.exceptions.RequestException(error_msg) from e

                    # Calculate delay for this retry
                    delay = initial_delay * (backoff_factor ** (attempt - 1))

                    logger.warning(
                        f"Attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.1f} seconds..."
                    )

                    time.sleep(delay)

            # This should never be reached, but for type safety
            if last_exception:
                raise last_exception

        return wrapper

    return decorator


@retry_with_exponential_backoff(max_attempts=3, initial_delay=1.0, backoff_factor=2.0)
def fetch_binance_data(
    symbol: str = "BTCUSDT", interval: str = "1h", limit: int = 1000
) -> pd.DataFrame:
    """
    Fetch historical Bitcoin price data from Binance API.

    This function retrieves OHLCV (Open, High, Low, Close, Volume) data
    from the Binance public API endpoint and parses it into a pandas DataFrame.

    Parameters
    ----------
    symbol : str, optional
        Trading pair symbol (default: "BTCUSDT")
    interval : str, optional
        Candlestick interval (default: "1h" for hourly data)
    limit : int, optional
        Number of data points to fetch (default: 1000, max: 1000)

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, open, high, low, close, volume
        - timestamp: datetime object representing the candlestick open time
        - open: opening price (float)
        - high: highest price (float)
        - low: lowest price (float)
        - close: closing price (float)
        - volume: trading volume (float)

    Raises
    ------
    requests.exceptions.RequestException
        If the API request fails after all retry attempts
    ValueError
        If the API response is invalid or cannot be parsed

    Examples
    --------
    >>> df = fetch_binance_data()
    >>> print(df.head())
    """
    # Binance API endpoint for historical klines (candlestick) data
    endpoint = "https://data-api.binance.vision/api/v3/klines"

    # Prepare request parameters
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    logger.info(f"Fetching {limit} {interval} bars for {symbol} from Binance API")

    try:
        # Make HTTP GET request to Binance API
        response = requests.get(endpoint, params=params, timeout=30)
        response.raise_for_status()  # Raise exception for HTTP errors

        # Parse JSON response
        data = response.json()

        if not isinstance(data, list):
            raise ValueError(f"Expected list response from API, got {type(data)}")

        if len(data) == 0:
            raise ValueError("API returned empty data")

        logger.info(f"Successfully fetched {len(data)} bars from Binance API")

        # Parse the response into structured format
        parsed_data = _parse_binance_response(data)

        # Apply data validation and preprocessing
        validated_data = _validate_and_preprocess_data(parsed_data)

        return validated_data

    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch data from Binance API: {e}")
        raise
    except (ValueError, KeyError) as e:
        logger.error(f"Failed to parse Binance API response: {e}")
        raise


def _parse_binance_response(data: List[List[Any]]) -> pd.DataFrame:
    """
    Parse Binance API response into a structured DataFrame.

    Binance klines API returns data in the following format:
    [
        [
            1499040000000,      // Open time (milliseconds)
            "0.01634000",       // Open price
            "0.80000000",       // High price
            "0.01575800",       // Low price
            "0.01577100",       // Close price
            "148976.11427815",  // Volume
            1499644799999,      // Close time
            "2434.19055334",    // Quote asset volume
            308,                // Number of trades
            "1756.87402397",    // Taker buy base asset volume
            "28.46694368",      // Taker buy quote asset volume
            "17928899.62484339" // Ignore
        ]
    ]

    Parameters
    ----------
    data : List[List[Any]]
        Raw response data from Binance API

    Returns
    -------
    pd.DataFrame
        Parsed DataFrame with timestamp, open, high, low, close, volume columns

    Raises
    ------
    ValueError
        If data format is invalid or cannot be parsed
    """
    parsed_records = []

    for record in data:
        if len(record) < 6:
            raise ValueError(
                f"Invalid record format: expected at least 6 fields, got {len(record)}"
            )

        try:
            # Extract relevant fields from the record
            # Index 0: Open time in milliseconds
            # Index 1: Open price
            # Index 2: High price
            # Index 3: Low price
            # Index 4: Close price
            # Index 5: Volume

            # Convert timestamp from milliseconds to datetime object
            timestamp_ms = int(record[0])
            timestamp = datetime.fromtimestamp(timestamp_ms / 1000.0)

            # Parse price and volume fields as floats
            open_price = float(record[1])
            high_price = float(record[2])
            low_price = float(record[3])
            close_price = float(record[4])
            volume = float(record[5])

            parsed_records.append(
                {
                    "timestamp": timestamp,
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume": volume,
                }
            )

        except (ValueError, IndexError) as e:
            logger.warning(f"Skipping invalid record: {e}")
            continue

    if len(parsed_records) == 0:
        raise ValueError("No valid records could be parsed from API response")

    # Create DataFrame from parsed records
    df = pd.DataFrame(parsed_records)

    logger.info(f"Parsed {len(df)} records into structured format")

    return df


def _validate_and_preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and preprocess Bitcoin price data.

    This function performs the following validations and preprocessing steps:
    1. Validates minimum 1000 hourly bars received
    2. Sorts data by timestamp in ascending order
    3. Validates all prices are positive
    4. Validates timestamps are in chronological order

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: timestamp, open, high, low, close, volume

    Returns
    -------
    pd.DataFrame
        Validated and preprocessed DataFrame sorted by timestamp

    Raises
    ------
    ValueError
        If any validation check fails
    """
    # Requirement 1.6: Validate minimum 1000 hourly bars received
    if len(df) < 1000:
        raise ValueError(
            f"Insufficient data: expected at least 1000 hourly bars, got {len(df)}"
        )

    logger.info(f"Validation: Received {len(df)} hourly bars (minimum 1000 required)")

    # Requirement 1.8: Sort data by timestamp in ascending order
    df = df.sort_values(by="timestamp", ascending=True).reset_index(drop=True)
    logger.info("Preprocessing: Sorted data by timestamp in ascending order")

    # Requirement 15.1: Validate all prices are positive
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        if (df[col] <= 0).any():
            invalid_count = (df[col] <= 0).sum()
            raise ValueError(
                f"Invalid price data: found {invalid_count} non-positive values in '{col}' column. "
                f"All prices must be positive."
            )

    logger.info("Validation: All prices are positive")

    # Requirement 15.2: Validate timestamps are in chronological order
    # After sorting, check that each timestamp is greater than or equal to the previous
    timestamps = df["timestamp"].values
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            raise ValueError(
                f"Timestamps are not in chronological order: "
                f"timestamp at index {i} ({timestamps[i]}) is before "
                f"timestamp at index {i-1} ({timestamps[i-1]})"
            )

    logger.info("Validation: Timestamps are in chronological order")

    return df
