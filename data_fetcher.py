"""
Data ingestion functions for fetching Bitcoin price data from Binance API.

This module handles fetching historical and live Bitcoin price data with proper
error handling and data validation.
"""

import requests
import pandas as pd
from datetime import datetime, timezone
import time
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def fetch_binance_data(limit: int = 1000, symbol: str = "BTCUSDT", interval: str = "1h") -> pd.DataFrame:
    """
    Fetch historical Bitcoin price data from Binance API.
    
    Args:
        limit: Number of klines to fetch (max 1000)
        symbol: Trading symbol (default BTCUSDT)
        interval: Kline interval (default 1h)
    
    Returns:
        DataFrame with columns: timestamp, open, high, low, close, volume
    
    Raises:
        Exception: If API request fails or data is invalid
    """
    url = "https://data-api.binance.vision/api/v3/klines"
    
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": min(limit, 1000)  # Binance API limit
    }
    
    logger.info(f"Fetching {limit} {interval} bars for {symbol} from Binance API")
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("Invalid or empty response from Binance API")
        
        # Convert to DataFrame
        # Kline format: [open_time, open, high, low, close, volume, close_time, ...]
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Keep only the columns we need
        df = df[['open_time', 'open', 'high', 'low', 'close', 'volume']].copy()
        
        # Convert timestamps from milliseconds to datetime
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms', utc=True)
        
        # Convert price columns to float
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
        
        # Drop the original timestamp column
        df = df.drop('open_time', axis=1)
        
        # Reorder columns
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        
        # Sort by timestamp (should already be sorted, but ensure it)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Validate data
        if df.isnull().any().any():
            logger.warning("Some data points contain NaN values")
        
        # Check for reasonable price ranges
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if (df[col] <= 0).any():
                raise ValueError(f"Found non-positive prices in {col} column")
            
            if (df[col] > 1_000_000).any():
                raise ValueError(f"Found unreasonably high prices in {col} column")
        
        logger.info(
            f"Successfully fetched {len(df)} bars. "
            f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}"
        )
        
        return df
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching Binance data: {e}")
        raise Exception(f"Failed to fetch data from Binance API: {e}")
    
    except (ValueError, KeyError) as e:
        logger.error(f"Data parsing error: {e}")
        raise Exception(f"Failed to parse Binance API response: {e}")


def fetch_current_price(symbol: str = "BTCUSDT") -> float:
    """
    Fetch current Bitcoin price from Binance API.
    
    Args:
        symbol: Trading symbol (default BTCUSDT)
    
    Returns:
        Current price as float
    
    Raises:
        Exception: If API request fails
    """
    url = "https://data-api.binance.vision/api/v3/ticker/price"
    
    params = {"symbol": symbol}
    
    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        
        if "price" not in data:
            raise ValueError("Invalid response format from Binance price API")
        
        price = float(data["price"])
        
        if price <= 0 or price > 1_000_000:
            raise ValueError(f"Unreasonable price value: {price}")
        
        logger.info(f"Current {symbol} price: ${price:.2f}")
        
        return price
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Network error fetching current price: {e}")
        raise Exception(f"Failed to fetch current price: {e}")
    
    except (ValueError, KeyError) as e:
        logger.error(f"Price parsing error: {e}")
        raise Exception(f"Failed to parse current price: {e}")


def get_live_data_for_prediction(lookback_hours: int = 500) -> Dict[str, Any]:
    """
    Fetch live data suitable for making predictions.
    
    Args:
        lookback_hours: Number of hours of historical data to fetch
    
    Returns:
        Dictionary with current_price and historical_prices
    """
    # Fetch historical data
    df = fetch_binance_data(limit=lookback_hours + 1)  # +1 to account for current forming bar
    
    # Check if the last bar is still forming (current hour)
    now = datetime.now(timezone.utc)
    last_timestamp = df['timestamp'].iloc[-1]
    
    # If the last bar is from the current hour, drop it (it's still forming)
    if last_timestamp.hour == now.hour and last_timestamp.date() == now.date():
        logger.info("Dropping the current forming hourly bar")
        df = df.iloc[:-1]
    
    # Get current price separately
    current_price = fetch_current_price()
    
    # Extract historical closing prices
    historical_prices = df['close'].tolist()
    
    logger.info(
        f"Prepared live data: {len(historical_prices)} historical prices, "
        f"current price: ${current_price:.2f}"
    )
    
    return {
        'current_price': current_price,
        'historical_prices': historical_prices,
        'historical_df': df,
        'data_timestamp': now
    }