"""
Real-time data fetcher for Bitcoin Prediction API.

This module implements the RealTimeDataFetcher class that fetches current Bitcoin
prices and historical data from live cryptocurrency APIs (Binance primary, CoinGecko backup)
with comprehensive retry logic and error handling.
"""

import asyncio
import logging
import math
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any

import httpx
from pydantic import ValidationError

from ..config import settings
from ..models.prediction import PriceData

logger = logging.getLogger(__name__)


class APIError(Exception):
    """Custom exception for API-related errors."""
    pass


class ValidationError(Exception):
    """Custom exception for data validation errors."""
    pass


class RealTimeDataFetcher:
    """
    Fetches current Bitcoin prices from live cryptocurrency APIs.
    
    This class implements the real-time data fetching functionality with:
    - Primary source: Binance API
    - Backup source: CoinGecko API (future implementation)
    - Exponential backoff retry logic (3 attempts: 1s, 2s, 4s delays)
    - 5-second timeout per request
    - Price validation and caching
    """
    
    def __init__(self):
        """Initialize the data fetcher with HTTP client and configuration."""
        self.binance_base_url = settings.binance_api_url
        self.coingecko_base_url = settings.coingecko_api_url
        self.timeout = settings.api_timeout_seconds
        self.cache_ttl = settings.cache_ttl_seconds
        
        # Price cache to avoid excessive API calls
        self._price_cache: Optional[PriceData] = None
        self._cache_timestamp: Optional[float] = None
        
        # HTTP client configuration
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout),
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
        
        logger.info(
            "RealTimeDataFetcher initialized",
            extra={
                "binance_url": self.binance_base_url,
                "timeout": self.timeout,
                "cache_ttl": self.cache_ttl
            }
        )
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
    
    async def fetch_current_price(self) -> PriceData:
        """
        Fetch current BTC price with retry logic and validation.
        
        Returns:
            PriceData: Current Bitcoin price data with timestamp and source
            
        Raises:
            APIError: When all retry attempts fail
            ValidationError: When price data is invalid
        """
        # Check cache first
        if self._is_cache_valid():
            logger.debug("Returning cached price data")
            return self._price_cache
        
        logger.info("Fetching current Bitcoin price from Binance API")
        
        # Retry configuration: 3 attempts with exponential backoff (1s, 2s, 4s)
        max_retries = 3
        base_delay = 1.0
        
        last_error = None
        
        for attempt in range(max_retries + 1):  # +1 for initial attempt
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))  # 1s, 2s, 4s
                    logger.info(
                        f"Retrying price fetch after {delay}s delay",
                        extra={"attempt": attempt, "delay": delay}
                    )
                    await asyncio.sleep(delay)
                
                price_data = await self._fetch_binance_price()
                
                # Validate the fetched price
                if not self.validate_price_data(price_data.price):
                    raise ValidationError(f"Invalid price data: {price_data.price}")
                
                # Cache the result
                self._cache_price_data(price_data)
                
                logger.info(
                    "Successfully fetched Bitcoin price",
                    extra={
                        "price": price_data.price,
                        "source": price_data.source,
                        "attempt": attempt
                    }
                )
                
                return price_data
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Price fetch attempt {attempt + 1} failed: {str(e)}",
                    extra={
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
                
                # If this was the last attempt, break
                if attempt == max_retries:
                    break
        
        # All retries exhausted
        error_msg = f"Failed to fetch Bitcoin price after {max_retries + 1} attempts"
        if last_error:
            error_msg += f". Last error: {str(last_error)}"
        
        logger.error(
            "All price fetch attempts failed",
            extra={
                "total_attempts": max_retries + 1,
                "last_error": str(last_error) if last_error else None
            }
        )
        
        raise APIError(error_msg)
    
    async def fetch_historical_prices(self, limit: int = 60) -> List[float]:
        """
        Fetch recent hourly closes for volatility calculation.
        
        Args:
            limit: Number of hourly closes to fetch (default 60 for ~2.5 days)
            
        Returns:
            List[float]: List of historical hourly closing prices
            
        Raises:
            APIError: When API request fails
            ValidationError: When historical data is invalid
        """
        logger.info(f"Fetching {limit} hourly historical prices from Binance API")
        
        # Retry configuration: 3 attempts with exponential backoff
        max_retries = 3
        base_delay = 1.0
        
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.info(
                        f"Retrying historical data fetch after {delay}s delay",
                        extra={"attempt": attempt, "delay": delay}
                    )
                    await asyncio.sleep(delay)
                
                historical_prices = await self._fetch_binance_klines(limit)
                
                # Validate historical data
                if not historical_prices:
                    raise ValidationError("No historical data received")
                
                if len(historical_prices) < min(limit, 10):  # At least 10 data points
                    raise ValidationError(f"Insufficient historical data: got {len(historical_prices)}, need at least 10")
                
                # Validate all prices are positive
                invalid_prices = [p for p in historical_prices if not self.validate_price_data(p)]
                if invalid_prices:
                    raise ValidationError(f"Invalid historical prices found: {invalid_prices[:5]}")
                
                logger.info(
                    "Successfully fetched historical prices",
                    extra={
                        "count": len(historical_prices),
                        "min_price": min(historical_prices),
                        "max_price": max(historical_prices),
                        "attempt": attempt
                    }
                )
                
                return historical_prices
                
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Historical data fetch attempt {attempt + 1} failed: {str(e)}",
                    extra={
                        "attempt": attempt + 1,
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
                
                if attempt == max_retries:
                    break
        
        # All retries exhausted
        error_msg = f"Failed to fetch historical prices after {max_retries + 1} attempts"
        if last_error:
            error_msg += f". Last error: {str(last_error)}"
        
        logger.error(
            "All historical data fetch attempts failed",
            extra={
                "total_attempts": max_retries + 1,
                "last_error": str(last_error) if last_error else None
            }
        )
        
        raise APIError(error_msg)
    
    def validate_price_data(self, price: float) -> bool:
        """
        Validate price is positive and within reasonable bounds.
        
        Args:
            price: Price value to validate
            
        Returns:
            bool: True if price is valid, False otherwise
        """
        if not isinstance(price, (int, float)):
            return False
        
        if not math.isfinite(price):
            return False
        
        if price <= 0:
            return False
        
        # Reasonable bounds for Bitcoin price (1000 to 1,000,000 USD)
        if not (1000 <= price <= 1_000_000):
            return False
        
        return True
    
    async def _fetch_binance_price(self) -> PriceData:
        """
        Fetch current price from Binance API.
        
        Returns:
            PriceData: Current price data from Binance
            
        Raises:
            APIError: When API request fails
        """
        url = f"{self.binance_base_url}/api/v3/ticker/price"
        params = {"symbol": "BTCUSDT"}
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Validate response structure
            if "price" not in data:
                raise APIError(f"Invalid Binance API response: missing 'price' field")
            
            price = float(data["price"])
            timestamp = datetime.now(timezone.utc)
            
            return PriceData(
                price=price,
                timestamp=timestamp,
                source="binance"
            )
            
        except httpx.HTTPStatusError as e:
            raise APIError(f"Binance API HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise APIError(f"Binance API request error: {str(e)}")
        except (ValueError, KeyError) as e:
            raise APIError(f"Binance API response parsing error: {str(e)}")
    
    async def _fetch_binance_klines(self, limit: int) -> List[float]:
        """
        Fetch historical klines (candlestick data) from Binance API.
        
        Args:
            limit: Number of klines to fetch
            
        Returns:
            List[float]: List of closing prices
            
        Raises:
            APIError: When API request fails
        """
        url = f"{self.binance_base_url}/api/v3/klines"
        params = {
            "symbol": "BTCUSDT",
            "interval": "1h",  # 1-hour intervals
            "limit": min(limit, 1000)  # Binance API limit
        }
        
        try:
            response = await self.client.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            if not isinstance(data, list):
                raise APIError("Invalid Binance klines response: expected list")
            
            if not data:
                raise APIError("Empty Binance klines response")
            
            # Extract closing prices (index 4 in kline data)
            # Kline format: [open_time, open, high, low, close, volume, close_time, ...]
            closing_prices = []
            for kline in data:
                if not isinstance(kline, list) or len(kline) < 5:
                    raise APIError(f"Invalid kline format: {kline}")
                
                close_price = float(kline[4])  # Closing price
                closing_prices.append(close_price)
            
            return closing_prices
            
        except httpx.HTTPStatusError as e:
            raise APIError(f"Binance klines API HTTP error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            raise APIError(f"Binance klines API request error: {str(e)}")
        except (ValueError, KeyError, IndexError) as e:
            raise APIError(f"Binance klines API response parsing error: {str(e)}")
    
    def _is_cache_valid(self) -> bool:
        """
        Check if cached price data is still valid.
        
        Returns:
            bool: True if cache is valid, False otherwise
        """
        if self._price_cache is None or self._cache_timestamp is None:
            return False
        
        cache_age = time.time() - self._cache_timestamp
        return cache_age < self.cache_ttl
    
    def _cache_price_data(self, price_data: PriceData) -> None:
        """
        Cache price data with timestamp.
        
        Args:
            price_data: Price data to cache
        """
        self._price_cache = price_data
        self._cache_timestamp = time.time()
        
        logger.debug(
            "Cached price data",
            extra={
                "price": price_data.price,
                "cache_ttl": self.cache_ttl
            }
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on external APIs.
        
        Returns:
            Dict[str, Any]: Health status of external APIs
        """
        health_status = {
            "binance": {"status": "unknown", "response_time_ms": None, "error": None},
            "overall_status": "unknown"
        }
        
        # Test Binance API
        start_time = time.time()
        try:
            await self._fetch_binance_price()
            response_time = (time.time() - start_time) * 1000
            health_status["binance"] = {
                "status": "up",
                "response_time_ms": round(response_time, 2),
                "error": None
            }
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            health_status["binance"] = {
                "status": "down",
                "response_time_ms": round(response_time, 2),
                "error": str(e)
            }
        
        # Determine overall status
        if health_status["binance"]["status"] == "up":
            health_status["overall_status"] = "healthy"
        else:
            health_status["overall_status"] = "degraded"
        
        return health_status