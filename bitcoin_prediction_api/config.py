"""
Configuration management for Bitcoin Prediction API.

This module provides centralized configuration management using Pydantic settings
with environment variable support for all service parameters.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
import os

class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    
    # Database Configuration
    mongodb_url: str = "mongodb://localhost:27017/bitcoin_predictions"
    mongodb_database: str = "bitcoin_predictions"
    
    # Cache Configuration
    redis_url: Optional[str] = None
    cache_ttl_seconds: int = 60
    
    # External API Configuration
    binance_api_url: str = "https://data-api.binance.vision"
    coingecko_api_url: str = "https://api.coingecko.com/api/v3"
    api_timeout_seconds: int = 5
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Mathematical Model Configuration
    lookback_window: int = 24
    degrees_of_freedom: float = 5.0
    n_simulations: int = 10000
    default_confidence_level: float = 0.95
    
    # Logging Configuration
    log_level: str = "INFO"
    log_format: str = "json"
        
    def validate_configuration(self) -> None:
        """Validate configuration parameters at startup."""
        if self.default_confidence_level < 0.5 or self.default_confidence_level >= 1.0:
            raise ValueError("default_confidence_level must be between 0.5 and 0.999")
        
        if self.lookback_window < 1:
            raise ValueError("lookback_window must be positive")
        
        if self.n_simulations < 1000:
            raise ValueError("n_simulations must be at least 1000")
        
        if self.degrees_of_freedom <= 0:
            raise ValueError("degrees_of_freedom must be positive")

# Global settings instance
settings = Settings()

# Validate configuration on import
settings.validate_configuration()