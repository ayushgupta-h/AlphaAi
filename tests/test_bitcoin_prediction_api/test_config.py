"""
Unit tests for Bitcoin Prediction API configuration.

This module tests the configuration management system including
environment variable loading and validation.
"""

import pytest
import os
from unittest.mock import patch

from bitcoin_prediction_api.config import Settings

class TestSettings:
    """Test cases for Settings configuration class."""
    
    def test_default_configuration(self):
        """Test that default configuration values are set correctly."""
        settings = Settings()
        
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
        assert settings.debug is False
        assert settings.default_confidence_level == 0.95
        assert settings.lookback_window == 24
        assert settings.n_simulations == 10000
        assert settings.log_level == "INFO"
    
    def test_environment_variable_override(self):
        """Test that environment variables override default values."""
        with patch.dict(os.environ, {
            'DEBUG': 'true',
            'PORT': '9000',
            'LOG_LEVEL': 'DEBUG',
            'RATE_LIMIT_PER_MINUTE': '120'
        }):
            settings = Settings()
            
            assert settings.debug is True
            assert settings.port == 9000
            assert settings.log_level == "DEBUG"
            assert settings.rate_limit_per_minute == 120
    
    def test_configuration_validation_valid(self):
        """Test that valid configuration passes validation."""
        settings = Settings(
            default_confidence_level=0.95,
            lookback_window=24,
            n_simulations=10000,
            degrees_of_freedom=5.0
        )
        
        # Should not raise any exception
        settings.validate_configuration()
    
    def test_configuration_validation_invalid_confidence_level(self):
        """Test that invalid confidence level fails validation."""
        settings = Settings(default_confidence_level=1.5)
        
        with pytest.raises(ValueError, match="default_confidence_level must be between"):
            settings.validate_configuration()
    
    def test_configuration_validation_invalid_lookback_window(self):
        """Test that invalid lookback window fails validation."""
        settings = Settings(lookback_window=0)
        
        with pytest.raises(ValueError, match="lookback_window must be positive"):
            settings.validate_configuration()
    
    def test_configuration_validation_invalid_simulations(self):
        """Test that invalid simulation count fails validation."""
        settings = Settings(n_simulations=500)
        
        with pytest.raises(ValueError, match="n_simulations must be at least 1000"):
            settings.validate_configuration()
    
    def test_configuration_validation_invalid_degrees_of_freedom(self):
        """Test that invalid degrees of freedom fails validation."""
        settings = Settings(degrees_of_freedom=-1.0)
        
        with pytest.raises(ValueError, match="degrees_of_freedom must be positive"):
            settings.validate_configuration()