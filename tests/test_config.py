"""
Unit tests for configuration validation module.

Tests valid configuration acceptance, invalid parameter rejection with
descriptive errors, and default value initialization.

**Validates: Requirements 13.5, 13.7**
"""

import pytest
from bitcoin_forecasting.config import (
    ForecastConfig,
    create_default_config,
    create_custom_config,
)


class TestDefaultConfiguration:
    """Test suite for default configuration initialization."""
    
    def test_default_config_values(self):
        """Test that default configuration has expected values."""
        config = ForecastConfig()
        
        assert config.lookback_window == 24
        assert config.degrees_of_freedom == 5.0
        assert config.n_simulations == 10000
        assert config.confidence_level == 0.95
    
    def test_create_default_config_function(self):
        """Test that create_default_config() returns valid configuration."""
        config = create_default_config()
        
        assert isinstance(config, ForecastConfig)
        assert config.lookback_window == 24
        assert config.degrees_of_freedom == 5.0
        assert config.n_simulations == 10000
        assert config.confidence_level == 0.95
    
    def test_default_config_passes_validation(self):
        """Test that default configuration passes all validation checks."""
        # Should not raise any exceptions
        config = ForecastConfig()
        
        # Verify derived values are correct
        assert abs(config.get_alpha() - 0.05) < 1e-10
        lower, upper = config.get_percentiles()
        assert abs(lower - 2.5) < 1e-10
        assert abs(upper - 97.5) < 1e-10


class TestValidConfiguration:
    """Test suite for valid configuration acceptance."""
    
    def test_valid_lookback_window_minimum(self):
        """Test that minimum valid lookback_window (10) is accepted."""
        config = ForecastConfig(lookback_window=10)
        assert config.lookback_window == 10
    
    def test_valid_lookback_window_maximum(self):
        """Test that maximum valid lookback_window (50) is accepted."""
        config = ForecastConfig(lookback_window=50)
        assert config.lookback_window == 50
    
    def test_valid_lookback_window_mid_range(self):
        """Test that mid-range lookback_window values are accepted."""
        config = ForecastConfig(lookback_window=30)
        assert config.lookback_window == 30
    
    def test_valid_degrees_of_freedom_minimum(self):
        """Test that degrees_of_freedom just above 2 is accepted."""
        config = ForecastConfig(degrees_of_freedom=2.1)
        assert config.degrees_of_freedom == 2.1
    
    def test_valid_degrees_of_freedom_maximum(self):
        """Test that maximum valid degrees_of_freedom (30) is accepted."""
        config = ForecastConfig(degrees_of_freedom=30)
        assert config.degrees_of_freedom == 30
    
    def test_valid_degrees_of_freedom_mid_range(self):
        """Test that mid-range degrees_of_freedom values are accepted."""
        config = ForecastConfig(degrees_of_freedom=10.5)
        assert config.degrees_of_freedom == 10.5
    
    def test_valid_n_simulations_minimum(self):
        """Test that minimum valid n_simulations (1000) is accepted."""
        config = ForecastConfig(n_simulations=1000)
        assert config.n_simulations == 1000
    
    def test_valid_n_simulations_maximum(self):
        """Test that maximum valid n_simulations (1,000,000) is accepted."""
        config = ForecastConfig(n_simulations=1000000)
        assert config.n_simulations == 1000000
    
    def test_valid_n_simulations_mid_range(self):
        """Test that mid-range n_simulations values are accepted."""
        config = ForecastConfig(n_simulations=50000)
        assert config.n_simulations == 50000
    
    def test_valid_confidence_level_minimum(self):
        """Test that confidence_level just above 0.5 is accepted."""
        config = ForecastConfig(confidence_level=0.51)
        assert config.confidence_level == 0.51
    
    def test_valid_confidence_level_maximum(self):
        """Test that confidence_level just below 1.0 is accepted."""
        config = ForecastConfig(confidence_level=0.999)
        assert config.confidence_level == 0.999
    
    def test_valid_confidence_level_standard_values(self):
        """Test that standard confidence levels (0.90, 0.95, 0.99) are accepted."""
        config_90 = ForecastConfig(confidence_level=0.90)
        config_95 = ForecastConfig(confidence_level=0.95)
        config_99 = ForecastConfig(confidence_level=0.99)
        
        assert config_90.confidence_level == 0.90
        assert config_95.confidence_level == 0.95
        assert config_99.confidence_level == 0.99
    
    def test_valid_custom_config_all_parameters(self):
        """Test that custom configuration with all parameters is accepted."""
        config = ForecastConfig(
            lookback_window=20,
            degrees_of_freedom=6.0,
            n_simulations=5000,
            confidence_level=0.90
        )
        
        assert config.lookback_window == 20
        assert config.degrees_of_freedom == 6.0
        assert config.n_simulations == 5000
        assert config.confidence_level == 0.90


class TestInvalidLookbackWindow:
    """Test suite for invalid lookback_window parameter rejection."""
    
    def test_lookback_window_too_small(self):
        """Test that lookback_window below 10 is rejected with descriptive error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(lookback_window=9)
        
        error_message = str(exc_info.value)
        assert "lookback_window must be at least 10 bars" in error_message
        assert "statistical stability" in error_message
        assert "9" in error_message
    
    def test_lookback_window_too_large(self):
        """Test that lookback_window above 50 is rejected with descriptive error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(lookback_window=51)
        
        error_message = str(exc_info.value)
        assert "lookback_window must be at most 50 bars" in error_message
        assert "responsiveness" in error_message
        assert "51" in error_message
    
    def test_lookback_window_zero(self):
        """Test that lookback_window of 0 is rejected."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(lookback_window=0)
        
        error_message = str(exc_info.value)
        assert "lookback_window must be at least 10 bars" in error_message
    
    def test_lookback_window_negative(self):
        """Test that negative lookback_window is rejected."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(lookback_window=-5)
        
        error_message = str(exc_info.value)
        assert "lookback_window must be at least 10 bars" in error_message
    
    def test_lookback_window_float(self):
        """Test that float lookback_window is rejected with type error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(lookback_window=24.5)
        
        error_message = str(exc_info.value)
        assert "lookback_window must be an integer" in error_message
        assert "float" in error_message
    
    def test_lookback_window_string(self):
        """Test that string lookback_window is rejected with type error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(lookback_window="24")
        
        error_message = str(exc_info.value)
        assert "lookback_window must be an integer" in error_message
        assert "str" in error_message


class TestInvalidDegreesOfFreedom:
    """Test suite for invalid degrees_of_freedom parameter rejection."""
    
    def test_degrees_of_freedom_too_small(self):
        """Test that degrees_of_freedom <= 2 is rejected with descriptive error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(degrees_of_freedom=2.0)
        
        error_message = str(exc_info.value)
        assert "degrees_of_freedom must be greater than 2" in error_message
        assert "finite variance" in error_message
        assert "2" in error_message
    
    def test_degrees_of_freedom_too_large(self):
        """Test that degrees_of_freedom above 30 is rejected with descriptive error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(degrees_of_freedom=31)
        
        error_message = str(exc_info.value)
        assert "degrees_of_freedom should be at most 30" in error_message
        assert "normal distribution" in error_message
        assert "31" in error_message
    
    def test_degrees_of_freedom_zero(self):
        """Test that degrees_of_freedom of 0 is rejected."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(degrees_of_freedom=0)
        
        error_message = str(exc_info.value)
        assert "degrees_of_freedom must be greater than 2" in error_message
    
    def test_degrees_of_freedom_negative(self):
        """Test that negative degrees_of_freedom is rejected."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(degrees_of_freedom=-3.5)
        
        error_message = str(exc_info.value)
        assert "degrees_of_freedom must be greater than 2" in error_message
    
    def test_degrees_of_freedom_string(self):
        """Test that string degrees_of_freedom is rejected with type error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(degrees_of_freedom="5")
        
        error_message = str(exc_info.value)
        assert "degrees_of_freedom must be numeric" in error_message
        assert "str" in error_message


class TestInvalidNSimulations:
    """Test suite for invalid n_simulations parameter rejection."""
    
    def test_n_simulations_too_small(self):
        """Test that n_simulations below 1000 is rejected with descriptive error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(n_simulations=999)
        
        error_message = str(exc_info.value)
        assert "n_simulations must be at least 1000" in error_message
        assert "stable percentile estimates" in error_message
        assert "999" in error_message
    
    def test_n_simulations_too_large(self):
        """Test that n_simulations above 1,000,000 is rejected with descriptive error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(n_simulations=1000001)
        
        error_message = str(exc_info.value)
        assert "n_simulations must be at most 1,000,000" in error_message
        assert "excessive computation" in error_message
        assert "1000001" in error_message
    
    def test_n_simulations_zero(self):
        """Test that n_simulations of 0 is rejected."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(n_simulations=0)
        
        error_message = str(exc_info.value)
        assert "n_simulations must be at least 1000" in error_message
    
    def test_n_simulations_negative(self):
        """Test that negative n_simulations is rejected."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(n_simulations=-5000)
        
        error_message = str(exc_info.value)
        assert "n_simulations must be at least 1000" in error_message
    
    def test_n_simulations_float(self):
        """Test that float n_simulations is rejected with type error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(n_simulations=10000.5)
        
        error_message = str(exc_info.value)
        assert "n_simulations must be an integer" in error_message
        assert "float" in error_message
    
    def test_n_simulations_string(self):
        """Test that string n_simulations is rejected with type error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(n_simulations="10000")
        
        error_message = str(exc_info.value)
        assert "n_simulations must be an integer" in error_message
        assert "str" in error_message


class TestInvalidConfidenceLevel:
    """Test suite for invalid confidence_level parameter rejection."""
    
    def test_confidence_level_too_small(self):
        """Test that confidence_level <= 0.5 is rejected with descriptive error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(confidence_level=0.5)
        
        error_message = str(exc_info.value)
        assert "confidence_level must be greater than 0.5" in error_message
        assert "0.5" in error_message
    
    def test_confidence_level_too_large(self):
        """Test that confidence_level >= 1.0 is rejected with descriptive error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(confidence_level=1.0)
        
        error_message = str(exc_info.value)
        assert "confidence_level must be less than 1.0" in error_message
        assert "1.0" in error_message
    
    def test_confidence_level_above_one(self):
        """Test that confidence_level > 1.0 is rejected."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(confidence_level=1.5)
        
        error_message = str(exc_info.value)
        assert "confidence_level must be less than 1.0" in error_message
    
    def test_confidence_level_zero(self):
        """Test that confidence_level of 0 is rejected."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(confidence_level=0.0)
        
        error_message = str(exc_info.value)
        assert "confidence_level must be greater than 0.5" in error_message
    
    def test_confidence_level_negative(self):
        """Test that negative confidence_level is rejected."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(confidence_level=-0.95)
        
        error_message = str(exc_info.value)
        assert "confidence_level must be greater than 0.5" in error_message
    
    def test_confidence_level_string(self):
        """Test that string confidence_level is rejected with type error."""
        with pytest.raises(ValueError) as exc_info:
            ForecastConfig(confidence_level="0.95")
        
        error_message = str(exc_info.value)
        assert "confidence_level must be numeric" in error_message
        assert "str" in error_message


class TestCreateCustomConfig:
    """Test suite for create_custom_config function."""
    
    def test_create_custom_config_no_parameters(self):
        """Test that create_custom_config with no parameters returns defaults."""
        config = create_custom_config()
        
        assert config.lookback_window == 24
        assert config.degrees_of_freedom == 5.0
        assert config.n_simulations == 10000
        assert config.confidence_level == 0.95
    
    def test_create_custom_config_partial_parameters(self):
        """Test that create_custom_config with partial parameters uses defaults for others."""
        config = create_custom_config(lookback_window=20, n_simulations=5000)
        
        assert config.lookback_window == 20
        assert config.degrees_of_freedom == 5.0  # default
        assert config.n_simulations == 5000
        assert config.confidence_level == 0.95  # default
    
    def test_create_custom_config_all_parameters(self):
        """Test that create_custom_config with all parameters uses provided values."""
        config = create_custom_config(
            lookback_window=15,
            degrees_of_freedom=7.5,
            n_simulations=20000,
            confidence_level=0.99
        )
        
        assert config.lookback_window == 15
        assert config.degrees_of_freedom == 7.5
        assert config.n_simulations == 20000
        assert config.confidence_level == 0.99
    
    def test_create_custom_config_validates_parameters(self):
        """Test that create_custom_config validates parameters."""
        with pytest.raises(ValueError) as exc_info:
            create_custom_config(lookback_window=5)
        
        error_message = str(exc_info.value)
        assert "lookback_window must be at least 10 bars" in error_message


class TestConfigurationHelperMethods:
    """Test suite for configuration helper methods."""
    
    def test_get_alpha_default(self):
        """Test that get_alpha returns correct value for default confidence level."""
        config = ForecastConfig()
        assert abs(config.get_alpha() - 0.05) < 1e-10
    
    def test_get_alpha_custom(self):
        """Test that get_alpha returns correct value for custom confidence level."""
        config = ForecastConfig(confidence_level=0.90)
        assert abs(config.get_alpha() - 0.10) < 1e-10
    
    def test_get_percentiles_default(self):
        """Test that get_percentiles returns correct values for default confidence level."""
        config = ForecastConfig()
        lower, upper = config.get_percentiles()
        
        assert abs(lower - 2.5) < 1e-10
        assert abs(upper - 97.5) < 1e-10
    
    def test_get_percentiles_custom_90(self):
        """Test that get_percentiles returns correct values for 90% confidence level."""
        config = ForecastConfig(confidence_level=0.90)
        lower, upper = config.get_percentiles()
        
        assert abs(lower - 5.0) < 1e-10
        assert abs(upper - 95.0) < 1e-10
    
    def test_get_percentiles_custom_99(self):
        """Test that get_percentiles returns correct values for 99% confidence level."""
        config = ForecastConfig(confidence_level=0.99)
        lower, upper = config.get_percentiles()
        
        assert abs(lower - 0.5) < 1e-10
        assert abs(upper - 99.5) < 1e-10
    
    def test_repr_includes_all_parameters(self):
        """Test that __repr__ includes all configuration parameters."""
        config = ForecastConfig()
        repr_str = repr(config)
        
        assert "ForecastConfig" in repr_str
        assert "lookback_window=24" in repr_str
        assert "degrees_of_freedom=5" in repr_str
        assert "n_simulations=10,000" in repr_str
        assert "confidence_level=0.95" in repr_str
        assert "alpha=" in repr_str
        assert "percentiles=" in repr_str
