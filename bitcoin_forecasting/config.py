"""
Configuration management module for Bitcoin Probabilistic Forecasting System.

This module defines the configuration dataclass with parameter validation
for all model hyperparameters including EWMA lookback window, Student-t
degrees of freedom, Monte Carlo simulation count, and confidence level.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ForecastConfig:
    """
    Configuration parameters for the Bitcoin probabilistic forecasting system.

    Attributes:
        lookback_window: Number of hourly bars for EWMA volatility calculation.
                        Must be between 10 and 50 bars. Default: 24.
        degrees_of_freedom: Degrees of freedom for Student-t distribution.
                           Must be greater than 2 for finite variance. Default: 5.
        n_simulations: Number of Monte Carlo price paths to generate.
                      Must be at least 1000 for statistical stability. Default: 10000.
        confidence_level: Confidence level for prediction intervals.
                         Must be between 0.5 and 0.999. Default: 0.95.

    Raises:
        ValueError: If any parameter is outside its valid range.

    Example:
        >>> config = ForecastConfig()  # Use defaults
        >>> config = ForecastConfig(lookback_window=20, degrees_of_freedom=6)
    """

    lookback_window: int = 24
    degrees_of_freedom: float = 5.0
    n_simulations: int = 10000
    confidence_level: float = 0.95

    def __post_init__(self):
        """
        Validate configuration parameters after initialization.

        Raises:
            ValueError: If any parameter is outside its valid range with
                       descriptive error message.
        """
        self._validate_lookback_window()
        self._validate_degrees_of_freedom()
        self._validate_n_simulations()
        self._validate_confidence_level()

    def _validate_lookback_window(self) -> None:
        """
        Validate EWMA lookback window parameter.

        The lookback window must be between 10 and 50 hourly bars to balance
        responsiveness to recent volatility changes with statistical stability.

        Raises:
            ValueError: If lookback_window is not an integer or outside [10, 50].
        """
        if not isinstance(self.lookback_window, int):
            raise ValueError(
                f"lookback_window must be an integer, got {type(self.lookback_window).__name__}"
            )

        if self.lookback_window < 10:
            raise ValueError(
                f"lookback_window must be at least 10 bars for statistical stability, "
                f"got {self.lookback_window}"
            )

        if self.lookback_window > 50:
            raise ValueError(
                f"lookback_window must be at most 50 bars to maintain responsiveness, "
                f"got {self.lookback_window}"
            )

    def _validate_degrees_of_freedom(self) -> None:
        """
        Validate Student-t degrees of freedom parameter.

        Degrees of freedom must be greater than 2 to ensure finite variance.
        Lower values (3-6) produce fatter tails that better capture extreme
        price movements in cryptocurrency markets.

        Raises:
            ValueError: If degrees_of_freedom is not positive or <= 2.
        """
        if not isinstance(self.degrees_of_freedom, (int, float)):
            raise ValueError(
                f"degrees_of_freedom must be numeric, got {type(self.degrees_of_freedom).__name__}"
            )

        if self.degrees_of_freedom <= 2:
            raise ValueError(
                f"degrees_of_freedom must be greater than 2 for finite variance, "
                f"got {self.degrees_of_freedom}"
            )

        if self.degrees_of_freedom > 30:
            raise ValueError(
                f"degrees_of_freedom should be at most 30 (higher values approach normal distribution), "
                f"got {self.degrees_of_freedom}"
            )

    def _validate_n_simulations(self) -> None:
        """
        Validate number of Monte Carlo simulations parameter.

        The number of simulations must be at least 1000 for stable percentile
        estimates. 10,000 simulations provide good balance between accuracy
        and computational cost.

        Raises:
            ValueError: If n_simulations is not an integer or less than 1000.
        """
        if not isinstance(self.n_simulations, int):
            raise ValueError(
                f"n_simulations must be an integer, got {type(self.n_simulations).__name__}"
            )

        if self.n_simulations < 1000:
            raise ValueError(
                f"n_simulations must be at least 1000 for stable percentile estimates, "
                f"got {self.n_simulations}"
            )

        if self.n_simulations > 1000000:
            raise ValueError(
                f"n_simulations must be at most 1,000,000 to avoid excessive computation, "
                f"got {self.n_simulations}"
            )

    def _validate_confidence_level(self) -> None:
        """
        Validate confidence level parameter.

        Confidence level must be between 0.5 and 0.999. Standard values are
        0.90 (90%), 0.95 (95%), and 0.99 (99%).

        Raises:
            ValueError: If confidence_level is not between 0.5 and 0.999.
        """
        if not isinstance(self.confidence_level, (int, float)):
            raise ValueError(
                f"confidence_level must be numeric, got {type(self.confidence_level).__name__}"
            )

        if self.confidence_level <= 0.5:
            raise ValueError(
                f"confidence_level must be greater than 0.5, got {self.confidence_level}"
            )

        if self.confidence_level >= 1.0:
            raise ValueError(
                f"confidence_level must be less than 1.0, got {self.confidence_level}"
            )

    def get_alpha(self) -> float:
        """
        Calculate alpha parameter from confidence level.

        Alpha represents the total probability in both tails of the distribution.
        For a 95% confidence level, alpha = 0.05 (2.5% in each tail).

        Returns:
            Alpha value (1 - confidence_level).

        Example:
            >>> config = ForecastConfig(confidence_level=0.95)
            >>> config.get_alpha()
            0.05
        """
        return 1.0 - self.confidence_level

    def get_percentiles(self) -> tuple[float, float]:
        """
        Calculate lower and upper percentiles for prediction intervals.

        For a 95% confidence level, returns (2.5, 97.5) representing the
        percentiles that bound the middle 95% of the distribution.

        Returns:
            Tuple of (lower_percentile, upper_percentile) in range [0, 100].

        Example:
            >>> config = ForecastConfig(confidence_level=0.95)
            >>> config.get_percentiles()
            (2.5, 97.5)
        """
        alpha = self.get_alpha()
        lower_percentile = (alpha / 2.0) * 100.0
        upper_percentile = (1.0 - alpha / 2.0) * 100.0
        return lower_percentile, upper_percentile

    def __repr__(self) -> str:
        """
        Return detailed string representation of configuration.

        Returns:
            String showing all configuration parameters and derived values.
        """
        lower_pct, upper_pct = self.get_percentiles()
        return (
            f"ForecastConfig(\n"
            f"  lookback_window={self.lookback_window} bars,\n"
            f"  degrees_of_freedom={self.degrees_of_freedom},\n"
            f"  n_simulations={self.n_simulations:,},\n"
            f"  confidence_level={self.confidence_level} "
            f"(alpha={self.get_alpha()}, percentiles=[{lower_pct}, {upper_pct}])\n"
            f")"
        )


def create_default_config() -> ForecastConfig:
    """
    Create a ForecastConfig instance with default parameters.

    Default values:
        - lookback_window: 24 bars (1 day of hourly data)
        - degrees_of_freedom: 5 (fat-tailed distribution for crypto volatility)
        - n_simulations: 10,000 (balance between accuracy and speed)
        - confidence_level: 0.95 (95% prediction intervals)

    Returns:
        ForecastConfig instance with default parameters.

    Example:
        >>> config = create_default_config()
        >>> print(config.lookback_window)
        24
    """
    return ForecastConfig()


def create_custom_config(
    lookback_window: Optional[int] = None,
    degrees_of_freedom: Optional[float] = None,
    n_simulations: Optional[int] = None,
    confidence_level: Optional[float] = None,
) -> ForecastConfig:
    """
    Create a ForecastConfig instance with custom parameters.

    Parameters not specified will use default values. All parameters are
    validated according to their respective constraints.

    Args:
        lookback_window: EWMA lookback window (10-50 bars). Default: 24.
        degrees_of_freedom: Student-t degrees of freedom (>2). Default: 5.
        n_simulations: Number of Monte Carlo simulations (>=1000). Default: 10000.
        confidence_level: Confidence level (0.5-0.999). Default: 0.95.

    Returns:
        ForecastConfig instance with specified parameters.

    Raises:
        ValueError: If any parameter is outside its valid range.

    Example:
        >>> config = create_custom_config(lookback_window=20, n_simulations=5000)
        >>> print(config.lookback_window)
        20
    """
    # Start with defaults
    defaults = create_default_config()

    # Override with custom values if provided
    return ForecastConfig(
        lookback_window=(
            lookback_window if lookback_window is not None else defaults.lookback_window
        ),
        degrees_of_freedom=(
            degrees_of_freedom
            if degrees_of_freedom is not None
            else defaults.degrees_of_freedom
        ),
        n_simulations=(
            n_simulations if n_simulations is not None else defaults.n_simulations
        ),
        confidence_level=(
            confidence_level
            if confidence_level is not None
            else defaults.confidence_level
        ),
    )
