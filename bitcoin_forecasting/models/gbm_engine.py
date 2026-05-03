"""
GBM (Geometric Brownian Motion) mathematical engine module.

This module implements Monte Carlo simulation using modified Geometric Brownian
Motion with Student-t distributed shocks to capture fat-tailed price movements
characteristic of cryptocurrency markets. The GBM engine generates probabilistic
price forecasts by simulating thousands of possible future price paths.
"""

import logging
import numpy as np
from scipy import stats
from typing import Optional

logger = logging.getLogger(__name__)


def simulate_gbm(
    current_price: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    degrees_of_freedom: float,
    n_simulations: int = 10000,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate terminal prices using modified Geometric Brownian Motion with Student-t shocks.

    This function generates Monte Carlo price simulations using a modified GBM formula
    that incorporates Student-t distributed shocks instead of normal distribution.
    The Student-t distribution with low degrees of freedom (3-6) produces fat tails
    that better capture extreme price movements in cryptocurrency markets.

    The modified GBM formula is:
        S_t = S_0 * exp((μ - 0.5*σ²)*Δt + σ*√Δt*Z)

    where:
        S_t = terminal price at time t
        S_0 = current price
        μ = drift (expected return)
        σ = volatility (annualized)
        Δt = time horizon (in years, e.g., 1/8760 for 1 hour)
        Z ~ Student-t(df) = random shock from Student-t distribution

    The exponential transformation ensures all simulated prices remain positive,
    which is a critical requirement for price modeling.

    Parameters
    ----------
    current_price : float
        Current Bitcoin price in USD. Must be positive.
    drift : float
        Expected return (drift parameter μ) in annualized terms.
        Can be positive (upward trend) or negative (downward trend).
        Typical range: -1.0 to 1.0 for hourly predictions.
    volatility : float
        Annualized volatility (σ) from EWMA calculation. Must be positive.
        Typical range: 0.5 to 2.0 for Bitcoin.
    time_horizon : float
        Time horizon in years. For 1-hour forecast: 1/8760.
        Must be positive.
    degrees_of_freedom : float
        Degrees of freedom for Student-t distribution. Must be > 2.
        Lower values (3-6) produce fatter tails.
        Higher values (>30) approach normal distribution.
    n_simulations : int, optional
        Number of Monte Carlo price paths to generate.
        Must be at least 1000 for stable percentile estimates.
        Default: 10000.
    random_seed : int, optional
        Random seed for reproducibility. If None, results are non-deterministic.
        Default: None.

    Returns
    -------
    np.ndarray
        Array of simulated terminal prices with shape (n_simulations,).
        All values are guaranteed to be positive and finite.

    Raises
    ------
    ValueError
        If current_price <= 0
        If volatility <= 0
        If time_horizon <= 0
        If degrees_of_freedom <= 2
        If n_simulations < 1000
        If any simulated prices are non-positive or non-finite

    Examples
    --------
    >>> # Simulate 1-hour ahead prices for Bitcoin at $50,000
    >>> current_price = 50000.0
    >>> drift = 0.0  # Neutral drift
    >>> volatility = 0.80  # 80% annualized volatility
    >>> time_horizon = 1.0 / 8760  # 1 hour in years
    >>> df = 5.0  # Fat-tailed distribution
    >>>
    >>> terminal_prices = simulate_gbm(
    ...     current_price=current_price,
    ...     drift=drift,
    ...     volatility=volatility,
    ...     time_horizon=time_horizon,
    ...     degrees_of_freedom=df,
    ...     n_simulations=10000
    ... )
    >>>
    >>> print(f"Mean terminal price: ${terminal_prices.mean():.2f}")
    >>> print(f"Min terminal price: ${terminal_prices.min():.2f}")
    >>> print(f"Max terminal price: ${terminal_prices.max():.2f}")

    Notes
    -----
    - Requirement 2.1: Implements modified GBM simulation for price path generation
    - Requirement 2.2: Uses Student-t distribution with configurable degrees of freedom
    - Requirement 2.3: Calls scipy.stats.t.rvs with df parameter and size 10000
    - Requirement 2.4: Does NOT use normal distribution for shock generation
    - Requirement 2.5: Accepts current price, drift, volatility, time horizon, and df
    - Requirement 2.6: Generates exactly 10000 price paths per prediction
    - Requirement 2.7: Ensures all simulated prices remain positive
    - Requirement 2.8: Returns array of 10000 simulated terminal prices

    References
    ----------
    - Hull, J. C. (2018). Options, Futures, and Other Derivatives (10th ed.)
    - Glasserman, P. (2003). Monte Carlo Methods in Financial Engineering
    """
    # Validate current_price
    if not isinstance(current_price, (int, float)):
        raise ValueError(
            f"current_price must be numeric, got {type(current_price).__name__}"
        )

    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")

    if not np.isfinite(current_price):
        raise ValueError(f"current_price must be finite, got {current_price}")

    # Validate drift
    if not isinstance(drift, (int, float)):
        raise ValueError(f"drift must be numeric, got {type(drift).__name__}")

    if not np.isfinite(drift):
        raise ValueError(f"drift must be finite, got {drift}")

    # Validate volatility
    if not isinstance(volatility, (int, float)):
        raise ValueError(f"volatility must be numeric, got {type(volatility).__name__}")

    if volatility <= 0:
        raise ValueError(f"volatility must be positive, got {volatility}")

    if not np.isfinite(volatility):
        raise ValueError(f"volatility must be finite, got {volatility}")

    # Validate time_horizon
    if not isinstance(time_horizon, (int, float)):
        raise ValueError(
            f"time_horizon must be numeric, got {type(time_horizon).__name__}"
        )

    if time_horizon <= 0:
        raise ValueError(f"time_horizon must be positive, got {time_horizon}")

    if not np.isfinite(time_horizon):
        raise ValueError(f"time_horizon must be finite, got {time_horizon}")

    # Validate degrees_of_freedom
    if not isinstance(degrees_of_freedom, (int, float)):
        raise ValueError(
            f"degrees_of_freedom must be numeric, got {type(degrees_of_freedom).__name__}"
        )

    if degrees_of_freedom <= 2:
        raise ValueError(
            f"degrees_of_freedom must be greater than 2 for finite variance, "
            f"got {degrees_of_freedom}"
        )

    if not np.isfinite(degrees_of_freedom):
        raise ValueError(f"degrees_of_freedom must be finite, got {degrees_of_freedom}")

    # Validate n_simulations
    if not isinstance(n_simulations, int):
        raise ValueError(
            f"n_simulations must be an integer, got {type(n_simulations).__name__}"
        )

    if n_simulations < 1000:
        raise ValueError(
            f"n_simulations must be at least 1000 for stable percentile estimates, "
            f"got {n_simulations}"
        )

    # Set random seed for reproducibility if provided
    if random_seed is not None:
        np.random.seed(random_seed)

    logger.debug(
        f"Starting GBM simulation: S_0={current_price:.2f}, μ={drift:.4f}, "
        f"σ={volatility:.4f}, Δt={time_horizon:.6f}, df={degrees_of_freedom:.1f}, "
        f"n={n_simulations}"
    )

    # Requirement 2.3: Generate exactly n_simulations random shocks using scipy.stats.t.rvs
    # with df parameter (Student-t distribution)
    # Requirement 2.4: NOT using normal distribution
    try:
        shocks = stats.t.rvs(df=degrees_of_freedom, size=n_simulations)
    except Exception as e:
        raise ValueError(
            f"Failed to generate Student-t random shocks: {type(e).__name__}: {str(e)}"
        )

    # Validate shocks are finite
    if not np.all(np.isfinite(shocks)):
        raise ValueError(
            f"Generated shocks contain non-finite values. "
            f"Check degrees_of_freedom parameter."
        )

    # Requirement 2.1: Apply modified GBM formula
    # S_t = S_0 * exp((μ - 0.5*σ²)*Δt + σ*√Δt*Z)
    #
    # Breaking down the formula:
    # 1. Drift term: (μ - 0.5*σ²)*Δt
    #    The -0.5*σ² term is the Itô correction for geometric processes
    # 2. Diffusion term: σ*√Δt*Z
    #    Scales the random shock by volatility and square root of time
    # 3. Exponential transformation: exp(...)
    #    Ensures prices remain positive (log-normal distribution)

    drift_term = (drift - 0.5 * volatility**2) * time_horizon
    diffusion_term = volatility * np.sqrt(time_horizon) * shocks

    # Compute log returns
    log_returns = drift_term + diffusion_term

    # Requirement 2.7: Transform to prices using exponential (ensures positivity)
    # S_t = S_0 * exp(log_return)
    terminal_prices = current_price * np.exp(log_returns)

    # Requirement 2.7: Validate all simulated prices are positive
    if not np.all(terminal_prices > 0):
        num_non_positive = np.sum(terminal_prices <= 0)
        raise ValueError(
            f"Generated {num_non_positive} non-positive prices out of {n_simulations}. "
            f"This should not happen with exponential transformation."
        )

    # Validate all prices are finite
    if not np.all(np.isfinite(terminal_prices)):
        num_non_finite = np.sum(~np.isfinite(terminal_prices))
        raise ValueError(
            f"Generated {num_non_finite} non-finite prices out of {n_simulations}. "
            f"Check input parameters for extreme values."
        )

    logger.debug(
        f"GBM simulation complete: mean=${terminal_prices.mean():.2f}, "
        f"std=${terminal_prices.std():.2f}, "
        f"min=${terminal_prices.min():.2f}, max=${terminal_prices.max():.2f}"
    )

    # Requirement 2.8: Return array of simulated terminal prices
    return terminal_prices


def extract_prediction_interval(
    terminal_prices: np.ndarray, confidence_level: float = 0.95
) -> tuple[float, float]:
    """
    Extract prediction interval from Monte Carlo simulation results.

    This function computes confidence intervals from simulated terminal prices
    by extracting percentiles. For a 95% confidence interval, it extracts the
    2.5th and 97.5th percentiles, which define the bounds within which we expect
    the actual price to fall 95% of the time.

    The prediction interval provides a measure of forecast uncertainty. Wider
    intervals indicate higher uncertainty, while narrower intervals indicate
    more confident predictions.

    Parameters
    ----------
    terminal_prices : np.ndarray
        Array of simulated terminal prices from Monte Carlo simulation.
        Must contain at least 1000 values for stable percentile estimates.
        All values must be positive and finite.
    confidence_level : float, optional
        Confidence level for the prediction interval.
        Must be between 0 and 1 (exclusive).
        Default: 0.95 (95% confidence interval).

    Returns
    -------
    tuple[float, float]
        A tuple (lower_bound, upper_bound) representing the prediction interval.
        - lower_bound: The lower bound of the confidence interval
        - upper_bound: The upper bound of the confidence interval
        Both bounds are guaranteed to be positive and finite.

    Raises
    ------
    ValueError
        If terminal_prices is empty or has fewer than 1000 values
        If any terminal_prices are non-positive or non-finite
        If confidence_level is not between 0 and 1 (exclusive)
        If lower_bound >= upper_bound (should never happen with valid inputs)
        If either bound is non-positive

    Examples
    --------
    >>> # Extract 95% confidence interval from simulation results
    >>> terminal_prices = simulate_gbm(
    ...     current_price=50000.0,
    ...     drift=0.0,
    ...     volatility=0.80,
    ...     time_horizon=1.0 / 8760,
    ...     degrees_of_freedom=5.0,
    ...     n_simulations=10000
    ... )
    >>>
    >>> lower_bound, upper_bound = extract_prediction_interval(terminal_prices)
    >>> print(f"95% Prediction Interval: [${lower_bound:.2f}, ${upper_bound:.2f}]")
    >>> print(f"Interval Width: ${upper_bound - lower_bound:.2f}")

    >>> # Extract 90% confidence interval
    >>> lower_bound, upper_bound = extract_prediction_interval(
    ...     terminal_prices,
    ...     confidence_level=0.90
    ... )

    Notes
    -----
    - Requirement 4.1: Extracts 2.5th percentile as lower bound using numpy.percentile
    - Requirement 4.2: Extracts 97.5th percentile as upper bound
    - Requirement 4.3: Returns tuple (lower_bound, upper_bound)
    - Requirement 4.4: Validates lower_bound < upper_bound
    - Requirement 4.5: Validates both bounds are positive

    The percentile method is robust and non-parametric, making no assumptions
    about the distribution of terminal prices. This is important because the
    Student-t shocks create a non-normal distribution of prices.

    For a 95% confidence interval:
    - Lower percentile = (1 - 0.95) / 2 = 0.025 = 2.5th percentile
    - Upper percentile = 1 - 0.025 = 0.975 = 97.5th percentile

    References
    ----------
    - Christoffersen, P. F. (1998). Evaluating Interval Forecasts.
      International Economic Review, 39(4), 841-862.
    """
    # Validate terminal_prices is a numpy array
    if not isinstance(terminal_prices, np.ndarray):
        raise ValueError(
            f"terminal_prices must be a numpy array, got {type(terminal_prices).__name__}"
        )

    # Validate terminal_prices is not empty
    if terminal_prices.size == 0:
        raise ValueError("terminal_prices array is empty")

    # Validate terminal_prices has at least 1000 values for stable percentile estimates
    if terminal_prices.size < 1000:
        raise ValueError(
            f"terminal_prices must contain at least 1000 values for stable percentile estimates, "
            f"got {terminal_prices.size}"
        )

    # Validate all prices are finite (must check before positivity check)
    if not np.all(np.isfinite(terminal_prices)):
        num_non_finite = np.sum(~np.isfinite(terminal_prices))
        raise ValueError(
            f"All terminal prices must be finite, found {num_non_finite} non-finite values"
        )

    # Requirement 4.5: Validate all prices are positive
    if not np.all(terminal_prices > 0):
        num_non_positive = np.sum(terminal_prices <= 0)
        raise ValueError(
            f"All terminal prices must be positive, found {num_non_positive} non-positive values"
        )

    # Validate confidence_level
    if not isinstance(confidence_level, (int, float)):
        raise ValueError(
            f"confidence_level must be numeric, got {type(confidence_level).__name__}"
        )

    if not (0 < confidence_level < 1):
        raise ValueError(
            f"confidence_level must be between 0 and 1 (exclusive), got {confidence_level}"
        )

    # Calculate percentiles for the confidence interval
    # For 95% confidence: lower = 2.5th percentile, upper = 97.5th percentile
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100  # Convert to percentage for numpy.percentile
    upper_percentile = (1 - alpha / 2) * 100

    logger.debug(
        f"Extracting prediction interval: confidence={confidence_level:.2f}, "
        f"lower_percentile={lower_percentile:.2f}%, upper_percentile={upper_percentile:.2f}%"
    )

    # Requirement 4.1: Extract lower bound using numpy.percentile
    lower_bound = np.percentile(terminal_prices, lower_percentile)

    # Requirement 4.2: Extract upper bound using numpy.percentile
    upper_bound = np.percentile(terminal_prices, upper_percentile)

    # Requirement 4.4: Validate lower_bound < upper_bound
    if lower_bound >= upper_bound:
        raise ValueError(
            f"Invalid prediction interval: lower_bound ({lower_bound}) >= upper_bound ({upper_bound}). "
            f"This should not happen with valid simulation results."
        )

    # Requirement 4.5: Validate both bounds are positive
    if lower_bound <= 0:
        raise ValueError(f"Lower bound must be positive, got {lower_bound}")

    if upper_bound <= 0:
        raise ValueError(f"Upper bound must be positive, got {upper_bound}")

    # Validate both bounds are finite
    if not np.isfinite(lower_bound):
        raise ValueError(f"Lower bound must be finite, got {lower_bound}")

    if not np.isfinite(upper_bound):
        raise ValueError(f"Upper bound must be finite, got {upper_bound}")

    logger.debug(
        f"Prediction interval extracted: [{lower_bound:.2f}, {upper_bound:.2f}], "
        f"width={upper_bound - lower_bound:.2f}"
    )

    # Requirement 4.3: Return tuple (lower_bound, upper_bound)
    return (lower_bound, upper_bound)
