"""
EWMA (Exponentially Weighted Moving Average) volatility calculator module.

This module implements adaptive volatility estimation using EWMA methodology
to capture volatility clustering in Bitcoin price data. The EWMA approach
gives more weight to recent observations, making it responsive to changing
market conditions while maintaining statistical stability.
"""

import logging
import numpy as np
import pandas as pd
from typing import Optional

logger = logging.getLogger(__name__)

# Fallback volatility estimate (annualized) for when insufficient historical data exists
# This represents approximately 80% annualized volatility, typical for Bitcoin
FALLBACK_VOLATILITY = 0.80


def compute_ewma_volatility(
    prices: pd.Series, lookback_window: int = 24, decay_param: float = 0.94
) -> float:
    """
    Compute EWMA-based volatility estimate from historical price series.

    This function calculates volatility using Exponentially Weighted Moving Average
    of squared log returns. The EWMA approach captures volatility clustering by
    giving exponentially decaying weights to historical observations, with recent
    data having more influence on the estimate.

    The calculation follows these steps:
    1. Calculate log returns: r_t = ln(P_t / P_{t-1})
    2. Apply exponential decay weights: w_i = (1-λ) * λ^i where λ is decay parameter
    3. Compute EWMA variance: σ²_EWMA = Σ(w_i * r_i²)
    4. Take square root to get volatility: σ_EWMA = √(σ²_EWMA)
    5. Annualize by multiplying by √(hours_per_year) = √8760 for hourly data

    Parameters
    ----------
    prices : pd.Series
        Historical close prices in chronological order (oldest first).
        Must contain at least lookback_window + 1 observations.
    lookback_window : int, optional
        Number of historical bars to use for volatility calculation.
        Must be between 10 and 50. Default: 24 (one day of hourly data).
    decay_param : float, optional
        Exponential decay parameter λ in range (0, 1).
        Higher values give more weight to older observations.
        Default: 0.94 (commonly used for financial data).

    Returns
    -------
    float
        Annualized volatility estimate (positive, finite value).
        Returns FALLBACK_VOLATILITY if insufficient data or calculation fails.

    Raises
    ------
    ValueError
        If lookback_window is outside valid range [10, 50]
        If decay_param is outside valid range (0, 1)

    Examples
    --------
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104])
    >>> vol = compute_ewma_volatility(prices, lookback_window=5)
    >>> print(f"Annualized volatility: {vol:.4f}")

    Notes
    -----
    - Requirement 3.1: Implements EWMA of historical returns
    - Requirement 3.2: Uses lookback window between 10 and 24 bars
    - Requirement 3.3: Calculates log returns from consecutive close prices
    - Requirement 3.4: Applies exponential decay weights
    - Requirement 3.5: Computes volatility as square root of EWMA variance
    - Requirement 3.6: Annualizes volatility by multiplying by sqrt(8760)
    - Requirement 3.7: Returns fallback when insufficient data
    - Requirement 3.8: Validates result is positive and finite
    """
    # Validate lookback_window parameter
    if not isinstance(lookback_window, int):
        raise ValueError(
            f"lookback_window must be an integer, got {type(lookback_window).__name__}"
        )

    if lookback_window < 10 or lookback_window > 50:
        raise ValueError(
            f"lookback_window must be between 10 and 50, got {lookback_window}"
        )

    # Validate decay_param parameter
    if not isinstance(decay_param, (int, float)):
        raise ValueError(
            f"decay_param must be numeric, got {type(decay_param).__name__}"
        )

    if decay_param <= 0 or decay_param >= 1:
        raise ValueError(f"decay_param must be in range (0, 1), got {decay_param}")

    # Check if we have sufficient historical data
    # We need lookback_window + 1 prices to compute lookback_window returns
    if len(prices) < lookback_window + 1:
        logger.warning(
            f"Insufficient data for EWMA calculation: need {lookback_window + 1} prices, "
            f"got {len(prices)}. Using fallback volatility {FALLBACK_VOLATILITY:.4f}"
        )
        return FALLBACK_VOLATILITY

    try:
        # Requirement 3.3: Calculate log returns from consecutive close prices
        # Log return: r_t = ln(P_t / P_{t-1})
        # Use the most recent lookback_window + 1 prices
        recent_prices = prices.iloc[-(lookback_window + 1) :]

        # Compute log returns
        log_returns = np.log(recent_prices / recent_prices.shift(1))

        # Drop the first NaN value (from shift operation)
        log_returns = log_returns.dropna()

        # Verify we have exactly lookback_window returns
        if len(log_returns) != lookback_window:
            logger.warning(
                f"Expected {lookback_window} returns, got {len(log_returns)}. "
                f"Using fallback volatility {FALLBACK_VOLATILITY:.4f}"
            )
            return FALLBACK_VOLATILITY

        # Requirement 3.4: Apply exponential decay weights
        # Weight formula: w_i = (1 - λ) * λ^i for i = 0, 1, ..., n-1
        # where i=0 is the most recent observation
        # We reverse the returns so index 0 is most recent
        log_returns_array = log_returns.values[
            ::-1
        ]  # Reverse to make index 0 most recent

        # Generate exponential weights
        # w_i = (1 - λ) * λ^i
        indices = np.arange(lookback_window)
        weights = (1 - decay_param) * (decay_param**indices)

        # Normalize weights to sum to 1
        weights = weights / weights.sum()

        # Requirement 3.5: Compute volatility as square root of EWMA variance
        # EWMA variance: σ²_EWMA = Σ(w_i * r_i²)
        squared_returns = log_returns_array**2
        ewma_variance = np.sum(weights * squared_returns)

        # Volatility is square root of variance
        volatility = np.sqrt(ewma_variance)

        # Requirement 3.6: Annualize volatility by multiplying by sqrt(8760)
        # For hourly data, there are 24 * 365 = 8760 hours per year
        # Annualization factor: √8760 ≈ 93.6
        hours_per_year = 24 * 365
        annualization_factor = np.sqrt(hours_per_year)
        annualized_volatility = volatility * annualization_factor

        # Requirement 3.8: Validate computed volatility is positive and finite
        if not np.isfinite(annualized_volatility):
            logger.warning(
                f"Computed volatility is not finite: {annualized_volatility}. "
                f"Using fallback volatility {FALLBACK_VOLATILITY:.4f}"
            )
            return FALLBACK_VOLATILITY

        if annualized_volatility <= 0:
            logger.warning(
                f"Computed volatility is not positive: {annualized_volatility}. "
                f"Using fallback volatility {FALLBACK_VOLATILITY:.4f}"
            )
            return FALLBACK_VOLATILITY

        logger.debug(
            f"Computed EWMA volatility: {annualized_volatility:.4f} "
            f"(lookback={lookback_window}, decay={decay_param:.2f})"
        )

        return annualized_volatility

    except Exception as e:
        # Requirement 3.7: Return fallback volatility on any calculation error
        logger.error(
            f"Error computing EWMA volatility: {type(e).__name__}: {str(e)}. "
            f"Using fallback volatility {FALLBACK_VOLATILITY:.4f}"
        )
        return FALLBACK_VOLATILITY


def compute_ewma_volatility_series(
    prices: pd.Series,
    lookback_window: int = 24,
    decay_param: float = 0.94,
    min_periods: Optional[int] = None,
) -> pd.Series:
    """
    Compute rolling EWMA volatility estimates for a price series.

    This function computes EWMA volatility at each point in time using only
    historical data available at that point (walk-forward approach). This is
    useful for backtesting where we need volatility estimates at each prediction
    timestamp without look-ahead bias.

    Parameters
    ----------
    prices : pd.Series
        Historical close prices in chronological order (oldest first).
    lookback_window : int, optional
        Number of historical bars to use for each volatility calculation.
        Default: 24.
    decay_param : float, optional
        Exponential decay parameter λ in range (0, 1).
        Default: 0.94.
    min_periods : int, optional
        Minimum number of observations required to compute volatility.
        If None, uses lookback_window + 1. Default: None.

    Returns
    -------
    pd.Series
        Series of annualized volatility estimates aligned with input prices.
        Early values (before min_periods) will be FALLBACK_VOLATILITY.

    Examples
    --------
    >>> prices = pd.Series([100, 102, 101, 103, 105, 104, 106, 108])
    >>> vol_series = compute_ewma_volatility_series(prices, lookback_window=5)
    >>> print(vol_series)
    """
    if min_periods is None:
        min_periods = lookback_window + 1

    volatilities = []

    for i in range(len(prices)):
        # Use only data up to and including current index
        historical_prices = prices.iloc[: i + 1]

        if len(historical_prices) < min_periods:
            # Not enough data yet, use fallback
            volatilities.append(FALLBACK_VOLATILITY)
        else:
            # Compute EWMA volatility using available historical data
            vol = compute_ewma_volatility(
                historical_prices,
                lookback_window=lookback_window,
                decay_param=decay_param,
            )
            volatilities.append(vol)

    return pd.Series(volatilities, index=prices.index)
