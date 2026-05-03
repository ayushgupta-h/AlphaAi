"""
Core mathematical engine for Bitcoin probabilistic forecasting.

This module contains the pure mathematical functions for EWMA volatility calculation
and Student-t GBM simulation, reusing the proven methodology from the existing
backtesting system.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_ewma_volatility(prices: pd.Series, lookback_window: int = 24, decay_param: float = 0.94) -> float:
    """
    Compute Exponentially Weighted Moving Average volatility from price series.
    
    Args:
        prices: Series of historical prices
        lookback_window: Number of periods to look back (default 24 hours)
        decay_param: Exponential decay parameter (default 0.94)
    
    Returns:
        Annualized volatility estimate
    """
    if len(prices) < 2:
        # Fallback volatility for Bitcoin (80% annualized)
        return 0.80
    
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    if len(log_returns) == 0:
        return 0.80
    
    # Use only the most recent lookback_window returns
    recent_returns = log_returns.tail(lookback_window)
    
    if len(recent_returns) < 2:
        return 0.80
    
    # Calculate EWMA variance
    weights = np.array([decay_param ** i for i in range(len(recent_returns))])
    weights = weights[::-1]  # Reverse so most recent gets highest weight
    weights = weights / weights.sum()  # Normalize
    
    # Compute weighted variance
    mean_return = np.average(recent_returns, weights=weights)
    variance = np.average((recent_returns - mean_return) ** 2, weights=weights)
    
    # Convert to volatility and annualize (8760 hours per year)
    volatility = np.sqrt(variance * 8760)
    
    # Ensure volatility is positive and reasonable
    volatility = max(volatility, 0.01)  # Minimum 1% volatility
    volatility = min(volatility, 5.0)   # Maximum 500% volatility
    
    return volatility


def simulate_gbm_student_t(
    current_price: float,
    drift: float,
    volatility: float,
    time_horizon: float,
    degrees_of_freedom: float = 5.0,
    n_simulations: int = 10000,
    random_seed: int = None
) -> np.ndarray:
    """
    Simulate Geometric Brownian Motion with Student-t distributed shocks.
    
    Args:
        current_price: Current asset price
        drift: Annualized drift parameter
        volatility: Annualized volatility
        time_horizon: Time horizon in years (1/8760 for 1 hour)
        degrees_of_freedom: Degrees of freedom for Student-t distribution
        n_simulations: Number of Monte Carlo paths
        random_seed: Random seed for reproducibility (optional)
    
    Returns:
        Array of simulated terminal prices
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Generate Student-t distributed random shocks
    shocks = stats.t.rvs(df=degrees_of_freedom, size=n_simulations)
    
    # Normalize shocks to have unit variance (Student-t with df > 2 has variance df/(df-2))
    if degrees_of_freedom > 2:
        shock_variance = degrees_of_freedom / (degrees_of_freedom - 2)
        shocks = shocks / np.sqrt(shock_variance)
    
    # Apply GBM formula: S_t = S_0 * exp((drift - 0.5*vol^2)*dt + vol*sqrt(dt)*shock)
    log_returns = (drift - 0.5 * volatility**2) * time_horizon + volatility * np.sqrt(time_horizon) * shocks
    
    # Calculate terminal prices
    terminal_prices = current_price * np.exp(log_returns)
    
    # Ensure all prices are positive (should be guaranteed by exp, but safety check)
    terminal_prices = np.maximum(terminal_prices, 0.01)
    
    return terminal_prices


def extract_prediction_interval(terminal_prices: np.ndarray, confidence_level: float = 0.95) -> Tuple[float, float]:
    """
    Extract prediction interval from Monte Carlo simulation results.
    
    Args:
        terminal_prices: Array of simulated terminal prices
        confidence_level: Confidence level (e.g., 0.95 for 95% interval)
    
    Returns:
        Tuple of (lower_bound, upper_bound)
    """
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(terminal_prices, lower_percentile)
    upper_bound = np.percentile(terminal_prices, upper_percentile)
    
    return lower_bound, upper_bound


def estimate_drift(prices: pd.Series, lookback_window: int = 24) -> float:
    """
    Estimate annualized drift from recent price returns.
    
    Args:
        prices: Series of historical prices
        lookback_window: Number of periods to use for estimation
    
    Returns:
        Annualized drift estimate
    """
    if len(prices) < 2:
        return 0.0
    
    # Calculate log returns
    log_returns = np.log(prices / prices.shift(1)).dropna()
    
    if len(log_returns) == 0:
        return 0.0
    
    # Use recent returns for drift estimation
    recent_returns = log_returns.tail(lookback_window)
    
    # Calculate mean hourly return
    mean_hourly_return = recent_returns.mean()
    
    # Annualize (8760 hours per year)
    annualized_drift = mean_hourly_return * 8760
    
    # Cap extreme drift values
    annualized_drift = np.clip(annualized_drift, -2.0, 2.0)
    
    return annualized_drift


def generate_prediction(
    current_price: float,
    historical_prices: List[float],
    confidence_level: float = 0.95,
    lookback_window: int = 24,
    degrees_of_freedom: float = 5.0,
    n_simulations: int = 10000
) -> dict:
    """
    Generate complete Bitcoin price prediction.
    
    Args:
        current_price: Current Bitcoin price
        historical_prices: List of recent hourly prices
        confidence_level: Confidence level for prediction interval
        lookback_window: EWMA lookback window
        degrees_of_freedom: Student-t degrees of freedom
        n_simulations: Number of Monte Carlo simulations
    
    Returns:
        Dictionary containing prediction results
    """
    # Convert to pandas Series
    price_series = pd.Series(historical_prices + [current_price])
    
    # Compute volatility using EWMA
    volatility = compute_ewma_volatility(price_series, lookback_window)
    
    # Estimate drift from recent returns
    drift = estimate_drift(price_series, lookback_window)
    
    # Run Monte Carlo simulation (1 hour = 1/8760 years)
    time_horizon = 1.0 / 8760
    terminal_prices = simulate_gbm_student_t(
        current_price=current_price,
        drift=drift,
        volatility=volatility,
        time_horizon=time_horizon,
        degrees_of_freedom=degrees_of_freedom,
        n_simulations=n_simulations
    )
    
    # Extract prediction interval
    lower_bound, upper_bound = extract_prediction_interval(terminal_prices, confidence_level)
    
    return {
        'current_price': current_price,
        'lower_bound': lower_bound,
        'upper_bound': upper_bound,
        'confidence_level': confidence_level,
        'volatility': volatility,
        'drift': drift,
        'interval_width': upper_bound - lower_bound,
        'terminal_prices': terminal_prices
    }


def calculate_winkler_score(actual_price: float, lower_bound: float, upper_bound: float, alpha: float = 0.05) -> float:
    """
    Calculate Winkler score for interval prediction.
    
    Args:
        actual_price: Actual observed price
        lower_bound: Lower bound of prediction interval
        upper_bound: Upper bound of prediction interval
        alpha: Alpha level (1 - confidence_level)
    
    Returns:
        Winkler score (lower is better)
    """
    interval_width = upper_bound - lower_bound
    
    if lower_bound <= actual_price <= upper_bound:
        # Price within interval
        score = interval_width
    elif actual_price < lower_bound:
        # Price below interval
        score = interval_width + (2 / alpha) * (lower_bound - actual_price)
    else:
        # Price above interval
        score = interval_width + (2 / alpha) * (actual_price - upper_bound)
    
    return score


def calculate_coverage(predictions: List[dict]) -> float:
    """
    Calculate coverage rate for a list of predictions.
    
    Args:
        predictions: List of prediction dictionaries with actual_price, lower_bound, upper_bound
    
    Returns:
        Coverage rate (fraction of predictions where actual fell within bounds)
    """
    if not predictions:
        return 0.0
    
    hits = 0
    for pred in predictions:
        if pred['lower_bound'] <= pred['actual_price'] <= pred['upper_bound']:
            hits += 1
    
    return hits / len(predictions)


def calculate_average_width(predictions: List[dict]) -> float:
    """
    Calculate average interval width for a list of predictions.
    
    Args:
        predictions: List of prediction dictionaries with lower_bound, upper_bound
    
    Returns:
        Average interval width
    """
    if not predictions:
        return 0.0
    
    widths = [pred['upper_bound'] - pred['lower_bound'] for pred in predictions]
    return np.mean(widths)


def calculate_mean_winkler_score(predictions: List[dict], alpha: float = 0.05) -> float:
    """
    Calculate mean Winkler score for a list of predictions.
    
    Args:
        predictions: List of prediction dictionaries with actual_price, lower_bound, upper_bound
        alpha: Alpha level (1 - confidence_level)
    
    Returns:
        Mean Winkler score
    """
    if not predictions:
        return 0.0
    
    scores = []
    for pred in predictions:
        score = calculate_winkler_score(
            pred['actual_price'],
            pred['lower_bound'],
            pred['upper_bound'],
            alpha
        )
        scores.append(score)
    
    return np.mean(scores)