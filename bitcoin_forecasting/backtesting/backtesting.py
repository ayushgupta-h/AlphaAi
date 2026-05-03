"""
Backtesting framework module for walk-forward validation.

This module implements rigorous walk-forward backtesting methodology that
ensures no look-ahead bias in predictions. Each prediction at step i uses
only historical data from rows 0 to i-1, simulating real-world forecasting
conditions where future data is not available.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from bitcoin_forecasting.config import ForecastConfig
from bitcoin_forecasting.models.ewma import compute_ewma_volatility
from bitcoin_forecasting.models.gbm_engine import (
    simulate_gbm,
    extract_prediction_interval,
)

logger = logging.getLogger(__name__)


def run_backtest(
    price_data: pd.DataFrame, config: ForecastConfig
) -> List[Dict[str, Any]]:
    """
    Execute walk-forward backtesting with strict temporal validation.

    This function implements walk-forward validation methodology where:
    1. First 280 hourly bars are used for initialization (warm-up period)
    2. Predictions are generated for subsequent 720 hourly bars (test period)
    3. For each prediction at step i, only data from rows 0 to i-1 is used
    4. No future data is ever used for any prediction (no look-ahead bias)
    5. Predictions are generated in strict chronological order

    The walk-forward approach simulates real-world forecasting conditions where
    we must make predictions using only historical data available at prediction
    time. This provides an unbiased estimate of model performance.

    Parameters
    ----------
    price_data : pd.DataFrame
        Full historical price data with columns: timestamp, open, high, low, close, volume.
        Must contain at least 1000 hourly bars (280 initialization + 720 test).
        Data must be sorted by timestamp in ascending order.
    config : ForecastConfig
        Configuration object containing model hyperparameters:
        - lookback_window: Number of bars for EWMA volatility calculation
        - degrees_of_freedom: Student-t distribution parameter
        - n_simulations: Number of Monte Carlo paths
        - confidence_level: Confidence level for prediction intervals

    Returns
    -------
    List[Dict[str, Any]]
        List of prediction dictionaries, one per test timestamp.
        Each dictionary contains:
        - timestamp: datetime object for the prediction
        - actual_price: actual close price at this timestamp
        - lower_bound: lower bound of 95% prediction interval
        - upper_bound: upper bound of 95% prediction interval
        - volatility: EWMA volatility estimate used for this prediction
        - drift: drift estimate used for this prediction

        The list contains exactly 720 predictions in chronological order.

    Raises
    ------
    ValueError
        If price_data has fewer than 1000 rows
        If price_data is not sorted by timestamp
        If any required columns are missing
        If predictions are not in chronological order (internal validation)

    Examples
    --------
    >>> from bitcoin_forecasting.data.data_ingestion import fetch_binance_data
    >>> from bitcoin_forecasting.config import create_default_config
    >>>
    >>> # Fetch data and run backtest
    >>> price_data = fetch_binance_data()
    >>> config = create_default_config()
    >>> predictions = run_backtest(price_data, config)
    >>>
    >>> print(f"Generated {len(predictions)} predictions")
    >>> print(f"First prediction: {predictions[0]}")

    Notes
    -----
    - Requirement 5.1: Uses first 280 hourly bars for initialization
    - Requirement 5.2: Generates predictions for subsequent 720 hourly bars
    - Requirement 5.3: For prediction at step i, uses only data from rows 0 to i-1
    - Requirement 5.4: Does NOT use future data for any prediction
    - Requirement 5.5: Iterates through each test timestamp sequentially
    - Requirement 5.8: Validates predictions are in chronological order

    The backtesting process:
    1. Split data into initialization period (first 280 bars) and test period (next 720 bars)
    2. For each test timestamp t:
       a. Extract historical data available at time t (all data before t)
       b. Compute EWMA volatility using historical data only
       c. Estimate drift from recent returns (last 24 hours)
       d. Run GBM simulation to generate 10,000 price paths
       e. Extract 95% prediction interval from simulated prices
       f. Store prediction with actual price for later evaluation
    3. Validate all predictions are in chronological order
    4. Return list of predictions for evaluation

    References
    ----------
    - Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy:
      an analysis and review. International Journal of Forecasting, 16(4), 437-450.
    """
    # Validate input data
    _validate_price_data(price_data)

    # Constants for backtesting
    INITIALIZATION_PERIOD = 280  # First 280 bars for warm-up
    TEST_PERIOD = 720  # Next 720 bars for testing
    TOTAL_REQUIRED = INITIALIZATION_PERIOD + TEST_PERIOD  # 1000 bars total

    # Requirement 5.1 & 5.2: Validate sufficient data
    if len(price_data) < TOTAL_REQUIRED:
        raise ValueError(
            f"Insufficient data for backtesting: need at least {TOTAL_REQUIRED} bars "
            f"({INITIALIZATION_PERIOD} initialization + {TEST_PERIOD} test), "
            f"got {len(price_data)}"
        )

    logger.info(
        f"Starting walk-forward backtest: {INITIALIZATION_PERIOD} initialization bars, "
        f"{TEST_PERIOD} test bars, total {len(price_data)} bars available"
    )

    # Split data into initialization and test periods
    # Initialization: rows 0 to 279 (first 280 bars)
    # Test: rows 280 to 999 (next 720 bars)
    initialization_end_idx = INITIALIZATION_PERIOD
    test_end_idx = INITIALIZATION_PERIOD + TEST_PERIOD

    # Extract test period data (we'll predict for these timestamps)
    test_data = price_data.iloc[initialization_end_idx:test_end_idx].copy()

    logger.info(
        f"Test period: {test_data['timestamp'].iloc[0]} to {test_data['timestamp'].iloc[-1]}"
    )

    # Initialize list to store predictions
    predictions: List[Dict[str, Any]] = []

    # Requirement 5.5: Iterate through each test timestamp sequentially
    for i, (idx, row) in enumerate(test_data.iterrows()):
        # Current test timestamp and actual price
        current_timestamp = row["timestamp"]
        actual_price = row["close"]

        # Requirement 5.3: Use only data from rows 0 to i-1
        # For the first test prediction (i=0), we use rows 0 to 279 (initialization period)
        # For subsequent predictions, we use all data up to but not including current row
        historical_end_idx = initialization_end_idx + i
        historical_data = price_data.iloc[:historical_end_idx].copy()

        # Requirement 5.4: Validate no future data is used
        # The last timestamp in historical data must be before current timestamp
        last_historical_timestamp = historical_data["timestamp"].iloc[-1]
        if last_historical_timestamp >= current_timestamp:
            raise ValueError(
                f"Look-ahead bias detected at step {i}: "
                f"historical data includes timestamp {last_historical_timestamp} "
                f"which is >= current timestamp {current_timestamp}"
            )

        # Log progress every 100 predictions
        if (i + 1) % 100 == 0:
            logger.info(
                f"Processing prediction {i + 1}/{TEST_PERIOD} at {current_timestamp}"
            )

        # Generate prediction using only historical data
        prediction = _generate_single_prediction(
            historical_data=historical_data,
            current_timestamp=current_timestamp,
            actual_price=actual_price,
            config=config,
        )

        predictions.append(prediction)

    # Requirement 5.8: Validate predictions are in chronological order
    _validate_chronological_order(predictions)

    logger.info(
        f"Backtest complete: generated {len(predictions)} predictions "
        f"from {predictions[0]['timestamp']} to {predictions[-1]['timestamp']}"
    )

    return predictions


def _validate_price_data(price_data: pd.DataFrame) -> None:
    """
    Validate price data meets requirements for backtesting.

    Parameters
    ----------
    price_data : pd.DataFrame
        Price data to validate

    Raises
    ------
    ValueError
        If data is invalid or missing required columns
    """
    # Check required columns exist
    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    missing_columns = [col for col in required_columns if col not in price_data.columns]

    if missing_columns:
        raise ValueError(
            f"Price data missing required columns: {missing_columns}. "
            f"Required columns: {required_columns}"
        )

    # Check data is not empty
    if len(price_data) == 0:
        raise ValueError("Price data is empty")

    # Check timestamps are in chronological order
    timestamps = price_data["timestamp"].values
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            raise ValueError(
                f"Price data is not sorted by timestamp: "
                f"timestamp at index {i} ({timestamps[i]}) is before "
                f"timestamp at index {i-1} ({timestamps[i-1]})"
            )

    # Check all prices are positive
    price_columns = ["open", "high", "low", "close"]
    for col in price_columns:
        if (price_data[col] <= 0).any():
            raise ValueError(
                f"Price data contains non-positive values in column '{col}'"
            )


def _generate_single_prediction(
    historical_data: pd.DataFrame,
    current_timestamp: datetime,
    actual_price: float,
    config: ForecastConfig,
) -> Dict[str, Any]:
    """
    Generate a single prediction using only historical data.

    This function encapsulates the prediction logic for a single timestamp:
    1. Compute EWMA volatility from historical prices
    2. Estimate drift from recent returns
    3. Run GBM Monte Carlo simulation
    4. Extract prediction interval

    Parameters
    ----------
    historical_data : pd.DataFrame
        Historical price data available at prediction time.
        Contains only data from before current_timestamp.
    current_timestamp : datetime
        Timestamp for which we are making the prediction.
    actual_price : float
        Actual close price at current_timestamp (for evaluation).
    config : ForecastConfig
        Configuration object with model hyperparameters.

    Returns
    -------
    Dict[str, Any]
        Prediction dictionary with timestamp, actual_price, bounds, volatility, drift.
    """
    # Extract historical close prices
    historical_prices = historical_data["close"]

    # Get current price (last price in historical data)
    current_price = historical_prices.iloc[-1]

    # Requirement 5.6: Compute volatility using EWMA on historical data only
    volatility = compute_ewma_volatility(
        prices=historical_prices,
        lookback_window=config.lookback_window,
        decay_param=0.94,  # Standard decay parameter for financial data
    )

    # Estimate drift from recent returns
    # Use mean of last 24 hourly log returns as drift estimate
    drift = _estimate_drift(historical_prices, lookback=24)

    # Time horizon: 1 hour in years (1 / 8760)
    # There are 24 * 365 = 8760 hours per year
    time_horizon = 1.0 / 8760.0

    # Run GBM Monte Carlo simulation
    terminal_prices = simulate_gbm(
        current_price=current_price,
        drift=drift,
        volatility=volatility,
        time_horizon=time_horizon,
        degrees_of_freedom=config.degrees_of_freedom,
        n_simulations=config.n_simulations,
    )

    # Extract prediction interval
    lower_bound, upper_bound = extract_prediction_interval(
        terminal_prices=terminal_prices, confidence_level=config.confidence_level
    )

    # Requirement 5.7: Store prediction with all relevant information
    prediction = {
        "timestamp": current_timestamp,
        "actual_price": actual_price,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "volatility": volatility,
        "drift": drift,
    }

    return prediction


def _estimate_drift(prices: pd.Series, lookback: int = 24) -> float:
    """
    Estimate drift parameter from recent returns.

    Drift represents the expected return (trend) in the price process.
    We estimate it as the mean of recent log returns, annualized to match
    the volatility time scale.

    Parameters
    ----------
    prices : pd.Series
        Historical close prices
    lookback : int, optional
        Number of recent returns to use for drift estimation.
        Default: 24 (last 24 hours for hourly data).

    Returns
    -------
    float
        Annualized drift estimate.
        Returns 0.0 if insufficient data or calculation fails.
    """
    try:
        # Need at least lookback + 1 prices to compute lookback returns
        if len(prices) < lookback + 1:
            logger.debug(
                f"Insufficient data for drift estimation: need {lookback + 1} prices, "
                f"got {len(prices)}. Using drift=0.0"
            )
            return 0.0

        # Use most recent lookback + 1 prices
        recent_prices = prices.iloc[-(lookback + 1) :]

        # Compute log returns
        log_returns = np.log(recent_prices / recent_prices.shift(1))
        log_returns = log_returns.dropna()

        # Mean return (hourly)
        mean_return = log_returns.mean()

        # Annualize: multiply by number of hours per year
        hours_per_year = 24 * 365
        annualized_drift = mean_return * hours_per_year

        # Validate result is finite
        if not np.isfinite(annualized_drift):
            logger.warning(
                f"Computed drift is not finite: {annualized_drift}. Using drift=0.0"
            )
            return 0.0

        return annualized_drift

    except Exception as e:
        logger.warning(
            f"Error computing drift: {type(e).__name__}: {str(e)}. Using drift=0.0"
        )
        return 0.0


def _validate_chronological_order(predictions: List[Dict[str, Any]]) -> None:
    """
    Validate that predictions are in chronological order.

    This is a critical validation to ensure walk-forward methodology was
    correctly implemented. Predictions must be generated in time order.

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        List of prediction dictionaries

    Raises
    ------
    ValueError
        If predictions are not in chronological order
    """
    if len(predictions) < 2:
        return  # Nothing to validate

    for i in range(1, len(predictions)):
        current_timestamp = predictions[i]["timestamp"]
        previous_timestamp = predictions[i - 1]["timestamp"]

        if current_timestamp <= previous_timestamp:
            raise ValueError(
                f"Predictions are not in chronological order: "
                f"prediction at index {i} has timestamp {current_timestamp} "
                f"which is <= previous timestamp {previous_timestamp}"
            )

    logger.debug(
        f"Chronological order validated: {len(predictions)} predictions "
        f"from {predictions[0]['timestamp']} to {predictions[-1]['timestamp']}"
    )
