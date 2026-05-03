"""
Main execution script for the Bitcoin Probabilistic Forecasting System.

This script orchestrates the full pipeline:
1. Load configuration from config module
2. Fetch historical Bitcoin price data from Binance API
3. Run walk-forward backtesting with GBM + EWMA volatility
4. Compute evaluation metrics (coverage, average width, Winkler score)
5. Save results to backtest_results.jsonl
6. Print summary statistics to console

Usage:
    python main.py

Requirements:
    See requirements.txt for Python dependencies.
    Active internet connection required for Binance API data fetch.
"""

import logging
import sys
from datetime import datetime
from typing import List, Dict, Any

import numpy as np

# ---------------------------------------------------------------------------
# Requirement 14.2: Logging with timestamps configured before imports
# ---------------------------------------------------------------------------
from bitcoin_forecasting.utils.logging_config import setup_logging

# Set up logging with timestamps before any other module uses the logger
setup_logging()

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline imports
# ---------------------------------------------------------------------------
from bitcoin_forecasting.config import create_default_config, ForecastConfig
from bitcoin_forecasting.data.data_ingestion import fetch_binance_data
from bitcoin_forecasting.backtesting.backtesting import run_backtest
from bitcoin_forecasting.evaluation.metrics import (
    compute_coverage,
    compute_average_width,
    compute_winkler_score,
)
from bitcoin_forecasting.persistence import save_results

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
OUTPUT_FILE = "backtest_results.jsonl"
EXPECTED_PREDICTIONS = 720
MIN_DATA_ROWS = 1000


# ---------------------------------------------------------------------------
# Input validation helpers (Requirement 12.3)
# ---------------------------------------------------------------------------


def validate_data(price_data) -> None:
    """
    Validate price data before running backtesting.

    Checks that enough historical data is present and that all
    required columns exist with positive values.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical price data from Binance.

    Raises
    ------
    ValueError
        If data has fewer than 1000 rows.
        If required columns are missing.
        If any price values are non-positive.
    """
    logger.info(f"Validating price data ({len(price_data)} rows)…")

    # Requirement 15.3: Validate sufficient historical data
    if len(price_data) < MIN_DATA_ROWS:
        raise ValueError(
            f"Insufficient historical data before backtesting: need at least "
            f"{MIN_DATA_ROWS} rows, got {len(price_data)}. "
            "Increase the 'limit' parameter in fetch_binance_data()."
        )

    required_columns = ["timestamp", "open", "high", "low", "close", "volume"]
    missing = [c for c in required_columns if c not in price_data.columns]
    if missing:
        raise ValueError(f"Price data is missing required columns: {missing}")

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if (price_data[col] <= 0).any():
            raise ValueError(
                f"Price data contains non-positive values in column '{col}'. "
                "All prices must be positive."
            )

    logger.info("Price data validation passed.")


def validate_config(config: ForecastConfig) -> None:
    """
    Validate configuration parameters are within acceptable ranges.

    The ForecastConfig dataclass already performs validation in __post_init__,
    but this function provides an additional explicit check with descriptive
    pipeline-level messages.

    Parameters
    ----------
    config : ForecastConfig
        Configuration object to validate.

    Raises
    ------
    ValueError
        If any configuration parameter is out of range.
    """
    logger.info("Validating pipeline configuration…")

    # Requirement 15.4: Validate configuration parameters
    if not (10 <= config.lookback_window <= 50):
        raise ValueError(
            f"lookback_window must be between 10 and 50, got {config.lookback_window}"
        )
    if config.degrees_of_freedom <= 2:
        raise ValueError(
            f"degrees_of_freedom must be > 2 for finite variance, "
            f"got {config.degrees_of_freedom}"
        )
    if config.n_simulations < 1000:
        raise ValueError(
            f"n_simulations must be >= 1000 for stable estimates, "
            f"got {config.n_simulations}"
        )
    if not (0.5 < config.confidence_level < 1.0):
        raise ValueError(
            f"confidence_level must be in (0.5, 1.0), got {config.confidence_level}"
        )

    logger.info(
        f"Config validated: lookback={config.lookback_window}, "
        f"df={config.degrees_of_freedom}, "
        f"n_sim={config.n_simulations:,}, "
        f"confidence={config.confidence_level}"
    )


def validate_predictions(predictions: List[Dict[str, Any]]) -> None:
    """
    Validate Monte Carlo prediction results before computing metrics.

    Ensures all prediction bounds are finite (no NaN / Inf from failed
    Monte Carlo simulations).

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        Prediction dictionaries from backtesting.

    Raises
    ------
    ValueError
        If any lower_bound or upper_bound is not finite.
        If predictions list is empty.
    """
    if not predictions:
        raise ValueError("Predictions list is empty after backtesting.")

    # Requirement 15.5: Validate Monte Carlo results are finite
    for i, pred in enumerate(predictions):
        for field in ("lower_bound", "upper_bound", "actual_price"):
            val = pred.get(field)
            if val is None or not np.isfinite(float(val)):
                raise ValueError(
                    f"Prediction at index {i} has non-finite '{field}': {val}. "
                    "This indicates a failure in Monte Carlo simulation."
                )

    logger.info(f"All {len(predictions)} predictions have finite bounds.")


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------


def fetch_data(config: ForecastConfig):
    """
    Fetch historical Bitcoin price data from Binance API.

    Parameters
    ----------
    config : ForecastConfig
        Pipeline configuration (unused here, included for future extension).

    Returns
    -------
    pd.DataFrame
        Historical OHLCV data with at least 1000 rows.
    """
    logger.info("=== Stage 1: Data Ingestion ===")
    logger.info("Fetching historical BTCUSDT hourly data from Binance API…")

    price_data = fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=1000)

    logger.info(
        f"Fetched {len(price_data)} hourly bars from "
        f"{price_data['timestamp'].iloc[0]} to {price_data['timestamp'].iloc[-1]}"
    )

    return price_data


def run_backtesting(price_data, config: ForecastConfig) -> List[Dict[str, Any]]:
    """
    Execute walk-forward backtesting.

    Parameters
    ----------
    price_data : pd.DataFrame
        Historical OHLCV price data.
    config : ForecastConfig
        Pipeline configuration.

    Returns
    -------
    List[Dict[str, Any]]
        List of 720 prediction dictionaries.
    """
    logger.info("=== Stage 2: Walk-Forward Backtesting ===")
    logger.info(
        f"Running walk-forward backtest: 280 init bars + 720 test bars. "
        f"Generating {EXPECTED_PREDICTIONS} predictions…"
    )

    start_time = datetime.now()
    predictions = run_backtest(price_data, config)
    elapsed = (datetime.now() - start_time).total_seconds()

    logger.info(
        f"Backtesting complete: {len(predictions)} predictions generated "
        f"in {elapsed:.1f} seconds ({elapsed / len(predictions):.3f}s per prediction)"
    )

    return predictions


def compute_metrics(
    predictions: List[Dict[str, Any]],
    config: ForecastConfig,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics for the predictions.

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        Backtesting predictions.
    config : ForecastConfig
        Pipeline configuration (used for alpha parameter).

    Returns
    -------
    Dict[str, float]
        Dictionary with keys: coverage, average_width, winkler_score.
    """
    logger.info("=== Stage 3: Computing Evaluation Metrics ===")

    alpha = config.get_alpha()

    # Requirement 6.1: Coverage metric
    logger.info("Computing coverage metric…")
    coverage = compute_coverage(predictions)
    logger.info(f"Coverage: {coverage:.4f}")

    # Requirement 7.1: Average width metric
    logger.info("Computing average width metric…")
    avg_width = compute_average_width(predictions)
    logger.info(f"Average Width: ${avg_width:,.2f}")

    # Requirement 8.1: Winkler score
    logger.info(f"Computing Winkler score (alpha={alpha})…")
    winkler = compute_winkler_score(predictions, alpha=alpha)
    logger.info(f"Winkler Score: ${winkler:,.2f}")

    # Warn if coverage is outside expected range
    if coverage < 0.85:
        logger.warning(
            f"Coverage {coverage:.4f} is below 0.85 — intervals may be too narrow "
            "(model overconfident). Consider adjusting confidence_level or n_simulations."
        )
    elif coverage > 0.98:
        logger.warning(
            f"Coverage {coverage:.4f} is above 0.98 — intervals may be too wide "
            "(model underconfident). Consider adjusting confidence_level."
        )
    else:
        logger.info(
            f"Coverage {coverage:.4f} is within the expected range [0.85, 0.98] [OK]"
        )

    return {
        "coverage": coverage,
        "average_width": avg_width,
        "winkler_score": winkler,
    }


def persist_results(predictions: List[Dict[str, Any]]) -> None:
    """
    Save prediction results to JSON Lines file.

    Parameters
    ----------
    predictions : List[Dict[str, Any]]
        Backtesting predictions to persist.
    """
    logger.info("=== Stage 4: Persisting Results ===")
    logger.info(f"Saving {len(predictions)} predictions to '{OUTPUT_FILE}'…")

    save_results(predictions, OUTPUT_FILE)

    logger.info(f"Results saved to '{OUTPUT_FILE}'")


def print_summary(metrics: Dict[str, float], predictions: List[Dict[str, Any]]) -> None:
    """
    Print summary statistics to console.

    Parameters
    ----------
    metrics : Dict[str, float]
        Computed evaluation metrics.
    predictions : List[Dict[str, Any]]
        Backtesting predictions (used to compute additional summary stats).
    """
    coverage = metrics["coverage"]
    avg_width = metrics["average_width"]
    winkler = metrics["winkler_score"]

    # Count violations
    violations = sum(
        1
        for p in predictions
        if not (p["lower_bound"] <= p["actual_price"] <= p["upper_bound"])
    )

    border = "=" * 60
    print(f"\n{border}")
    print("  BITCOIN PROBABILISTIC FORECASTING - BACKTEST RESULTS")
    print(border)
    print(f"  Predictions generated : {len(predictions)}")
    print(f"  Coverage              : {coverage:.4f}  ({coverage * 100:.2f}%)")
    print(f"  Average Interval Width: ${avg_width:>12,.2f}")
    print(f"  Mean Winkler Score    : ${winkler:>12,.2f}")
    print(f"  Interval violations   : {violations} / {len(predictions)}")
    print(border)

    # Coverage interpretation
    if coverage >= 0.93:
        quality = "EXCELLENT"
    elif coverage >= 0.85:
        quality = "GOOD"
    elif coverage >= 0.75:
        quality = "FAIR"
    else:
        quality = "POOR (intervals too narrow)"
    print(f"  Coverage quality      : {quality}")
    print(border)
    print(f"\n  Results saved to: {OUTPUT_FILE}")
    print(f"  Launch dashboard: streamlit run dashboard.py\n")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """
    Execute the full Bitcoin probabilistic forecasting pipeline.

    Returns
    -------
    int
        Exit code: 0 on success, 1 on failure.
    """
    logger.info("Bitcoin Probabilistic Forecasting Pipeline starting…")
    start_time = datetime.now()

    try:
        # --- Load configuration ---
        logger.info("Loading pipeline configuration…")
        config = create_default_config()
        validate_config(config)
        logger.info(f"Configuration: {config}")

        # --- Fetch data ---
        price_data = fetch_data(config)
        validate_data(price_data)

        # --- Run backtesting ---
        predictions = run_backtesting(price_data, config)
        validate_predictions(predictions)

        # --- Compute metrics ---
        metrics = compute_metrics(predictions, config)

        # --- Persist results ---
        persist_results(predictions)

        # --- Print summary ---
        print_summary(metrics, predictions)

        total_elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"Pipeline completed successfully in {total_elapsed:.1f} seconds.")

        return 0

    except ValueError as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        print(f"\n[ERROR] Validation failed: {e}", file=sys.stderr)
        return 1

    except OSError as e:
        logger.error(f"File I/O error: {e}", exc_info=True)
        print(f"\n[ERROR] File I/O error: {e}", file=sys.stderr)
        return 1

    except Exception as e:
        logger.error(
            f"Unexpected error in pipeline: {type(e).__name__}: {e}", exc_info=True
        )
        print(f"\n[ERROR] Unexpected error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
