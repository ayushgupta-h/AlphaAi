"""
Integration tests for the main pipeline script.

Tests cover:
- End-to-end execution with mock data
- Error propagation and handling
- Input validation at pipeline entry points
- Logging output
"""

import logging
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from main import (
    validate_data,
    validate_config,
    validate_predictions,
    compute_metrics,
    main,
)
from bitcoin_forecasting.config import ForecastConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_data(rows: int = 1000) -> pd.DataFrame:
    """Return a synthetic price DataFrame with the required columns."""
    base_price = 50000.0
    timestamps = [datetime(2024, 1, 1) + timedelta(hours=i) for i in range(rows)]
    prices = [base_price + (i % 100) * 10 for i in range(rows)]
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": prices,
            "high": [p + 100 for p in prices],
            "low": [p - 100 for p in prices],
            "close": prices,
            "volume": [100.0] * rows,
        }
    )


def _make_predictions(n: int = 720) -> list:
    """Return a list of synthetic prediction dicts."""
    return [
        {
            "timestamp": datetime(2024, 1, 12) + timedelta(hours=i),
            "actual_price": 50000.0 + i,
            "lower_bound": 49000.0,
            "upper_bound": 51000.0,
            "volatility": 0.02,
            "drift": 0.001,
        }
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Tests for validate_data
# ---------------------------------------------------------------------------

class TestValidateData:
    """Tests for validate_data function."""

    def test_valid_data_passes(self):
        """Sufficient, well-formed data must not raise."""
        df = _make_price_data(1000)
        validate_data(df)  # should not raise

    def test_insufficient_rows_raises(self):
        """Fewer than 1000 rows must raise ValueError."""
        df = _make_price_data(500)
        with pytest.raises(ValueError, match="Insufficient historical data"):
            validate_data(df)

    def test_missing_column_raises(self):
        """Missing required column must raise ValueError."""
        df = _make_price_data(1000).drop(columns=["close"])
        with pytest.raises(ValueError, match="missing required columns"):
            validate_data(df)

    def test_non_positive_price_raises(self):
        """Non-positive price values must raise ValueError."""
        df = _make_price_data(1000)
        df.loc[0, "close"] = 0
        with pytest.raises(ValueError, match="non-positive values"):
            validate_data(df)


# ---------------------------------------------------------------------------
# Tests for validate_config
# ---------------------------------------------------------------------------

class TestValidateConfig:
    """Tests for validate_config function."""

    def test_default_config_passes(self):
        """Default config must pass all validation checks."""
        config = ForecastConfig()
        validate_config(config)  # should not raise

    def test_lookback_window_too_small_raises(self):
        """lookback_window < 10 must raise ValueError."""
        with pytest.raises(ValueError):
            ForecastConfig(lookback_window=5)

    def test_degrees_of_freedom_too_small_raises(self):
        """degrees_of_freedom <= 2 must raise ValueError."""
        with pytest.raises(ValueError):
            ForecastConfig(degrees_of_freedom=2.0)

    def test_n_simulations_too_small_raises(self):
        """n_simulations < 1000 must raise ValueError."""
        with pytest.raises(ValueError):
            ForecastConfig(n_simulations=500)

    def test_invalid_confidence_level_raises(self):
        """confidence_level >= 1.0 must raise ValueError."""
        with pytest.raises(ValueError):
            ForecastConfig(confidence_level=1.0)


# ---------------------------------------------------------------------------
# Tests for validate_predictions
# ---------------------------------------------------------------------------

class TestValidatePredictions:
    """Tests for validate_predictions function."""

    def test_valid_predictions_pass(self):
        """Well-formed finite predictions must not raise."""
        preds = _make_predictions(10)
        validate_predictions(preds)  # should not raise

    def test_empty_predictions_raises(self):
        """Empty predictions list must raise ValueError."""
        with pytest.raises(ValueError, match="empty"):
            validate_predictions([])

    def test_nan_bound_raises(self):
        """NaN in lower_bound must raise ValueError."""
        preds = _make_predictions(2)
        preds[0]["lower_bound"] = float("nan")
        with pytest.raises(ValueError, match="non-finite"):
            validate_predictions(preds)

    def test_inf_upper_bound_raises(self):
        """Infinite upper_bound must raise ValueError."""
        preds = _make_predictions(2)
        preds[1]["upper_bound"] = float("inf")
        with pytest.raises(ValueError, match="non-finite"):
            validate_predictions(preds)


# ---------------------------------------------------------------------------
# Tests for compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    """Tests for compute_metrics function."""

    def test_returns_all_three_metrics(self):
        """compute_metrics must return coverage, average_width, winkler_score."""
        preds = _make_predictions(10)
        config = ForecastConfig()
        metrics = compute_metrics(preds, config)
        assert "coverage" in metrics
        assert "average_width" in metrics
        assert "winkler_score" in metrics

    def test_coverage_is_float_in_range(self):
        """Coverage must be a float in [0, 1]."""
        preds = _make_predictions(10)
        config = ForecastConfig()
        metrics = compute_metrics(preds, config)
        assert isinstance(metrics["coverage"], float)
        assert 0.0 <= metrics["coverage"] <= 1.0

    def test_all_inside_gives_coverage_one(self):
        """Predictions where actual is always inside interval give coverage 1.0."""
        preds = [
            {
                "timestamp": datetime(2024, 1, 1, i),
                "actual_price": 50000.0,
                "lower_bound": 49000.0,
                "upper_bound": 51000.0,
            }
            for i in range(10)
        ]
        config = ForecastConfig()
        metrics = compute_metrics(preds, config)
        assert metrics["coverage"] == 1.0

    def test_winkler_score_non_negative(self):
        """Winkler score must always be non-negative."""
        preds = _make_predictions(10)
        config = ForecastConfig()
        metrics = compute_metrics(preds, config)
        assert metrics["winkler_score"] >= 0.0


# ---------------------------------------------------------------------------
# Integration tests for full pipeline (main function)
# ---------------------------------------------------------------------------

class TestMainPipeline:
    """Integration tests for the main() function using mocks."""

    def test_successful_run_returns_zero(self, tmp_path):
        """Successful pipeline execution must return exit code 0."""
        price_data = _make_price_data(1000)
        predictions = _make_predictions(720)

        with (
            patch("main.fetch_data", return_value=price_data),
            patch("main.run_backtesting", return_value=predictions),
            patch("main.persist_results"),
        ):
            exit_code = main()

        assert exit_code == 0

    def test_data_fetch_failure_returns_one(self):
        """API fetch failure must propagate and return exit code 1."""
        import requests
        with patch(
            "main.fetch_data",
            side_effect=requests.exceptions.RequestException("Connection refused"),
        ):
            exit_code = main()
        assert exit_code == 1

    def test_validation_failure_returns_one(self):
        """Insufficient data validation failure must return exit code 1."""
        small_df = _make_price_data(100)  # too few rows
        with patch("main.fetch_data", return_value=small_df):
            exit_code = main()
        assert exit_code == 1

    def test_persistence_failure_returns_one(self, tmp_path):
        """File write failure must return exit code 1."""
        price_data = _make_price_data(1000)
        predictions = _make_predictions(720)

        with (
            patch("main.fetch_data", return_value=price_data),
            patch("main.run_backtesting", return_value=predictions),
            patch("main.persist_results", side_effect=OSError("Disk full")),
        ):
            exit_code = main()
        assert exit_code == 1

    def test_logging_contains_pipeline_stages(self, caplog):
        """Log output must contain key pipeline stage log messages."""
        price_data = _make_price_data(1000)
        predictions = _make_predictions(720)

        with (
            patch("main.fetch_data", return_value=price_data),
            patch("main.run_backtesting", return_value=predictions),
            patch("main.persist_results"),
            caplog.at_level(logging.INFO),
        ):
            main()

        log_text = caplog.text
        # Config validation always logs
        assert "Config validated" in log_text or "configuration" in log_text.lower()
        # Metrics stage always runs; coverage is logged
        assert "Stage 3" in log_text or "coverage" in log_text.lower() or "Coverage" in log_text
        # Pipeline always completes successfully
        assert "Pipeline completed" in log_text or "completed" in log_text.lower()
