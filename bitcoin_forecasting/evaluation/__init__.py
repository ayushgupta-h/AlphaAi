"""
Evaluation metrics module for forecast quality assessment.

This module provides functions to compute coverage, average width, and
Winkler scores for probabilistic forecasts.
"""

from bitcoin_forecasting.evaluation.metrics import (
    compute_coverage,
    compute_average_width,
    compute_winkler_score,
)

__all__ = ["compute_coverage", "compute_average_width", "compute_winkler_score"]
