"""
Mathematical models module including EWMA volatility calculator and GBM simulation engine.
"""

from bitcoin_forecasting.models.ewma import (
    compute_ewma_volatility,
    compute_ewma_volatility_series,
    FALLBACK_VOLATILITY,
)

from bitcoin_forecasting.models.gbm_engine import (
    simulate_gbm,
    extract_prediction_interval,
)

__all__ = [
    "compute_ewma_volatility",
    "compute_ewma_volatility_series",
    "FALLBACK_VOLATILITY",
    "simulate_gbm",
    "extract_prediction_interval",
]
