"""
API module for Bitcoin Prediction API.

This module contains all FastAPI route handlers and API-related functionality.
"""

from bitcoin_prediction_api.api.prediction import router as prediction_router
from bitcoin_prediction_api.api.health import router as health_router

__all__ = ["prediction_router", "health_router"]