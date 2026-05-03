"""
FastAPI backend for the Bitcoin Probabilistic Forecasting System.

Endpoints:
  GET  /api/health          – health check
  GET  /api/predict         – fetch live BTC price, run GBM, return interval
  GET  /api/backtest        – return all 720 saved backtest predictions
  GET  /api/metrics         – compute & return evaluation metrics
  GET  /api/price           – return current BTC/USDT price from Binance

Run locally:
    uvicorn api:app --reload --port 8000

Environment variables (optional):
    HOST  – bind address (default "0.0.0.0")
    PORT  – bind port     (default 8000)
"""

import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from bitcoin_forecasting.config import create_default_config
from bitcoin_forecasting.data.data_ingestion import fetch_binance_data
from bitcoin_forecasting.evaluation.metrics import (
    compute_average_width,
    compute_coverage,
    compute_winkler_score,
)
from bitcoin_forecasting.models.ewma import compute_ewma_volatility
from bitcoin_forecasting.models.gbm_engine import (
    extract_prediction_interval,
    simulate_gbm,
)
from bitcoin_forecasting.persistence import load_results

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Bitcoin Probabilistic Forecasting API",
    description=(
        "On-demand GBM inference for next-hour BTC/USDT price prediction intervals. "
        "Returns 95% confidence bounds using EWMA volatility estimation."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Allow all origins so the React frontend (any port / Vercel URL) can call this
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_FILE = "backtest_results.jsonl"
BINANCE_PRICE_URL = "https://data-api.binance.vision/api/v3/ticker/price"
BINANCE_KLINES_URL = "https://data-api.binance.vision/api/v3/klines"
SYMBOL = "BTCUSDT"

# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


class PriceResponse(BaseModel):
    symbol: str
    price: float
    timestamp: str


class PredictionResponse(BaseModel):
    symbol: str
    timestamp: str
    current_price: float
    lower_bound: float
    upper_bound: float
    volatility: float
    drift: float
    confidence_level: float
    horizon_hours: int
    interval_width: float


class MetricsResponse(BaseModel):
    coverage: float
    average_width: float
    winkler_score: float
    total_predictions: int
    violations: int
    coverage_quality: str


class BacktestRecord(BaseModel):
    timestamp: str
    actual_price: float
    lower_bound: float
    upper_bound: float
    covered: bool
    volatility: Optional[float] = None
    drift: Optional[float] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_current_btc_price() -> float:
    """Fetch latest BTC/USDT price from Binance ticker endpoint."""
    try:
        resp = requests.get(
            BINANCE_PRICE_URL, params={"symbol": SYMBOL}, timeout=10
        )
        resp.raise_for_status()
        return float(resp.json()["price"])
    except Exception as e:
        logger.error(f"Failed to fetch BTC price: {e}")
        raise HTTPException(status_code=503, detail=f"Binance API unavailable: {e}")


def _fetch_recent_prices(limit: int = 60) -> list:
    """Fetch recent hourly closes for volatility estimation."""
    try:
        resp = requests.get(
            BINANCE_KLINES_URL,
            params={"symbol": SYMBOL, "interval": "1h", "limit": limit},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return [float(candle[4]) for candle in data]  # close prices
    except Exception as e:
        logger.error(f"Failed to fetch recent klines: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Failed to fetch historical prices for volatility: {e}",
        )


def _estimate_drift(prices: list, lookback: int = 24) -> float:
    """Estimate annualised drift from recent log returns."""
    import numpy as np

    if len(prices) < 2:
        return 0.0
    recent = prices[-min(lookback + 1, len(prices)) :]
    arr = np.array(recent, dtype=float)
    log_returns = np.log(arr[1:] / arr[:-1])
    if not np.all(np.isfinite(log_returns)):
        return 0.0
    return float(log_returns.mean() * 24 * 365)


def _coverage_quality(coverage: float) -> str:
    if coverage >= 0.93:
        return "Excellent"
    elif coverage >= 0.85:
        return "Good"
    elif coverage >= 0.75:
        return "Fair"
    return "Poor"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        timestamp=datetime.utcnow().isoformat() + "Z",
        version="1.0.0",
    )


@app.get("/api/price", response_model=PriceResponse, tags=["Live"])
def get_price():
    """Return current BTC/USDT spot price from Binance."""
    price = _get_current_btc_price()
    return PriceResponse(
        symbol=SYMBOL,
        price=price,
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/api/predict", response_model=PredictionResponse, tags=["Live"])
def predict():
    """
    Run on-demand GBM inference for the next 1 hour.

    Fetches current BTC price, estimates EWMA volatility from the
    last 60 hourly closes, runs 10,000 Monte Carlo GBM paths with
    Student-t shocks, and returns the 2.5th – 97.5th percentile interval.
    """
    import pandas as pd

    config = create_default_config()

    # 1. Fetch recent closes for volatility + drift estimation
    logger.info("Fetching recent BTC/USDT prices for on-demand inference…")
    closes = _fetch_recent_prices(limit=60)

    if len(closes) < 10:
        raise HTTPException(
            status_code=503,
            detail="Insufficient price history returned by Binance.",
        )

    # 2. Current price is the latest close
    current_price = closes[-1]

    # 3. EWMA volatility
    prices_series = pd.Series(closes)
    volatility = compute_ewma_volatility(
        prices=prices_series,
        lookback_window=min(config.lookback_window, len(closes) - 1),
        decay_param=0.94,
    )

    # 4. Drift estimate
    drift = _estimate_drift(closes, lookback=24)

    # 5. GBM simulation — 1 hour horizon
    time_horizon = 1.0 / 8760.0
    terminal_prices = simulate_gbm(
        current_price=current_price,
        drift=drift,
        volatility=volatility,
        time_horizon=time_horizon,
        degrees_of_freedom=config.degrees_of_freedom,
        n_simulations=config.n_simulations,
    )

    # 6. Extract prediction interval
    lower_bound, upper_bound = extract_prediction_interval(
        terminal_prices=terminal_prices,
        confidence_level=config.confidence_level,
    )

    logger.info(
        f"Prediction: current=${current_price:,.2f} "
        f"interval=[${lower_bound:,.2f}, ${upper_bound:,.2f}] "
        f"vol={volatility:.4f}"
    )

    return PredictionResponse(
        symbol=SYMBOL,
        timestamp=datetime.utcnow().isoformat() + "Z",
        current_price=round(current_price, 2),
        lower_bound=round(lower_bound, 2),
        upper_bound=round(upper_bound, 2),
        volatility=round(volatility, 6),
        drift=round(drift, 6),
        confidence_level=config.confidence_level,
        horizon_hours=1,
        interval_width=round(upper_bound - lower_bound, 2),
    )


@app.get("/api/backtest", response_model=List[BacktestRecord], tags=["Backtest"])
def get_backtest():
    """
    Return all saved backtest predictions from backtest_results.jsonl.

    Adds a boolean `covered` field indicating whether the actual price
    fell within the predicted interval.

    Run `python main.py` first to generate the results file.
    """
    if not os.path.exists(RESULTS_FILE):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Results file '{RESULTS_FILE}' not found. "
                "Run `python main.py` first to generate backtest results."
            ),
        )

    try:
        records = load_results(RESULTS_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {e}")

    result = []
    for r in records:
        covered = r["lower_bound"] <= r["actual_price"] <= r["upper_bound"]
        result.append(
            BacktestRecord(
                timestamp=r["timestamp"],
                actual_price=r["actual_price"],
                lower_bound=r["lower_bound"],
                upper_bound=r["upper_bound"],
                covered=covered,
                volatility=r.get("volatility"),
                drift=r.get("drift"),
            )
        )

    return result


@app.get("/api/metrics", response_model=MetricsResponse, tags=["Backtest"])
def get_metrics():
    """
    Compute and return evaluation metrics from backtest_results.jsonl.

    Returns coverage, average width, and Winkler score.
    """
    if not os.path.exists(RESULTS_FILE):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Results file '{RESULTS_FILE}' not found. "
                "Run `python main.py` first."
            ),
        )

    try:
        predictions = load_results(RESULTS_FILE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load results: {e}")

    coverage = compute_coverage(predictions)
    avg_width = compute_average_width(predictions)
    winkler = compute_winkler_score(predictions)

    violations = sum(
        1
        for p in predictions
        if not (p["lower_bound"] <= p["actual_price"] <= p["upper_bound"])
    )

    return MetricsResponse(
        coverage=coverage,
        average_width=avg_width,
        winkler_score=winkler,
        total_predictions=len(predictions),
        violations=violations,
        coverage_quality=_coverage_quality(coverage),
    )


# ---------------------------------------------------------------------------
# Dev entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("api:app", host=host, port=port, reload=True)
