"""
Mathematical Engine Service for Bitcoin Prediction API.

This module integrates the existing GBM engine, EWMA calculator, and configuration
from the bitcoin_forecasting package into the API service architecture. It provides
a service layer that adapts the mathematical components for use in the FastAPI service.
"""

import logging
import numpy as np
import pandas as pd
from typing import List, Optional, NamedTuple
from datetime import datetime

# Import existing mathematical components from bitcoin_forecasting package
from bitcoin_forecasting.models.gbm_engine import simulate_gbm, extract_prediction_interval
from bitcoin_forecasting.models.ewma import compute_ewma_volatility
from bitcoin_forecasting.config import ForecastConfig

from bitcoin_prediction_api.config import settings

logger = logging.getLogger(__name__)


class PredictionResult(NamedTuple):
    """Result container for mathematical engine prediction."""
    lower_bound: float
    upper_bound: float
    volatility: float
    drift: float
    terminal_prices: np.ndarray


class MathEngineService:
    """
    Service that orchestrates the mathematical engine components.
    
    This service integrates the existing GBM engine, EWMA calculator, and configuration
    from the bitcoin_forecasting package, adapting them for use in the API service.
    It maintains mathematical consistency with the backtesting system while providing
    a clean interface for the prediction service.
    """
    
    def __init__(self):
        """Initialize the mathematical engine service with configuration."""
        # Create ForecastConfig from API settings
        self.config = self._create_forecast_config()
        logger.info(
            "Mathematical engine initialized - lookback_window=%d, degrees_of_freedom=%.1f, n_simulations=%d",
            self.config.lookback_window,
            self.config.degrees_of_freedom,
            self.config.n_simulations
        )
    
    def _create_forecast_config(self) -> ForecastConfig:
        """
        Create ForecastConfig from API settings.
        
        This method adapts the API service configuration to the mathematical
        engine's expected configuration format, ensuring consistency between
        the API service and the underlying mathematical components.
        
        Returns:
            ForecastConfig instance configured from API settings
        """
        return ForecastConfig(
            lookback_window=settings.lookback_window,
            degrees_of_freedom=settings.degrees_of_freedom,
            n_simulations=settings.n_simulations,
            confidence_level=settings.default_confidence_level
        )
    
    async def generate_prediction(
        self,
        current_price: float,
        historical_prices: List[float],
        confidence_level: float = 0.95
    ) -> PredictionResult:
        """
        Generate probabilistic price prediction using existing mathematical engine.
        
        This method orchestrates the complete prediction workflow:
        1. Compute EWMA volatility from historical prices
        2. Estimate drift from recent returns
        3. Run GBM Monte Carlo simulation
        4. Extract prediction intervals
        
        Args:
            current_price: Current Bitcoin price in USD
            historical_prices: List of recent hourly close prices
            confidence_level: Confidence level for prediction interval
        
        Returns:
            PredictionResult containing bounds, volatility, drift, and raw simulation data
        
        Raises:
            ValueError: If inputs are invalid
            Exception: If mathematical computation fails
        """
        logger.debug(
            "Starting prediction generation - current_price=%.2f, historical_count=%d, confidence_level=%.3f",
            current_price,
            len(historical_prices),
            confidence_level
        )
        
        # Validate inputs
        self._validate_prediction_inputs(current_price, historical_prices, confidence_level)
        
        try:
            # Step 1: Compute EWMA volatility from historical prices
            volatility = self._compute_volatility(historical_prices)
            logger.debug(f"Computed EWMA volatility: {volatility:.4f}")
            
            # Step 2: Estimate drift from recent returns
            drift = self._estimate_drift(historical_prices)
            logger.debug(f"Estimated drift: {drift:.4f}")
            
            # Step 3: Run GBM Monte Carlo simulation using existing engine
            terminal_prices = simulate_gbm(
                current_price=current_price,
                drift=drift,
                volatility=volatility,
                time_horizon=1.0 / 8760,  # 1 hour in years
                degrees_of_freedom=self.config.degrees_of_freedom,
                n_simulations=self.config.n_simulations,
                random_seed=None  # Non-deterministic for production
            )
            
            # Step 4: Extract prediction intervals using existing function
            lower_bound, upper_bound = extract_prediction_interval(
                terminal_prices=terminal_prices,
                confidence_level=confidence_level
            )
            
            # Validate mathematical engine output
            self._validate_prediction_output(lower_bound, upper_bound, volatility, drift)
            
            logger.info(
                "Prediction generated successfully - lower_bound=%.2f, upper_bound=%.2f, interval_width=%.2f, volatility=%.4f, drift=%.6f",
                lower_bound,
                upper_bound,
                upper_bound - lower_bound,
                volatility,
                drift
            )
            
            return PredictionResult(
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                volatility=volatility,
                drift=drift,
                terminal_prices=terminal_prices
            )
            
        except Exception as e:
            logger.error(f"Prediction generation failed: {e}", exc_info=True)
            raise
    
    def _compute_volatility(self, historical_prices: List[float]) -> float:
        """
        Compute EWMA volatility from historical prices.
        
        Uses the existing EWMA calculator with the configured lookback window
        and decay parameter. Converts list to pandas Series as expected by
        the EWMA function.
        
        Args:
            historical_prices: List of recent hourly close prices
        
        Returns:
            Annualized volatility estimate
        """
        # Convert to pandas Series as expected by EWMA function
        price_series = pd.Series(historical_prices)
        
        # Use existing EWMA calculator with configuration
        volatility = compute_ewma_volatility(
            prices=price_series,
            lookback_window=self.config.lookback_window,
            decay_param=0.94  # Standard decay parameter for financial data
        )
        
        return volatility
    
    def _estimate_drift(self, historical_prices: List[float]) -> float:
        """
        Estimate annualized drift from recent price returns.
        
        Computes the mean log return from recent prices and annualizes it.
        For short-term predictions (1 hour), drift is typically small but
        can capture recent momentum.
        
        Args:
            historical_prices: List of recent hourly close prices
        
        Returns:
            Annualized drift estimate
        """
        if len(historical_prices) < 2:
            # Not enough data for drift estimation, use neutral drift
            return 0.0
        
        # Convert to pandas Series for easier computation
        price_series = pd.Series(historical_prices)
        
        # Compute log returns
        log_returns = np.log(price_series / price_series.shift(1)).dropna()
        
        if len(log_returns) == 0:
            return 0.0
        
        # Compute mean hourly return
        mean_hourly_return = log_returns.mean()
        
        # Annualize by multiplying by hours per year
        hours_per_year = 24 * 365
        annualized_drift = mean_hourly_return * hours_per_year
        
        # Cap extreme drift values to avoid numerical instability
        max_drift = 2.0  # 200% annualized drift cap
        annualized_drift = np.clip(annualized_drift, -max_drift, max_drift)
        
        return annualized_drift
    
    def _validate_prediction_inputs(
        self,
        current_price: float,
        historical_prices: List[float],
        confidence_level: float
    ) -> None:
        """
        Validate inputs for prediction generation.
        
        Args:
            current_price: Current Bitcoin price
            historical_prices: Historical price data
            confidence_level: Confidence level for intervals
        
        Raises:
            ValueError: If any input is invalid
        """
        # Validate current price
        if not isinstance(current_price, (int, float)):
            raise ValueError(f"current_price must be numeric, got {type(current_price)}")
        
        if current_price <= 0:
            raise ValueError(f"current_price must be positive, got {current_price}")
        
        if not np.isfinite(current_price):
            raise ValueError(f"current_price must be finite, got {current_price}")
        
        # Validate historical prices
        if not isinstance(historical_prices, list):
            raise ValueError(f"historical_prices must be a list, got {type(historical_prices)}")
        
        if len(historical_prices) < 10:
            raise ValueError(f"Need at least 10 historical prices, got {len(historical_prices)}")
        
        for i, price in enumerate(historical_prices):
            if not isinstance(price, (int, float)):
                raise ValueError(f"historical_prices[{i}] must be numeric, got {type(price)}")
            
            if price <= 0:
                raise ValueError(f"historical_prices[{i}] must be positive, got {price}")
            
            if not np.isfinite(price):
                raise ValueError(f"historical_prices[{i}] must be finite, got {price}")
        
        # Validate confidence level
        if not isinstance(confidence_level, (int, float)):
            raise ValueError(f"confidence_level must be numeric, got {type(confidence_level)}")
        
        if not (0.5 <= confidence_level <= 0.999):
            raise ValueError(f"confidence_level must be between 0.5 and 0.999, got {confidence_level}")
    
    def _validate_prediction_output(
        self,
        lower_bound: float,
        upper_bound: float,
        volatility: float,
        drift: float
    ) -> None:
        """
        Validate mathematical engine produces consistent results.
        
        This validation ensures the mathematical engine produces finite,
        positive price predictions as required by the specification.
        
        Args:
            lower_bound: Lower prediction bound
            upper_bound: Upper prediction bound
            volatility: Volatility estimate
            drift: Drift estimate
        
        Raises:
            ValueError: If any output is invalid
        """
        # Validate bounds are finite and positive
        if not np.isfinite(lower_bound):
            raise ValueError(f"lower_bound must be finite, got {lower_bound}")
        
        if not np.isfinite(upper_bound):
            raise ValueError(f"upper_bound must be finite, got {upper_bound}")
        
        if lower_bound <= 0:
            raise ValueError(f"lower_bound must be positive, got {lower_bound}")
        
        if upper_bound <= 0:
            raise ValueError(f"upper_bound must be positive, got {upper_bound}")
        
        # Validate bound ordering
        if lower_bound >= upper_bound:
            raise ValueError(f"lower_bound ({lower_bound}) must be less than upper_bound ({upper_bound})")
        
        # Validate volatility
        if not np.isfinite(volatility):
            raise ValueError(f"volatility must be finite, got {volatility}")
        
        if volatility <= 0:
            raise ValueError(f"volatility must be positive, got {volatility}")
        
        # Validate drift (can be negative, but must be finite)
        if not np.isfinite(drift):
            raise ValueError(f"drift must be finite, got {drift}")
        
        logger.debug("Mathematical engine output validation passed")
    
    def get_config_summary(self) -> dict:
        """
        Get summary of mathematical engine configuration.
        
        Returns:
            Dictionary containing configuration parameters
        """
        return {
            "lookback_window": self.config.lookback_window,
            "degrees_of_freedom": self.config.degrees_of_freedom,
            "n_simulations": self.config.n_simulations,
            "default_confidence_level": self.config.confidence_level,
            "model_version": "gbm-ewma-v1.0"
        }