"""
Prediction API endpoints for Bitcoin Prediction API.

This module contains FastAPI route handlers for prediction generation
and historical data retrieval.
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional
from datetime import datetime
import logging

from bitcoin_prediction_api.models.prediction import (
    PredictionResponse, 
    PredictionRecord, 
    HistoryQueryRequest
)
from bitcoin_prediction_api.services.prediction_service import PredictionService
from bitcoin_prediction_api.dependencies import get_prediction_service

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/predict", response_model=PredictionResponse)
async def predict(
    confidence_level: float = Query(
        default=0.95,
        ge=0.5,
        le=0.999,
        description="Confidence level for prediction interval (0.5-0.999)"
    ),
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> PredictionResponse:
    """
    Generate a new Bitcoin price prediction.
    
    This endpoint fetches current Bitcoin price data and generates a probabilistic
    forecast using the GBM mathematical engine with EWMA volatility estimation.
    
    Args:
        confidence_level: Confidence level for prediction interval (default: 0.95)
        prediction_service: Injected prediction service dependency
    
    Returns:
        PredictionResponse containing current price, prediction bounds, and metadata
    
    Raises:
        HTTPException: 400 for invalid parameters, 500 for prediction failures
    """
    try:
        logger.info(
            "Generating prediction",
            confidence_level=confidence_level
        )
        
        prediction = await prediction_service.generate_prediction(confidence_level)
        
        logger.info(
            "Prediction generated successfully",
            current_price=prediction.current_price,
            interval_width=prediction.interval_width
        )
        
        return prediction
        
    except ValueError as e:
        logger.warning(f"Invalid prediction parameters: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Prediction generation failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "prediction_generation_failed",
                "message": "Failed to generate prediction",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )

@router.get("/predictions/history", response_model=List[PredictionRecord])
async def get_prediction_history(
    start_date: Optional[datetime] = Query(
        default=None,
        description="Start date for historical data query (ISO format)"
    ),
    end_date: Optional[datetime] = Query(
        default=None,
        description="End date for historical data query (ISO format)"
    ),
    limit: int = Query(
        default=1000,
        le=1000,
        ge=1,
        description="Maximum number of records to return (1-1000)"
    ),
    prediction_service: PredictionService = Depends(get_prediction_service)
) -> List[PredictionRecord]:
    """
    Retrieve historical prediction data.
    
    This endpoint queries the MongoDB database for historical predictions
    within the specified date range, sorted by timestamp in descending order.
    
    Args:
        start_date: Start date for query (optional)
        end_date: End date for query (optional)
        limit: Maximum number of records to return (default: 1000)
        prediction_service: Injected prediction service dependency
    
    Returns:
        List of PredictionRecord objects containing historical predictions
    
    Raises:
        HTTPException: 400 for invalid parameters, 500 for query failures
    """
    try:
        # Validate date range
        if start_date and end_date and start_date >= end_date:
            raise ValueError("start_date must be before end_date")
        
        logger.info(
            "Querying prediction history",
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        history = await prediction_service.get_prediction_history(
            start_date=start_date,
            end_date=end_date,
            limit=limit
        )
        
        logger.info(f"Retrieved {len(history)} historical predictions")
        
        return history
        
    except ValueError as e:
        logger.warning(f"Invalid history query parameters: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"History query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": "history_query_failed",
                "message": "Failed to retrieve historical data",
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        )