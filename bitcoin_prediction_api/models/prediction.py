"""
Prediction data models for Bitcoin Prediction API.

This module contains Pydantic models for prediction requests, responses,
and database records.
"""

from pydantic import BaseModel, Field, validator
from typing import Optional
from datetime import datetime

class PredictionRequest(BaseModel):
    """Request model for prediction generation."""
    
    confidence_level: float = Field(
        default=0.95,
        ge=0.5,
        le=0.999,
        description="Confidence level for prediction interval"
    )
    
    @validator('confidence_level')
    def validate_confidence_level(cls, v):
        """Validate confidence level is within acceptable range."""
        if not (0.5 <= v <= 0.999):
            raise ValueError('confidence_level must be between 0.5 and 0.999')
        return v

class PredictionResponse(BaseModel):
    """Response model for prediction API."""
    
    symbol: str = Field(default="BTCUSDT", description="Trading symbol")
    timestamp: datetime = Field(description="Prediction generation timestamp")
    current_price: float = Field(description="Current Bitcoin price in USD")
    lower_bound: float = Field(description="Lower bound of prediction interval")
    upper_bound: float = Field(description="Upper bound of prediction interval")
    confidence_level: float = Field(description="Confidence level of the interval")
    prediction_horizon: float = Field(
        default=1.0, 
        description="Prediction horizon in hours"
    )
    volatility: float = Field(description="EWMA volatility estimate")
    drift: float = Field(description="Estimated drift parameter")
    interval_width: float = Field(description="Width of prediction interval")
    model_version: str = Field(
        default="gbm-ewma-v1.0", 
        description="Mathematical model version"
    )
    
    @validator('current_price', 'lower_bound', 'upper_bound', 'volatility')
    def validate_positive_values(cls, v):
        """Validate that price and volatility values are positive."""
        if v <= 0:
            raise ValueError('Value must be positive')
        return v
    
    @validator('upper_bound')
    def validate_bounds_order(cls, v, values):
        """Validate that upper_bound is greater than lower_bound."""
        if 'lower_bound' in values and v <= values['lower_bound']:
            raise ValueError('upper_bound must be greater than lower_bound')
        return v

class PredictionRecord(BaseModel):
    """Database model for stored predictions."""
    
    id: Optional[str] = Field(alias="_id", description="MongoDB document ID")
    timestamp: datetime = Field(description="Prediction generation timestamp")
    current_price: float = Field(description="Current Bitcoin price in USD")
    lower_bound: float = Field(description="Lower bound of prediction interval")
    upper_bound: float = Field(description="Upper bound of prediction interval")
    confidence_level: float = Field(description="Confidence level of the interval")
    prediction_horizon: float = Field(description="Prediction horizon in hours")
    volatility: float = Field(description="EWMA volatility estimate")
    drift: float = Field(description="Estimated drift parameter")
    model_version: str = Field(description="Mathematical model version")
    created_at: datetime = Field(description="Document creation timestamp")
    
    class Config:
        populate_by_name = True  # Updated from allow_population_by_field_name
        json_encoders = {
            datetime: lambda v: v.isoformat() + "Z"
        }

class HistoryQueryRequest(BaseModel):
    """Request model for historical data queries."""
    
    start_date: Optional[datetime] = Field(
        default=None,
        description="Start date for query range"
    )
    end_date: Optional[datetime] = Field(
        default=None,
        description="End date for query range"
    )
    limit: int = Field(
        default=1000,
        le=1000,
        ge=1,
        description="Maximum number of records to return"
    )
    confidence_level: Optional[float] = Field(
        default=None,
        ge=0.5,
        le=0.999,
        description="Filter by specific confidence level"
    )
    
    @validator('end_date')
    def validate_date_range(cls, v, values):
        """Validate that end_date is after start_date."""
        if v and 'start_date' in values and values['start_date']:
            if v <= values['start_date']:
                raise ValueError('end_date must be after start_date')
        return v

class PriceData(BaseModel):
    """Model for current price data from external APIs."""
    
    price: float = Field(description="Current Bitcoin price")
    timestamp: datetime = Field(description="Price timestamp")
    source: str = Field(description="Data source (binance, coingecko)")
    
    @validator('price')
    def validate_price_positive(cls, v):
        """Validate that price is positive."""
        if v <= 0:
            raise ValueError('Price must be positive')
        return v