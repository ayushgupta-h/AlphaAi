"""
Unit tests for Bitcoin Prediction API data models.

This module tests Pydantic models for request/response validation
and data structure correctness.
"""

import pytest
from datetime import datetime
from pydantic import ValidationError

from bitcoin_prediction_api.models.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionRecord,
    HistoryQueryRequest,
    PriceData
)
from bitcoin_prediction_api.models.health import (
    DependencyStatus,
    HealthResponse
)

class TestPredictionModels:
    """Test cases for prediction-related models."""
    
    def test_prediction_request_valid(self):
        """Test valid prediction request creation."""
        request = PredictionRequest(confidence_level=0.95)
        
        assert request.confidence_level == 0.95
    
    def test_prediction_request_invalid_confidence_level(self):
        """Test invalid confidence level validation."""
        with pytest.raises(ValidationError):
            PredictionRequest(confidence_level=1.5)
        
        with pytest.raises(ValidationError):
            PredictionRequest(confidence_level=0.3)
    
    def test_prediction_response_valid(self):
        """Test valid prediction response creation."""
        response = PredictionResponse(
            timestamp=datetime.utcnow(),
            current_price=50000.0,
            lower_bound=48000.0,
            upper_bound=52000.0,
            confidence_level=0.95,
            volatility=0.8,
            drift=0.1,
            interval_width=4000.0
        )
        
        assert response.current_price == 50000.0
        assert response.lower_bound == 48000.0
        assert response.upper_bound == 52000.0
        assert response.symbol == "BTCUSDT"
        assert response.model_version == "gbm-ewma-v1.0"
    
    def test_prediction_response_invalid_bounds(self):
        """Test invalid bounds validation."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                timestamp=datetime.utcnow(),
                current_price=50000.0,
                lower_bound=52000.0,  # Invalid: lower > upper
                upper_bound=48000.0,
                confidence_level=0.95,
                volatility=0.8,
                drift=0.1,
                interval_width=4000.0
            )
    
    def test_prediction_response_negative_values(self):
        """Test negative value validation."""
        with pytest.raises(ValidationError):
            PredictionResponse(
                timestamp=datetime.utcnow(),
                current_price=-50000.0,  # Invalid: negative price
                lower_bound=48000.0,
                upper_bound=52000.0,
                confidence_level=0.95,
                volatility=0.8,
                drift=0.1,
                interval_width=4000.0
            )
    
    def test_history_query_request_valid(self):
        """Test valid history query request."""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 2)
        
        request = HistoryQueryRequest(
            start_date=start_date,
            end_date=end_date,
            limit=100
        )
        
        assert request.start_date == start_date
        assert request.end_date == end_date
        assert request.limit == 100
    
    def test_history_query_request_invalid_date_range(self):
        """Test invalid date range validation."""
        start_date = datetime(2024, 1, 2)
        end_date = datetime(2024, 1, 1)  # Invalid: end before start
        
        with pytest.raises(ValidationError):
            HistoryQueryRequest(
                start_date=start_date,
                end_date=end_date
            )
    
    def test_price_data_valid(self):
        """Test valid price data creation."""
        price_data = PriceData(
            price=50000.0,
            timestamp=datetime.utcnow(),
            source="binance"
        )
        
        assert price_data.price == 50000.0
        assert price_data.source == "binance"
    
    def test_price_data_negative_price(self):
        """Test negative price validation."""
        with pytest.raises(ValidationError):
            PriceData(
                price=-50000.0,  # Invalid: negative price
                timestamp=datetime.utcnow(),
                source="binance"
            )

class TestHealthModels:
    """Test cases for health check models."""
    
    def test_dependency_status_valid(self):
        """Test valid dependency status creation."""
        status = DependencyStatus(
            name="mongodb",
            status="up",
            response_time_ms=50.0,
            last_check=datetime.utcnow()
        )
        
        assert status.name == "mongodb"
        assert status.status == "up"
        assert status.response_time_ms == 50.0
    
    def test_health_response_valid(self):
        """Test valid health response creation."""
        dependencies = {
            "mongodb": DependencyStatus(
                name="mongodb",
                status="up",
                last_check=datetime.utcnow()
            )
        }
        
        response = HealthResponse(
            status="healthy",
            timestamp=datetime.utcnow(),
            version="1.0.0",
            dependencies=dependencies
        )
        
        assert response.status == "healthy"
        assert response.version == "1.0.0"
        assert "mongodb" in response.dependencies