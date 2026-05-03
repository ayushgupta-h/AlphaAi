"""
Tests for MongoDB persistence layer.

This module tests the MongoDB persistence layer implementation including
connection handling, data validation, and CRUD operations.

Note: These tests require MongoDB to be running. If MongoDB is not available,
the tests will be skipped with appropriate warnings.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from typing import List
import os
import math

from bitcoin_prediction_api.persistence.mongodb_persistence import MongoDBPersistenceLayer
from bitcoin_prediction_api.models.prediction import PredictionRecord


# Skip all tests if MongoDB is not available
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_MONGODB_TESTS", "false").lower() == "true",
    reason="MongoDB tests skipped - set SKIP_MONGODB_TESTS=false to enable"
)


@pytest.fixture
async def persistence_layer():
    """Create a test MongoDB persistence layer."""
    # Use test database
    test_db_url = os.getenv("TEST_MONGODB_URL", "mongodb://localhost:27017/bitcoin_predictions_test")
    persistence = MongoDBPersistenceLayer(
        mongodb_url=test_db_url,
        database_name="bitcoin_predictions_test"
    )
    
    try:
        await persistence.connect()
        yield persistence
    except Exception as e:
        pytest.skip(f"MongoDB not available: {e}")
    finally:
        # Clean up test data
        try:
            if persistence._predictions_collection:
                await persistence._predictions_collection.delete_many({})
            await persistence.disconnect()
        except Exception:
            pass  # Ignore cleanup errors


@pytest.fixture
def sample_prediction():
    """Create a sample prediction record for testing."""
    return PredictionRecord(
        id=None,
        timestamp=datetime.utcnow(),
        current_price=45000.0,
        lower_bound=43000.0,
        upper_bound=47000.0,
        confidence_level=0.95,
        prediction_horizon=1.0,
        volatility=0.25,
        drift=0.05,
        model_version="gbm-ewma-v1.0",
        created_at=datetime.utcnow()
    )


def test_validation_logic():
    """Test validation logic without requiring MongoDB connection."""
    persistence = MongoDBPersistenceLayer()
    
    # Test valid prediction
    valid_prediction = PredictionRecord(
        id=None,
        timestamp=datetime.utcnow(),
        current_price=45000.0,
        lower_bound=43000.0,
        upper_bound=47000.0,
        confidence_level=0.95,
        prediction_horizon=1.0,
        volatility=0.25,
        drift=0.05,
        model_version="gbm-ewma-v1.0",
        created_at=datetime.utcnow()
    )
    
    # Should not raise any exception
    persistence._validate_prediction_record(valid_prediction)
    
    # Test invalid predictions
    invalid_cases = [
        # Negative current_price
        (lambda p: setattr(p, 'current_price', -1000.0), "current_price must be positive"),
        # Invalid confidence_level
        (lambda p: setattr(p, 'confidence_level', 1.5), "confidence_level must be between"),
        # lower_bound >= upper_bound
        (lambda p: (setattr(p, 'lower_bound', 47000.0), setattr(p, 'upper_bound', 43000.0)), "lower_bound must be less than upper_bound"),
        # Infinite values
        (lambda p: setattr(p, 'volatility', float('inf')), "volatility must be finite"),
        # NaN values
        (lambda p: setattr(p, 'drift', float('nan')), "drift must be finite"),
    ]
    
    for modifier, expected_error in invalid_cases:
        test_prediction = PredictionRecord(
            id=None,
            timestamp=datetime.utcnow(),
            current_price=45000.0,
            lower_bound=43000.0,
            upper_bound=47000.0,
            confidence_level=0.95,
            prediction_horizon=1.0,
            volatility=0.25,
            drift=0.05,
            model_version="gbm-ewma-v1.0",
            created_at=datetime.utcnow()
        )
        
        # Apply the modification
        if isinstance(modifier(test_prediction), tuple):
            pass  # Multiple modifications applied
        
        with pytest.raises(ValueError, match=expected_error):
            persistence._validate_prediction_record(test_prediction)


def test_document_conversion():
    """Test conversion between PredictionRecord and MongoDB document."""
    persistence = MongoDBPersistenceLayer()
    
    # Create test prediction
    prediction = PredictionRecord(
        id="test_id",
        timestamp=datetime(2024, 1, 15, 10, 30, 0),
        current_price=45000.0,
        lower_bound=43000.0,
        upper_bound=47000.0,
        confidence_level=0.95,
        prediction_horizon=1.0,
        volatility=0.25,
        drift=0.05,
        model_version="gbm-ewma-v1.0",
        created_at=datetime(2024, 1, 15, 10, 30, 5)
    )
    
    # Convert to document
    document = persistence._prediction_to_document(prediction)
    
    # Verify document structure
    expected_fields = [
        'timestamp', 'current_price', 'lower_bound', 'upper_bound',
        'confidence_level', 'prediction_horizon', 'volatility', 'drift',
        'model_version', 'created_at'
    ]
    
    for field in expected_fields:
        assert field in document
    
    # Verify values
    assert document['timestamp'] == prediction.timestamp
    assert document['current_price'] == prediction.current_price
    assert document['lower_bound'] == prediction.lower_bound
    assert document['upper_bound'] == prediction.upper_bound
    assert document['confidence_level'] == prediction.confidence_level
    
    # Test conversion back to PredictionRecord
    # Add _id field for document conversion
    document['_id'] = "test_object_id"
    
    converted_prediction = persistence._document_to_prediction(document)
    
    # Verify converted prediction
    assert converted_prediction.id == "test_object_id"
    assert converted_prediction.timestamp == prediction.timestamp
    assert converted_prediction.current_price == prediction.current_price
    assert converted_prediction.lower_bound == prediction.lower_bound
    assert converted_prediction.upper_bound == prediction.upper_bound
    assert converted_prediction.confidence_level == prediction.confidence_level


def test_connection_string_masking():
    """Test connection string masking for security."""
    persistence = MongoDBPersistenceLayer()
    
    # Test cases for connection string masking
    test_cases = [
        ("mongodb://user:password@host:27017/db", "mongodb://user:***@host:27017/db"),
        ("mongodb://localhost:27017/db", "mongodb://localhost:27017/db"),
        ("mongodb+srv://user:secret@cluster.mongodb.net/db", "mongodb+srv://user:***@cluster.mongodb.net/db"),
        ("invalid-connection-string", "invalid-connection-string"),
    ]
    
    for original, expected in test_cases:
        masked = persistence._mask_connection_string(original)
        assert masked == expected


@pytest.mark.asyncio
async def test_connection_and_health_check(persistence_layer):
    """Test MongoDB connection and health check."""
    health = await persistence_layer.health_check()
    
    assert health['status'] == 'healthy'
    assert 'response_time_ms' in health
    assert health['database'] == 'bitcoin_predictions_test'
    assert 'collection_stats' in health
    assert 'connection_pool' in health


@pytest.mark.asyncio
async def test_store_prediction(persistence_layer, sample_prediction):
    """Test storing a prediction record."""
    # Store prediction
    prediction_id = await persistence_layer.store_prediction(sample_prediction)
    
    # Verify ID is returned
    assert prediction_id is not None
    assert isinstance(prediction_id, str)
    
    # Verify prediction count
    count = await persistence_layer.get_prediction_count()
    assert count == 1


@pytest.mark.asyncio
async def test_get_latest_prediction(persistence_layer, sample_prediction):
    """Test retrieving the latest prediction."""
    # Initially no predictions
    latest = await persistence_layer.get_latest_prediction()
    assert latest is None
    
    # Store a prediction
    await persistence_layer.store_prediction(sample_prediction)
    
    # Retrieve latest prediction
    latest = await persistence_layer.get_latest_prediction()
    assert latest is not None
    assert latest.current_price == sample_prediction.current_price
    assert latest.confidence_level == sample_prediction.confidence_level


@pytest.mark.asyncio
async def test_query_predictions_by_date_range(persistence_layer):
    """Test querying predictions by date range."""
    now = datetime.utcnow()
    
    # Create predictions with different timestamps
    predictions = []
    for i in range(5):
        prediction = PredictionRecord(
            id=None,
            timestamp=now - timedelta(hours=i),
            current_price=45000.0 + i * 100,
            lower_bound=43000.0 + i * 100,
            upper_bound=47000.0 + i * 100,
            confidence_level=0.95,
            prediction_horizon=1.0,
            volatility=0.25,
            drift=0.05,
            model_version="gbm-ewma-v1.0",
            created_at=now
        )
        predictions.append(prediction)
        await persistence_layer.store_prediction(prediction)
    
    # Query last 3 hours
    start_date = now - timedelta(hours=3)
    end_date = now + timedelta(minutes=1)  # Include current time
    
    results = await persistence_layer.query_predictions_by_date_range(
        start_date=start_date,
        end_date=end_date,
        limit=10
    )
    
    # Should get 4 predictions (0, 1, 2, 3 hours ago)
    assert len(results) == 4
    
    # Results should be sorted by timestamp descending
    for i in range(len(results) - 1):
        assert results[i].timestamp >= results[i + 1].timestamp


@pytest.mark.asyncio
async def test_validation_errors(persistence_layer):
    """Test validation of invalid prediction data."""
    
    # Test negative price
    invalid_prediction = PredictionRecord(
        id=None,
        timestamp=datetime.utcnow(),
        current_price=-1000.0,  # Invalid: negative
        lower_bound=43000.0,
        upper_bound=47000.0,
        confidence_level=0.95,
        prediction_horizon=1.0,
        volatility=0.25,
        drift=0.05,
        model_version="gbm-ewma-v1.0",
        created_at=datetime.utcnow()
    )
    
    with pytest.raises(ValueError, match="current_price must be positive"):
        await persistence_layer.store_prediction(invalid_prediction)
    
    # Test invalid confidence level
    invalid_prediction.current_price = 45000.0
    invalid_prediction.confidence_level = 1.5  # Invalid: > 1.0
    
    with pytest.raises(ValueError, match="confidence_level must be between"):
        await persistence_layer.store_prediction(invalid_prediction)
    
    # Test lower_bound >= upper_bound
    invalid_prediction.confidence_level = 0.95
    invalid_prediction.lower_bound = 47000.0
    invalid_prediction.upper_bound = 43000.0  # Invalid: lower > upper
    
    with pytest.raises(ValueError, match="lower_bound must be less than upper_bound"):
        await persistence_layer.store_prediction(invalid_prediction)


@pytest.mark.asyncio
async def test_date_range_validation(persistence_layer):
    """Test validation of date range parameters."""
    
    now = datetime.utcnow()
    
    # Test start_date >= end_date
    with pytest.raises(ValueError, match="start_date must be before end_date"):
        await persistence_layer.query_predictions_by_date_range(
            start_date=now,
            end_date=now - timedelta(hours=1),
            limit=10
        )
    
    # Test invalid limit
    with pytest.raises(ValueError, match="limit must be between 1 and 1000"):
        await persistence_layer.query_predictions_by_date_range(
            start_date=now - timedelta(hours=1),
            end_date=now,
            limit=1500  # Invalid: > 1000
        )


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])