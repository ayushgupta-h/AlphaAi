# Design Document

## Overview

The Bitcoin Prediction API is a production-ready real-time forecasting service that provides live Bitcoin price predictions through REST API endpoints. The system builds upon the proven mathematical engine from the existing bitcoin-probabilistic-forecasting system, implementing the same GBM (Geometric Brownian Motion) with EWMA volatility and Student-t shocks methodology, but operates as a scalable web service with MongoDB persistence and a comprehensive frontend dashboard.

### Key Design Principles

1. **Mathematical Consistency**: Reuse the exact same GBM engine, EWMA calculator, and Monte Carlo simulator from the existing backtesting system to ensure prediction quality and methodology consistency.

2. **Production Readiness**: Implement comprehensive error handling, structured logging, monitoring, rate limiting, and graceful degradation for reliable operation under load.

3. **Persistence-First Architecture**: Store every prediction in MongoDB for historical analysis, accuracy tracking, and frontend visualization while ensuring the API remains responsive.

4. **Scalable Service Design**: Stateless service architecture with configurable caching, horizontal scaling support, and proper resource management.

5. **Real-Time Responsiveness**: Optimize for sub-10-second prediction generation while maintaining mathematical accuracy through efficient data fetching and computation.

### System Boundaries

**In Scope:**
- Real-time Bitcoin price prediction API service
- MongoDB persistence layer for prediction history
- Enhanced React frontend with historical charts
- Production deployment with Docker containerization
- Comprehensive monitoring and observability
- Rate limiting and caching mechanisms

**Out of Scope:**
- Multi-cryptocurrency support (Bitcoin only)
- User authentication and authorization
- Prediction model training or parameter optimization
- Real-time streaming or WebSocket connections
- Advanced analytics beyond basic accuracy metrics

## Architecture

### High-Level System Architecture

```mermaid
graph TB
    subgraph "External Services"
        BINANCE[Binance API<br/>Live Price Data]
        COINGECKO[CoinGecko API<br/>Backup Price Source]
    end
    
    subgraph "Load Balancer"
        LB[Nginx/ALB<br/>Rate Limiting]
    end
    
    subgraph "Application Layer"
        API1[API Service Instance 1<br/>FastAPI + Uvicorn]
        API2[API Service Instance 2<br/>FastAPI + Uvicorn]
        API3[API Service Instance N<br/>FastAPI + Uvicorn]
    end
    
    subgraph "Caching Layer"
        REDIS[Redis Cache<br/>Price & Prediction Cache]
    end
    
    subgraph "Data Layer"
        MONGO[MongoDB<br/>Prediction History]
    end
    
    subgraph "Frontend"
        REACT[React Dashboard<br/>Vite + Tailwind]
    end
    
    subgraph "Monitoring"
        PROM[Prometheus<br/>Metrics Collection]
        GRAFANA[Grafana<br/>Dashboards]
        LOGS[Structured Logging<br/>JSON Format]
    end
    
    BINANCE --> API1
    BINANCE --> API2
    BINANCE --> API3
    COINGECKO --> API1
    
    LB --> API1
    LB --> API2
    LB --> API3
    
    API1 --> REDIS
    API2 --> REDIS
    API3 --> REDIS
    
    API1 --> MONGO
    API2 --> MONGO
    API3 --> MONGO
    
    REACT --> LB
    
    API1 --> PROM
    API2 --> PROM
    API3 --> PROM
    
    API1 --> LOGS
    API2 --> LOGS
    API3 --> LOGS
```

### Service Architecture Layers

**1. API Gateway Layer**
- Nginx or AWS Application Load Balancer
- SSL termination and HTTPS enforcement
- Rate limiting (60 requests/minute per IP)
- Request routing and load balancing
- CORS handling for frontend access

**2. Application Service Layer**
- FastAPI framework with Pydantic validation
- Uvicorn ASGI server for high performance
- Stateless design for horizontal scaling
- Structured logging with correlation IDs
- Health checks and graceful shutdown

**3. Mathematical Engine Layer**
- Reused GBM engine from existing system
- EWMA volatility calculator
- Monte Carlo simulation (10,000 paths)
- Student-t shock generation
- Prediction interval extraction

**4. Data Access Layer**
- MongoDB async driver (Motor)
- Connection pooling and retry logic
- Schema validation and indexing
- Atomic operations for concurrent writes
- TTL indexes for data retention

**5. Caching Layer**
- Redis for price and prediction caching
- 60-second TTL for price data
- 60-second TTL for prediction results
- Cache invalidation strategies
- Fallback to direct computation

## Components and Interfaces

### Core Service Components

#### 1. Real-Time Data Fetcher

```python
class RealTimeDataFetcher:
    """Fetches current Bitcoin prices from live cryptocurrency APIs."""
    
    async def fetch_current_price(self) -> PriceData:
        """Fetch current BTC price with retry logic and validation."""
        
    async def fetch_historical_prices(self, limit: int = 60) -> List[float]:
        """Fetch recent hourly closes for volatility calculation."""
        
    def validate_price_data(self, price: float) -> bool:
        """Validate price is positive and within reasonable bounds."""
```

**Interface Specifications:**
- Primary source: Binance API (`/api/v3/ticker/price`, `/api/v3/klines`)
- Backup source: CoinGecko API for redundancy
- Timeout: 5 seconds per request
- Retry policy: 3 attempts with exponential backoff (1s, 2s, 4s)
- Validation: Price > 0 and within [1000, 1000000] USD range
- Caching: 60-second TTL to avoid excessive API calls

#### 2. Prediction Engine Service

```python
class PredictionEngineService:
    """Orchestrates the complete prediction workflow."""
    
    def __init__(self, config: ForecastConfig, cache: CacheManager):
        self.gbm_engine = GBMEngine(config)
        self.ewma_calculator = EWMACalculator(config)
        self.cache = cache
        
    async def generate_prediction(self, confidence_level: float = 0.95) -> PredictionResult:
        """Generate complete prediction with caching and validation."""
        
    def _compute_volatility(self, prices: List[float]) -> float:
        """Compute EWMA volatility from historical prices."""
        
    def _estimate_drift(self, prices: List[float]) -> float:
        """Estimate annualized drift from recent returns."""
```

**Mathematical Engine Integration:**
- Direct import from `bitcoin_forecasting.models.gbm_engine`
- Direct import from `bitcoin_forecasting.models.ewma`
- Configuration via `bitcoin_forecasting.config.ForecastConfig`
- No modifications to existing mathematical components
- Validation of all mathematical outputs (finite, positive prices)

#### 3. MongoDB Persistence Layer

```python
class MongoDBPersistenceLayer:
    """Handles all database operations for prediction storage."""
    
    async def store_prediction(self, prediction: PredictionRecord) -> str:
        """Store prediction record with validation and indexing."""
        
    async def query_predictions_by_date_range(
        self, start_date: datetime, end_date: datetime, limit: int = 1000
    ) -> List[PredictionRecord]:
        """Query historical predictions with efficient indexing."""
        
    async def get_latest_prediction(self) -> Optional[PredictionRecord]:
        """Retrieve most recent prediction for dashboard display."""
```

**Database Schema Design:**
```javascript
// predictions collection schema
{
  _id: ObjectId,
  timestamp: ISODate,           // Prediction generation time
  current_price: Number,        // BTC price at prediction time
  lower_bound: Number,          // 2.5th percentile
  upper_bound: Number,          // 97.5th percentile
  confidence_level: Number,     // 0.95 for 95% CI
  prediction_horizon: Number,   // 1.0 (hours)
  volatility: Number,           // EWMA volatility estimate
  drift: Number,                // Estimated drift parameter
  model_version: String,        // "gbm-ewma-v1.0"
  created_at: ISODate          // Document creation time
}
```

**Indexing Strategy:**
```javascript
// Compound index for efficient date range queries
db.predictions.createIndex({ timestamp: -1, confidence_level: 1 })

// TTL index for automatic data retention (optional)
db.predictions.createIndex({ created_at: 1 }, { expireAfterSeconds: 7776000 }) // 90 days

// Single field index for latest prediction queries
db.predictions.createIndex({ timestamp: -1 })
```

#### 4. API Endpoint Controllers

```python
class PredictionController:
    """FastAPI route handlers for prediction endpoints."""
    
    @app.get("/api/predict", response_model=PredictionResponse)
    async def predict(confidence_level: float = 0.95):
        """Generate and return new prediction with persistence."""
        
    @app.get("/api/predictions/history", response_model=List[PredictionRecord])
    async def get_prediction_history(
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 1000
    ):
        """Retrieve historical predictions for frontend charts."""
        
    @app.get("/api/health", response_model=HealthResponse)
    async def health_check():
        """Service health and dependency status."""
        
    @app.get("/api/metrics", response_model=MetricsResponse)
    async def get_metrics():
        """Prometheus-format metrics for monitoring."""
```

### Frontend Architecture Components

#### 1. Enhanced React Dashboard

```typescript
// Component hierarchy
App
├── Header (navigation, status indicators)
├── LivePredictionPanel
│   ├── CurrentPredictionCard
│   ├── PredictionMetrics
│   └── RefreshControls
├── HistoricalChartsPanel
│   ├── TimelineChart (prediction intervals + actual prices)
│   ├── AccuracyChart (coverage over time)
│   └── VolatilityChart (EWMA volatility trends)
└── SystemStatusPanel
    ├── APIHealthIndicator
    ├── DatabaseStatus
    └── PerformanceMetrics
```

#### 2. State Management Architecture

```typescript
// Context-based state management
interface AppState {
  currentPrediction: PredictionResult | null;
  historicalData: PredictionRecord[];
  systemHealth: HealthStatus;
  loading: LoadingState;
  errors: ErrorState;
}

// API integration with React Query/SWR
const usePredictionData = () => {
  const { data, error, mutate } = useSWR('/api/predict', fetcher, {
    refreshInterval: 300000, // 5 minutes
    revalidateOnFocus: true
  });
};
```

#### 3. Charting and Visualization

```typescript
// Chart.js integration for historical timeline
interface TimelineChartProps {
  predictions: PredictionRecord[];
  actualPrices: PricePoint[];
  timeRange: '1d' | '1w' | '1m';
}

// Visualization features:
// - Prediction intervals as shaded bands
// - Actual price line overlay
// - Coverage violation highlighting
// - Interactive zoom and pan
// - Responsive design for mobile
```

## Data Models

### Core Data Structures

#### 1. Prediction Request/Response Models

```python
class PredictionRequest(BaseModel):
    """Request model for prediction generation."""
    confidence_level: float = Field(default=0.95, ge=0.5, le=0.999)
    
class PredictionResponse(BaseModel):
    """Response model for prediction API."""
    symbol: str = "BTCUSDT"
    timestamp: datetime
    current_price: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    prediction_horizon: float = 1.0  # hours
    volatility: float
    drift: float
    interval_width: float
    model_version: str = "gbm-ewma-v1.0"
```

#### 2. Historical Data Models

```python
class PredictionRecord(BaseModel):
    """Database model for stored predictions."""
    id: Optional[str] = Field(alias="_id")
    timestamp: datetime
    current_price: float
    lower_bound: float
    upper_bound: float
    confidence_level: float
    prediction_horizon: float
    volatility: float
    drift: float
    model_version: str
    created_at: datetime
    
class HistoryQueryRequest(BaseModel):
    """Request model for historical data queries."""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    limit: int = Field(default=1000, le=1000)
    confidence_level: Optional[float] = None
```

#### 3. System Health Models

```python
class HealthResponse(BaseModel):
    """Health check response model."""
    status: Literal["healthy", "degraded", "unhealthy"]
    timestamp: datetime
    version: str
    dependencies: Dict[str, DependencyStatus]
    
class DependencyStatus(BaseModel):
    """Individual dependency health status."""
    name: str
    status: Literal["up", "down", "timeout"]
    response_time_ms: Optional[float]
    last_check: datetime
    error_message: Optional[str]
```

### Database Schema Validation

```python
# MongoDB schema validation
prediction_schema = {
    "$jsonSchema": {
        "bsonType": "object",
        "required": ["timestamp", "current_price", "lower_bound", "upper_bound"],
        "properties": {
            "timestamp": {"bsonType": "date"},
            "current_price": {"bsonType": "number", "minimum": 0},
            "lower_bound": {"bsonType": "number", "minimum": 0},
            "upper_bound": {"bsonType": "number", "minimum": 0},
            "confidence_level": {"bsonType": "number", "minimum": 0.5, "maximum": 0.999},
            "volatility": {"bsonType": "number", "minimum": 0},
            "drift": {"bsonType": "number"}
        }
    }
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

Before writing correctness properties, I need to analyze the acceptance criteria to determine which are suitable for property-based testing.

### Property 1: Price Validation Correctness

*For any* price value input to the Real_Time_Data_Fetcher validation, the validation should correctly identify positive finite values within reasonable bounds as valid, and reject negative, zero, infinite, or extreme values as invalid.

**Validates: Requirements 1.3**

### Property 2: Retry Logic Consistency

*For any* simulated API failure scenario, the Real_Time_Data_Fetcher should attempt exactly 3 retries with exponential backoff timing, and the total number of attempts should never exceed 4 (initial + 3 retries).

**Validates: Requirements 1.4**

### Property 3: Error Propagation Completeness

*For any* failure type that exhausts all retry attempts, the Real_Time_Data_Fetcher should raise a descriptive error that includes the failure reason and maintains error context throughout the retry chain.

**Validates: Requirements 1.5**

### Property 4: Response Format Consistency

*For any* successful price fetch operation, the returned data structure should always include a timestamp field indicating when the price was fetched, regardless of the underlying API source.

**Validates: Requirements 1.6**

### Property 5: Cache Behavior Correctness

*For any* sequence of price fetch requests, calls made within 60 seconds should return cached data (same price and timestamp), while calls after 60 seconds should fetch fresh data with updated timestamps.

**Validates: Requirements 1.8**

### Property 6: Mathematical Engine Consistency

*For any* identical set of input parameters (current_price, drift, volatility, time_horizon, degrees_of_freedom), the GBM_Engine should produce statistically equivalent results when using the same random seed, ensuring mathematical reproducibility.

**Validates: Requirements 2.3, 4.1**

### Property 7: API Response Structure Completeness

*For any* successful prediction generation, the JSON response should always contain all required fields (current_price, lower_bound, upper_bound, confidence_level, timestamp) with valid data types and finite values.

**Validates: Requirements 2.4**

### Property 8: Error Response Format Consistency

*For any* prediction generation failure scenario, the API should return HTTP 500 status with a structured JSON error response containing error details and maintaining consistent error format.

**Validates: Requirements 2.6**

### Property 9: Confidence Level Processing Accuracy

*For any* valid confidence level parameter (between 0.5 and 0.99), the prediction endpoint should correctly process the parameter and generate prediction intervals with the specified confidence level.

**Validates: Requirements 2.7, 2.8**

### Property 10: Database Document Structure Consistency

*For any* prediction result that gets persisted, the saved MongoDB document should always contain all required fields (timestamp, current_price, lower_bound, upper_bound, confidence_level, prediction_horizon) with correct data types.

**Validates: Requirements 3.2, 3.3**

### Property 11: Numeric Validation Robustness

*For any* numeric values before database storage, the validation should correctly identify and reject infinite or NaN values while accepting all finite numeric values, ensuring database integrity.

**Validates: Requirements 3.7**

### Property 12: Date Range Query Accuracy

*For any* valid date range parameters, the MongoDB query should return only prediction records with timestamps within the specified range, sorted correctly and respecting the limit parameter.

**Validates: Requirements 3.8**

### Property 13: EWMA Calculation Correctness

*For any* historical price series and lookback window configuration, the EWMA_Calculator should produce volatility estimates that are positive, finite, and mathematically consistent with the exponential weighting formula.

**Validates: Requirements 4.2, 4.5**

### Property 14: Monte Carlo Path Generation Consistency

*For any* GBM simulation parameters, the engine should generate exactly 10,000 price paths, and all generated paths should be positive finite values representing valid future price scenarios.

**Validates: Requirements 4.3, 4.8**

### Property 15: Percentile Extraction Accuracy

*For any* array of simulated terminal prices and confidence level, the percentile extraction should return lower and upper bounds where lower_bound < upper_bound, and both bounds are positive finite values.

**Validates: Requirements 4.7**

## Error Handling

### Error Classification and Response Strategy

#### 1. External API Failures

**Binance/CoinGecko API Errors:**
- **Timeout Errors**: Retry with exponential backoff (1s, 2s, 4s)
- **Rate Limiting (429)**: Implement jittered backoff and switch to backup API
- **Service Unavailable (503)**: Fail fast after retries, return cached data if available
- **Invalid Response**: Validate response format, fallback to backup API

**Error Response Format:**
```json
{
  "error": "external_api_failure",
  "message": "Failed to fetch Bitcoin price after 3 retries",
  "details": {
    "primary_api": "binance",
    "backup_api": "coingecko", 
    "last_error": "Connection timeout after 5 seconds",
    "retry_count": 3
  },
  "timestamp": "2024-01-15T10:30:00Z"
}
```

#### 2. Mathematical Engine Errors

**GBM Simulation Failures:**
- **Invalid Parameters**: Validate inputs before simulation, return 400 Bad Request
- **Numerical Instability**: Detect infinite/NaN results, fallback to conservative estimates
- **Memory Errors**: Reduce simulation count temporarily, log performance degradation

**EWMA Calculation Failures:**
- **Insufficient Data**: Use fallback volatility (0.80) with warning log
- **Invalid Price Series**: Validate price data, reject negative/zero values
- **Calculation Overflow**: Implement numerical safeguards, cap extreme values

#### 3. Database Operation Failures

**MongoDB Connection Issues:**
- **Connection Timeout**: Retry with circuit breaker pattern
- **Write Failures**: Log error but continue serving predictions (graceful degradation)
- **Query Failures**: Return cached results or empty arrays with appropriate status

**Data Validation Errors:**
- **Schema Violations**: Log validation errors, reject invalid documents
- **Constraint Violations**: Handle duplicate keys, invalid ranges gracefully
- **Index Failures**: Continue operation, alert for manual intervention

#### 4. Service-Level Error Handling

**Request Validation Errors:**
```python
@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": "validation_error",
            "message": "Invalid request parameters",
            "details": exc.errors(),
            "timestamp": datetime.utcnow().isoformat() + "Z"
        }
    )
```

**Rate Limiting Errors:**
```python
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests",
            "retry_after": exc.retry_after,
            "limit": exc.limit
        },
        headers={"Retry-After": str(exc.retry_after)}
    )
```

### Circuit Breaker Implementation

```python
class CircuitBreaker:
    """Circuit breaker for external API calls."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
            else:
                raise CircuitBreakerOpenError("Circuit breaker is OPEN")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
```

## Testing Strategy

### Dual Testing Approach

The testing strategy combines property-based testing for mathematical correctness with integration testing for external dependencies and example-based testing for specific scenarios.

#### Property-Based Testing

**Framework**: Hypothesis (Python) for generating test cases
**Configuration**: Minimum 100 iterations per property test
**Coverage**: All mathematical engine components and core business logic

**Property Test Examples:**

```python
from hypothesis import given, strategies as st
import pytest

class TestPredictionEngine:
    
    @given(
        current_price=st.floats(min_value=1000, max_value=100000),
        confidence_level=st.floats(min_value=0.5, max_value=0.99)
    )
    def test_prediction_response_structure_completeness(self, current_price, confidence_level):
        """Feature: bitcoin-prediction-api, Property 7: API Response Structure Completeness"""
        # Generate prediction with random valid inputs
        response = generate_prediction(current_price, confidence_level)
        
        # Verify all required fields are present
        required_fields = ['current_price', 'lower_bound', 'upper_bound', 'confidence_level', 'timestamp']
        for field in required_fields:
            assert field in response
            assert response[field] is not None
            
        # Verify data types and finite values
        assert isinstance(response['current_price'], (int, float))
        assert isinstance(response['lower_bound'], (int, float))
        assert isinstance(response['upper_bound'], (int, float))
        assert math.isfinite(response['current_price'])
        assert math.isfinite(response['lower_bound'])
        assert math.isfinite(response['upper_bound'])
    
    @given(
        prices=st.lists(st.floats(min_value=0.01, max_value=1000000), min_size=25, max_size=100)
    )
    def test_ewma_calculation_correctness(self, prices):
        """Feature: bitcoin-prediction-api, Property 13: EWMA Calculation Correctness"""
        # Compute EWMA volatility
        volatility = compute_ewma_volatility(pd.Series(prices))
        
        # Verify result is positive and finite
        assert volatility > 0
        assert math.isfinite(volatility)
        
        # Verify mathematical consistency (volatility should be reasonable for crypto)
        assert 0.01 <= volatility <= 10.0  # Reasonable bounds for annualized volatility
```

#### Integration Testing

**Framework**: pytest with async support
**Scope**: External API integration, database operations, end-to-end workflows

**Integration Test Examples:**

```python
class TestExternalIntegration:
    
    @pytest.mark.asyncio
    async def test_binance_api_integration(self):
        """Verify Binance API integration works correctly."""
        fetcher = RealTimeDataFetcher()
        
        # Test successful price fetch
        price_data = await fetcher.fetch_current_price()
        assert price_data.price > 0
        assert price_data.timestamp is not None
        
        # Test historical data fetch
        historical_prices = await fetcher.fetch_historical_prices(limit=24)
        assert len(historical_prices) == 24
        assert all(price > 0 for price in historical_prices)
    
    @pytest.mark.asyncio
    async def test_mongodb_persistence_integration(self):
        """Verify MongoDB persistence works correctly."""
        persistence = MongoDBPersistenceLayer()
        
        # Test prediction storage
        prediction = create_test_prediction()
        record_id = await persistence.store_prediction(prediction)
        assert record_id is not None
        
        # Test query functionality
        records = await persistence.query_predictions_by_date_range(
            start_date=datetime.now() - timedelta(hours=1),
            end_date=datetime.now()
        )
        assert len(records) >= 1
        assert any(record.id == record_id for record in records)
```

#### Unit Testing

**Framework**: pytest for specific examples and edge cases
**Scope**: Individual component behavior, error conditions, boundary cases

**Unit Test Examples:**

```python
class TestPredictionValidation:
    
    def test_confidence_level_validation_boundaries(self):
        """Test confidence level validation at boundaries."""
        # Valid boundaries
        assert validate_confidence_level(0.5) == True
        assert validate_confidence_level(0.99) == True
        
        # Invalid boundaries
        assert validate_confidence_level(0.49) == False
        assert validate_confidence_level(1.0) == False
        
    def test_price_validation_edge_cases(self):
        """Test price validation with edge cases."""
        # Valid prices
        assert validate_price(1000.0) == True
        assert validate_price(100000.0) == True
        
        # Invalid prices
        assert validate_price(0.0) == False
        assert validate_price(-1000.0) == False
        assert validate_price(float('inf')) == False
        assert validate_price(float('nan')) == False
```

### Test Configuration and Execution

**Property Test Configuration:**
```python
# pytest.ini configuration
[tool:pytest]
addopts = --hypothesis-show-statistics --hypothesis-verbosity=verbose
markers = 
    property: Property-based tests
    integration: Integration tests
    unit: Unit tests

# Hypothesis settings
from hypothesis import settings, Verbosity

settings.register_profile("ci", max_examples=1000, verbosity=Verbosity.verbose)
settings.register_profile("dev", max_examples=100, verbosity=Verbosity.normal)
```

**Test Execution Strategy:**
```bash
# Run all tests
pytest

# Run only property-based tests
pytest -m property

# Run integration tests (requires external services)
pytest -m integration --hypothesis-profile=ci

# Run with coverage
pytest --cov=bitcoin_prediction_api --cov-report=html
```

### Performance Testing

**Load Testing**: Use locust or artillery to simulate concurrent prediction requests
**Stress Testing**: Test system behavior under high load (1000+ requests/minute)
**Endurance Testing**: Run continuous load for extended periods to detect memory leaks

**Performance Benchmarks:**
- Prediction generation: < 10 seconds (95th percentile)
- Database writes: < 1 second (95th percentile)
- API response time: < 5 seconds (95th percentile)
- Cache hit ratio: > 80% during normal operation

## Deployment Architecture

### Containerization Strategy

#### 1. Multi-Stage Docker Build

```dockerfile
# Dockerfile for Bitcoin Prediction API
FROM python:3.11-slim as builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Set working directory
WORKDIR /app

# Copy application code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Expose port
EXPOSE 8000

# Start application
CMD ["uvicorn", "bitcoin_prediction_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

#### 2. Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MONGODB_URL=mongodb://mongodb:27017/bitcoin_predictions
      - REDIS_URL=redis://redis:6379/0
      - LOG_LEVEL=INFO
      - RATE_LIMIT_PER_MINUTE=60
    depends_on:
      - mongodb
      - redis
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  mongodb:
    image: mongo:7.0
    ports:
      - "27017:27017"
    environment:
      - MONGO_INITDB_DATABASE=bitcoin_predictions
    volumes:
      - mongodb_data:/data/db
      - ./mongo-init.js:/docker-entrypoint-initdb.d/mongo-init.js:ro
    restart: unless-stopped

  redis:
    image: redis:7.2-alpine
    ports:
      - "6379:6379"
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    restart: unless-stopped

  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - VITE_API_BASE=http://localhost:8000
    depends_on:
      - api
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/nginx/ssl:ro
    depends_on:
      - api
      - frontend
    restart: unless-stopped

volumes:
  mongodb_data:
  redis_data:
```

#### 3. Production Kubernetes Deployment

```yaml
# k8s-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bitcoin-prediction-api
  labels:
    app: bitcoin-prediction-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: bitcoin-prediction-api
  template:
    metadata:
      labels:
        app: bitcoin-prediction-api
    spec:
      containers:
      - name: api
        image: bitcoin-prediction-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: MONGODB_URL
          valueFrom:
            secretKeyRef:
              name: mongodb-secret
              key: connection-string
        - name: REDIS_URL
          valueFrom:
            configMapKeyRef:
              name: redis-config
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "250m"
          limits:
            memory: "1Gi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /api/health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5

---
apiVersion: v1
kind: Service
metadata:
  name: bitcoin-prediction-api-service
spec:
  selector:
    app: bitcoin-prediction-api
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: LoadBalancer
```

### Environment Configuration

#### 1. Configuration Management

```python
# config.py
from pydantic import BaseSettings, Field
from typing import Optional

class Settings(BaseSettings):
    """Application configuration with environment variable support."""
    
    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    debug: bool = Field(default=False, env="DEBUG")
    
    # Database Configuration
    mongodb_url: str = Field(env="MONGODB_URL")
    mongodb_database: str = Field(default="bitcoin_predictions", env="MONGODB_DATABASE")
    
    # Cache Configuration
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    cache_ttl_seconds: int = Field(default=60, env="CACHE_TTL_SECONDS")
    
    # External API Configuration
    binance_api_url: str = Field(default="https://data-api.binance.vision", env="BINANCE_API_URL")
    coingecko_api_url: str = Field(default="https://api.coingecko.com/api/v3", env="COINGECKO_API_URL")
    api_timeout_seconds: int = Field(default=5, env="API_TIMEOUT_SECONDS")
    
    # Rate Limiting
    rate_limit_per_minute: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Mathematical Model Configuration
    lookback_window: int = Field(default=24, env="LOOKBACK_WINDOW")
    degrees_of_freedom: float = Field(default=5.0, env="DEGREES_OF_FREEDOM")
    n_simulations: int = Field(default=10000, env="N_SIMULATIONS")
    default_confidence_level: float = Field(default=0.95, env="DEFAULT_CONFIDENCE_LEVEL")
    
    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_format: str = Field(default="json", env="LOG_FORMAT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
```

#### 2. Environment-Specific Configurations

```bash
# .env.development
DEBUG=true
LOG_LEVEL=DEBUG
MONGODB_URL=mongodb://localhost:27017/bitcoin_predictions_dev
REDIS_URL=redis://localhost:6379/0
RATE_LIMIT_PER_MINUTE=1000

# .env.production
DEBUG=false
LOG_LEVEL=INFO
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/bitcoin_predictions
REDIS_URL=redis://redis-cluster:6379/0
RATE_LIMIT_PER_MINUTE=60
```

### Scaling and Load Balancing

#### 1. Horizontal Scaling Strategy

```yaml
# Horizontal Pod Autoscaler
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: bitcoin-prediction-api-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: bitcoin-prediction-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

#### 2. Load Balancer Configuration

```nginx
# nginx.conf
upstream api_backend {
    least_conn;
    server api1:8000 max_fails=3 fail_timeout=30s;
    server api2:8000 max_fails=3 fail_timeout=30s;
    server api3:8000 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name api.bitcoin-predictions.com;
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api:10m rate=60r/m;
    
    location /api/ {
        limit_req zone=api burst=10 nodelay;
        
        proxy_pass http://api_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 10s;
        proxy_read_timeout 15s;
        
        # Health checks
        proxy_next_upstream error timeout invalid_header http_500 http_502 http_503;
    }
}
```

## Data Flow Architecture

### Request/Response Flow Patterns

#### 1. Prediction Generation Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Cache
    participant DataFetcher
    participant MathEngine
    participant Database
    participant ExtAPI as External API

    Client->>API: GET /api/predict?confidence_level=0.95
    API->>Cache: Check cached prediction
    
    alt Cache Hit (< 60s old)
        Cache-->>API: Return cached prediction
        API-->>Client: Return prediction response
    else Cache Miss
        API->>DataFetcher: Fetch current price & history
        DataFetcher->>ExtAPI: Get BTC price & klines
        ExtAPI-->>DataFetcher: Price data
        DataFetcher-->>API: Validated price data
        
        API->>MathEngine: Generate prediction
        MathEngine->>MathEngine: Compute EWMA volatility
        MathEngine->>MathEngine: Run GBM simulation (10k paths)
        MathEngine->>MathEngine: Extract percentiles
        MathEngine-->>API: Prediction result
        
        par Async Operations
            API->>Database: Store prediction (async)
            API->>Cache: Cache prediction (60s TTL)
        end
        
        API-->>Client: Return prediction response
    end
```

#### 2. Historical Data Query Flow

```mermaid
sequenceDiagram
    participant Client
    participant API
    participant Database
    participant Cache

    Client->>API: GET /api/predictions/history?start_date=...&end_date=...
    API->>API: Validate query parameters
    
    API->>Cache: Check cached query result
    alt Cache Hit
        Cache-->>API: Return cached results
    else Cache Miss
        API->>Database: Query predictions by date range
        Database-->>API: Prediction records
        API->>Cache: Cache query result (300s TTL)
    end
    
    API->>API: Format response
    API-->>Client: Return historical data
```

### Data Persistence Patterns

#### 1. Write-Through Caching Pattern

```python
class PredictionService:
    """Service implementing write-through caching for predictions."""
    
    async def generate_and_store_prediction(self, confidence_level: float) -> PredictionResponse:
        """Generate prediction with write-through caching."""
        
        # Check cache first
        cache_key = f"prediction:{confidence_level}:{int(time.time() // 60)}"
        cached_result = await self.cache.get(cache_key)
        if cached_result:
            return cached_result
        
        # Generate new prediction
        prediction = await self._generate_prediction(confidence_level)
        
        # Write to database and cache simultaneously
        await asyncio.gather(
            self.database.store_prediction(prediction),
            self.cache.set(cache_key, prediction, ttl=60)
        )
        
        return prediction
```

#### 2. Event-Driven Persistence

```python
class PredictionEventHandler:
    """Handle prediction events for persistence and notifications."""
    
    async def on_prediction_generated(self, event: PredictionGeneratedEvent):
        """Handle prediction generation event."""
        
        # Store in database
        await self.database.store_prediction(event.prediction)
        
        # Update metrics
        await self.metrics.increment_prediction_count()
        
        # Invalidate related caches
        await self.cache.invalidate_pattern("history:*")
        
        # Emit analytics event
        await self.analytics.track_prediction(event.prediction)
```

### Caching Strategy

#### 1. Multi-Level Caching Architecture

```python
class CacheManager:
    """Multi-level cache management for optimal performance."""
    
    def __init__(self):
        self.l1_cache = {}  # In-memory cache (fastest)
        self.l2_cache = RedisCache()  # Redis cache (shared)
        
    async def get(self, key: str) -> Optional[Any]:
        """Get value with L1 -> L2 fallback."""
        
        # Check L1 cache first
        if key in self.l1_cache:
            return self.l1_cache[key]
        
        # Check L2 cache
        value = await self.l2_cache.get(key)
        if value:
            # Populate L1 cache
            self.l1_cache[key] = value
            return value
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int):
        """Set value in both cache levels."""
        
        # Set in both caches
        self.l1_cache[key] = value
        await self.l2_cache.set(key, value, ttl)
        
        # Schedule L1 eviction
        asyncio.create_task(self._evict_l1_after_ttl(key, ttl))
```

#### 2. Cache Invalidation Strategies

```python
class CacheInvalidationService:
    """Intelligent cache invalidation based on data changes."""
    
    async def invalidate_on_new_prediction(self, prediction: PredictionRecord):
        """Invalidate caches when new prediction is stored."""
        
        # Invalidate current prediction cache
        await self.cache.delete("current_prediction")
        
        # Invalidate historical queries that might include this prediction
        date_key = prediction.timestamp.strftime("%Y-%m-%d")
        await self.cache.delete_pattern(f"history:*{date_key}*")
        
        # Invalidate metrics cache
        await self.cache.delete("metrics:*")
    
    async def invalidate_on_price_update(self, new_price: float):
        """Invalidate price-dependent caches."""
        
        # Only invalidate if price change is significant (> 1%)
        cached_price = await self.cache.get("current_price")
        if cached_price and abs(new_price - cached_price) / cached_price > 0.01:
            await self.cache.delete("current_price")
            await self.cache.delete("prediction:*")
```

## Monitoring and Observability

### Metrics Collection

#### 1. Application Metrics

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Request metrics
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'Request duration', ['endpoint'])
ACTIVE_CONNECTIONS = Gauge('api_active_connections', 'Active connections')

# Business metrics
PREDICTIONS_GENERATED = Counter('predictions_generated_total', 'Total predictions generated')
PREDICTION_GENERATION_TIME = Histogram('prediction_generation_seconds', 'Prediction generation time')
CACHE_HIT_RATE = Gauge('cache_hit_rate', 'Cache hit rate percentage')

# External API metrics
EXTERNAL_API_CALLS = Counter('external_api_calls_total', 'External API calls', ['provider', 'status'])
EXTERNAL_API_DURATION = Histogram('external_api_duration_seconds', 'External API call duration', ['provider'])

# Database metrics
DB_OPERATIONS = Counter('database_operations_total', 'Database operations', ['operation', 'status'])
DB_CONNECTION_POOL = Gauge('database_connection_pool_size', 'Database connection pool size')

@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Middleware to collect request metrics."""
    start_time = time.time()
    
    response = await call_next(request)
    
    # Record metrics
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    REQUEST_DURATION.labels(endpoint=request.url.path).observe(time.time() - start_time)
    
    return response
```

#### 2. Health Check Endpoints

```python
@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check with dependency status."""
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "1.0.0",
        "dependencies": {}
    }
    
    # Check MongoDB
    try:
        await database.ping()
        health_status["dependencies"]["mongodb"] = {
            "status": "up",
            "response_time_ms": await measure_db_latency()
        }
    except Exception as e:
        health_status["dependencies"]["mongodb"] = {
            "status": "down",
            "error": str(e)
        }
        health_status["status"] = "degraded"
    
    # Check Redis
    try:
        await cache.ping()
        health_status["dependencies"]["redis"] = {
            "status": "up",
            "response_time_ms": await measure_cache_latency()
        }
    except Exception as e:
        health_status["dependencies"]["redis"] = {
            "status": "down",
            "error": str(e)
        }
    
    # Check external APIs
    try:
        await data_fetcher.health_check()
        health_status["dependencies"]["external_apis"] = {
            "status": "up",
            "binance": "available",
            "coingecko": "available"
        }
    except Exception as e:
        health_status["dependencies"]["external_apis"] = {
            "status": "degraded",
            "error": str(e)
        }
    
    return health_status

@app.get("/api/metrics")
async def metrics_endpoint():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

### Structured Logging

#### 1. Logging Configuration

```python
import structlog
from pythonjsonlogger import jsonlogger

def configure_logging():
    """Configure structured logging with JSON format."""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

# Usage in application
logger = structlog.get_logger(__name__)

async def generate_prediction(confidence_level: float):
    """Generate prediction with structured logging."""
    
    correlation_id = str(uuid.uuid4())
    logger = logger.bind(correlation_id=correlation_id, confidence_level=confidence_level)
    
    logger.info("Starting prediction generation")
    
    try:
        # Fetch data
        logger.info("Fetching current price data")
        price_data = await data_fetcher.fetch_current_price()
        logger.info("Price data fetched", current_price=price_data.price)
        
        # Generate prediction
        logger.info("Running mathematical engine")
        prediction = await math_engine.generate_prediction(price_data, confidence_level)
        logger.info("Prediction generated", 
                   lower_bound=prediction.lower_bound,
                   upper_bound=prediction.upper_bound,
                   interval_width=prediction.interval_width)
        
        return prediction
        
    except Exception as e:
        logger.error("Prediction generation failed", 
                    error=str(e), 
                    error_type=type(e).__name__)
        raise
```

#### 2. Alert Configuration

```yaml
# alertmanager.yml
groups:
- name: bitcoin-prediction-api
  rules:
  - alert: HighErrorRate
    expr: rate(api_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"
  
  - alert: PredictionGenerationSlow
    expr: histogram_quantile(0.95, prediction_generation_seconds) > 10
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "Prediction generation is slow"
      description: "95th percentile prediction time is {{ $value }} seconds"
  
  - alert: DatabaseConnectionFailure
    expr: up{job="mongodb"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Database connection failed"
      description: "MongoDB is not responding"
  
  - alert: ExternalAPIFailure
    expr: rate(external_api_calls_total{status="error"}[5m]) > 0.5
    for: 3m
    labels:
      severity: warning
    annotations:
      summary: "External API failure rate high"
      description: "External API error rate is {{ $value }} per second"
```

### Performance Monitoring

#### 1. Application Performance Monitoring (APM)

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Configure tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Instrument prediction generation
@tracer.start_as_current_span("generate_prediction")
async def generate_prediction(confidence_level: float):
    """Generate prediction with distributed tracing."""
    
    current_span = trace.get_current_span()
    current_span.set_attribute("confidence_level", confidence_level)
    
    with tracer.start_as_current_span("fetch_price_data") as span:
        price_data = await data_fetcher.fetch_current_price()
        span.set_attribute("current_price", price_data.price)
    
    with tracer.start_as_current_span("mathematical_engine") as span:
        prediction = await math_engine.generate_prediction(price_data, confidence_level)
        span.set_attribute("interval_width", prediction.interval_width)
    
    current_span.set_attribute("prediction_successful", True)
    return prediction
```

This comprehensive design document provides a production-ready architecture for the Bitcoin Prediction API service, covering all aspects from mathematical engine integration to deployment and monitoring. The design ensures scalability, reliability, and maintainability while preserving the proven mathematical methodology from the existing system.