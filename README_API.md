# Bitcoin Prediction API

A production-ready real-time Bitcoin price prediction service that provides live forecasts through REST API endpoints. Built with FastAPI, MongoDB, and Redis, this service reuses the proven mathematical engine from the existing bitcoin-probabilistic-forecasting system.

## Features

- **Real-time Predictions**: Generate Bitcoin price predictions using advanced GBM mathematical models
- **REST API**: Clean, well-documented API endpoints for prediction generation and historical data
- **MongoDB Persistence**: Store all predictions for historical analysis and accuracy tracking
- **Redis Caching**: High-performance caching for improved response times
- **Production Ready**: Comprehensive logging, monitoring, health checks, and error handling
- **Scalable Architecture**: Stateless design with horizontal scaling support
- **React Dashboard**: Modern web interface for visualization and monitoring

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and Docker Compose
- MongoDB (or use Docker Compose)
- Redis (or use Docker Compose)

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd bitcoin-prediction-api
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

4. **Start services with Docker Compose**
   ```bash
   docker-compose up -d mongodb redis
   ```

5. **Run the API server**
   ```bash
   uvicorn bitcoin_prediction_api.main:app --reload
   ```

6. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/api/health
   - Generate Prediction: http://localhost:8000/api/predict

### Docker Deployment

1. **Build and run with Docker Compose**
   ```bash
   docker-compose up --build
   ```

2. **Access the services**
   - API: http://localhost:8000
   - MongoDB: localhost:27017
   - Redis: localhost:6379

## API Endpoints

### Prediction Generation

```http
GET /api/predict?confidence_level=0.95
```

Generate a new Bitcoin price prediction with specified confidence level.

**Response:**
```json
{
  "symbol": "BTCUSDT",
  "timestamp": "2024-01-15T10:30:00Z",
  "current_price": 45000.0,
  "lower_bound": 43200.0,
  "upper_bound": 46800.0,
  "confidence_level": 0.95,
  "prediction_horizon": 1.0,
  "volatility": 0.85,
  "drift": 0.12,
  "interval_width": 3600.0,
  "model_version": "gbm-ewma-v1.0"
}
```

### Historical Data

```http
GET /api/predictions/history?start_date=2024-01-01&end_date=2024-01-15&limit=100
```

Retrieve historical prediction data for analysis and visualization.

### Health Check

```http
GET /api/health
```

Check service health and dependency status.

## Configuration

The service is configured through environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host address |
| `PORT` | `8000` | Server port |
| `DEBUG` | `false` | Enable debug mode |
| `MONGODB_URL` | `mongodb://localhost:27017/bitcoin_predictions` | MongoDB connection string |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `RATE_LIMIT_PER_MINUTE` | `60` | API rate limit per IP |
| `LOG_LEVEL` | `INFO` | Logging level |

See `.env.example` for complete configuration options.

## Testing

### Run All Tests

```bash
pytest
```

### Run Specific Test Categories

```bash
# Unit tests only
pytest -m unit

# Integration tests (requires services)
pytest -m integration

# Property-based tests
pytest -m property
```

### Test Coverage

```bash
pytest --cov=bitcoin_prediction_api --cov-report=html
```

## Architecture

### System Components

- **FastAPI Application**: REST API server with async support
- **Prediction Service**: Business logic for prediction generation
- **Data Fetcher**: External API integration (Binance, CoinGecko)
- **Mathematical Engine**: GBM simulation with EWMA volatility
- **MongoDB Persistence**: Prediction storage and querying
- **Redis Cache**: Performance optimization and rate limiting
- **Health Service**: Dependency monitoring and health checks

### Data Flow

1. Client requests prediction via REST API
2. Service fetches current Bitcoin price from external APIs
3. Mathematical engine generates probabilistic forecast
4. Prediction is stored in MongoDB and cached in Redis
5. Formatted response is returned to client

## Mathematical Model

The service uses the same proven mathematical engine as the backtesting system:

- **Geometric Brownian Motion (GBM)**: Core price simulation model
- **EWMA Volatility**: Exponentially weighted moving average for volatility estimation
- **Student-t Shocks**: Heavy-tailed distribution for realistic price movements
- **Monte Carlo Simulation**: 10,000 price paths for statistical accuracy

## Monitoring and Observability

### Health Checks

- `/api/health`: Comprehensive dependency health status
- Docker health checks for container orchestration
- Dependency response time monitoring

### Logging

- Structured JSON logging with correlation IDs
- Configurable log levels and formats
- Request/response logging with timing

### Metrics

- `/api/metrics`: Prometheus-format metrics
- Request count, response time, error rate
- Prediction generation performance
- Cache hit rates and database performance

## Production Deployment

### Docker

```bash
docker build -t bitcoin-prediction-api .
docker run -p 8000:8000 bitcoin-prediction-api
```

### Kubernetes

```bash
kubectl apply -f k8s-deployment.yaml
```

### Environment-Specific Configuration

- `.env.development`: Development settings
- `.env.production`: Production settings
- Docker Compose profiles for different environments

## Development

### Code Quality

```bash
# Format code
black bitcoin_prediction_api/

# Sort imports
isort bitcoin_prediction_api/

# Type checking
mypy bitcoin_prediction_api/

# Linting
flake8 bitcoin_prediction_api/
```

### Adding New Features

1. Create feature branch
2. Implement changes with tests
3. Run full test suite
4. Update documentation
5. Submit pull request

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions, issues, or contributions:

- Create an issue on GitHub
- Check the documentation at `/docs`
- Review the API specification at `/api/docs`