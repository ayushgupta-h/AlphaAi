# Requirements Document

## Introduction

This document specifies the requirements for a real-time Bitcoin prediction API service that provides live price forecasts with confidence intervals through a REST API endpoint. The system implements the same advanced mathematical engine (GBM with EWMA volatility and Student-t shocks) as the existing backtesting system, but operates as a production-ready service with MongoDB persistence and a frontend dashboard displaying both current predictions and historical timeline charts.

Unlike the existing bitcoin-probabilistic-forecasting system which focuses on backtesting historical data, this service provides real-time predictions for live trading and decision-making, with full persistence of prediction history for analysis and visualization.

## Glossary

- **Prediction_API_Service**: The complete real-time Bitcoin prediction service architecture
- **GBM_Engine**: The mathematical engine implementing modified Geometric Brownian Motion simulation (reused from existing system)
- **EWMA_Calculator**: Component that computes Exponentially Weighted Moving Average for volatility estimation (reused from existing system)
- **Monte_Carlo_Simulator**: Component that generates 10,000 price path simulations per prediction (reused from existing system)
- **Real_Time_Data_Fetcher**: Component responsible for fetching current Bitcoin price from live APIs
- **MongoDB_Persistence_Layer**: NoSQL database component for storing prediction history
- **Prediction_Endpoint**: REST API endpoint /predict that generates and returns probabilistic forecasts
- **Frontend_Dashboard**: Web interface displaying current predictions and historical charts
- **Prediction_Record**: Database document containing timestamp, current_price, lower_bound, upper_bound, and confidence_level
- **Historical_Timeline_Chart**: Visualization showing prediction intervals over time with actual price movements
- **Confidence_Interval**: The range [lower_bound, upper_bound] representing probabilistic forecast bounds
- **Production_Ready_Service**: Service with proper error handling, logging, monitoring, and persistence
- **Live_Price_Feed**: Real-time Bitcoin price data from cryptocurrency exchanges

## Requirements

### Requirement 1: Real-Time Bitcoin Price Fetching

**User Story:** As an API user, I want the service to fetch current Bitcoin prices from live sources, so that predictions are based on the most recent market data.

#### Acceptance Criteria

1. THE Real_Time_Data_Fetcher SHALL fetch current Bitcoin price from a live cryptocurrency API
2. WHEN fetching current price, THE Real_Time_Data_Fetcher SHALL use CoinGecko API or Binance API as primary source
3. THE Real_Time_Data_Fetcher SHALL validate that fetched price is positive and within reasonable bounds
4. WHEN API request fails, THE Real_Time_Data_Fetcher SHALL retry up to 3 times with exponential backoff
5. IF all retry attempts fail, THEN THE Real_Time_Data_Fetcher SHALL raise a descriptive error with failure reason
6. THE Real_Time_Data_Fetcher SHALL return price data with timestamp indicating when price was fetched
7. THE Real_Time_Data_Fetcher SHALL complete price fetching within 5 seconds
8. THE Real_Time_Data_Fetcher SHALL cache price data for up to 60 seconds to avoid excessive API calls

### Requirement 2: REST API Prediction Endpoint

**User Story:** As a client application, I want a REST API endpoint to request Bitcoin price predictions, so that I can integrate forecasts into my trading or analysis system.

#### Acceptance Criteria

1. THE Prediction_API_Service SHALL provide a /predict endpoint accepting HTTP GET requests
2. WHEN /predict is called, THE Prediction_Endpoint SHALL fetch current Bitcoin price using Real_Time_Data_Fetcher
3. THE Prediction_Endpoint SHALL generate probabilistic forecast using the same GBM_Engine as the backtesting system
4. THE Prediction_Endpoint SHALL return JSON response containing current_price, lower_bound, upper_bound, confidence_level, and timestamp
5. THE Prediction_Endpoint SHALL complete prediction generation within 10 seconds
6. WHEN prediction generation fails, THE Prediction_Endpoint SHALL return HTTP 500 with error details
7. THE Prediction_Endpoint SHALL accept optional query parameter confidence_level (default 0.95)
8. THE Prediction_Endpoint SHALL validate that confidence_level is between 0.5 and 0.99

### Requirement 3: MongoDB Prediction History Persistence

**User Story:** As a data analyst, I want all predictions stored in a database, so that I can analyze prediction accuracy and visualize historical trends.

#### Acceptance Criteria

1. THE MongoDB_Persistence_Layer SHALL store every prediction request in a MongoDB collection named predictions
2. WHEN a prediction is generated, THE MongoDB_Persistence_Layer SHALL save a Prediction_Record document
3. THE Prediction_Record SHALL contain fields: timestamp, current_price, lower_bound, upper_bound, confidence_level, and prediction_horizon
4. THE MongoDB_Persistence_Layer SHALL use MongoDB ObjectId as primary key
5. THE MongoDB_Persistence_Layer SHALL create indexes on timestamp field for efficient querying
6. WHEN database write fails, THE MongoDB_Persistence_Layer SHALL log error but not fail the API request
7. THE MongoDB_Persistence_Layer SHALL validate that all numeric fields are finite before storage
8. THE MongoDB_Persistence_Layer SHALL support querying predictions by date range

### Requirement 4: Mathematical Engine Integration

**User Story:** As a quantitative analyst, I want the same proven mathematical model from the backtesting system, so that I can trust the prediction quality and methodology.

#### Acceptance Criteria

1. THE Prediction_API_Service SHALL reuse the GBM_Engine from the existing bitcoin-probabilistic-forecasting system
2. THE Prediction_API_Service SHALL reuse the EWMA_Calculator for volatility estimation
3. THE GBM_Engine SHALL generate exactly 10,000 Monte Carlo price paths per prediction
4. THE GBM_Engine SHALL use Student-t distribution with configurable degrees of freedom for shock generation
5. THE EWMA_Calculator SHALL compute volatility using historical price data with lookback window of 10-24 hours
6. THE Prediction_API_Service SHALL fetch sufficient historical data to compute EWMA volatility estimates
7. THE GBM_Engine SHALL extract percentiles to form confidence intervals (e.g., 2.5th and 97.5th for 95% confidence)
8. THE Prediction_API_Service SHALL validate that mathematical engine produces finite, positive price predictions

### Requirement 5: Frontend Dashboard with Current Predictions

**User Story:** As a trader, I want a web interface showing current Bitcoin predictions, so that I can quickly see the latest forecast without making API calls.

#### Acceptance Criteria

1. THE Frontend_Dashboard SHALL display the most recent Bitcoin price prediction prominently
2. THE Frontend_Dashboard SHALL show current_price, lower_bound, upper_bound, and confidence_level
3. THE Frontend_Dashboard SHALL display prediction timestamp and time since last update
4. THE Frontend_Dashboard SHALL refresh predictions automatically every 5 minutes
5. THE Frontend_Dashboard SHALL provide manual refresh button for immediate updates
6. WHEN prediction data is unavailable, THE Frontend_Dashboard SHALL display appropriate error message
7. THE Frontend_Dashboard SHALL use responsive design for mobile and desktop viewing
8. THE Frontend_Dashboard SHALL highlight when predictions are stale (older than 10 minutes)

### Requirement 6: Historical Timeline Charts

**User Story:** As a portfolio manager, I want to see historical prediction accuracy over time, so that I can assess model performance and calibration.

#### Acceptance Criteria

1. THE Frontend_Dashboard SHALL display Historical_Timeline_Chart showing prediction intervals over time
2. THE Historical_Timeline_Chart SHALL plot actual Bitcoin prices as a line overlaid on prediction intervals
3. THE Historical_Timeline_Chart SHALL render prediction intervals as shaded bands around price line
4. THE Historical_Timeline_Chart SHALL allow users to select time ranges (1 day, 1 week, 1 month)
5. THE Historical_Timeline_Chart SHALL highlight predictions where actual price fell outside confidence intervals
6. THE Historical_Timeline_Chart SHALL load data from MongoDB_Persistence_Layer via API endpoints
7. THE Historical_Timeline_Chart SHALL support zooming and panning for detailed analysis
8. WHEN insufficient historical data exists, THE Historical_Timeline_Chart SHALL display message indicating minimum data requirements

### Requirement 7: Historical Data API Endpoints

**User Story:** As a frontend developer, I want API endpoints to retrieve historical predictions, so that I can populate charts and analysis views.

#### Acceptance Criteria

1. THE Prediction_API_Service SHALL provide /predictions/history endpoint accepting GET requests
2. THE /predictions/history endpoint SHALL accept query parameters: start_date, end_date, and limit
3. THE /predictions/history endpoint SHALL return JSON array of Prediction_Records within specified date range
4. THE /predictions/history endpoint SHALL sort results by timestamp in descending order
5. THE /predictions/history endpoint SHALL limit results to maximum 1000 records per request
6. THE /predictions/history endpoint SHALL validate that start_date is before end_date
7. WHEN no predictions exist in date range, THE /predictions/history endpoint SHALL return empty array
8. THE /predictions/history endpoint SHALL complete queries within 5 seconds

### Requirement 8: Production-Ready Error Handling

**User Story:** As a system administrator, I want comprehensive error handling and logging, so that I can monitor service health and diagnose issues quickly.

#### Acceptance Criteria

1. THE Prediction_API_Service SHALL log all API requests with timestamp, endpoint, and response status
2. THE Prediction_API_Service SHALL log errors with full stack traces and request context
3. THE Prediction_API_Service SHALL return structured JSON error responses with error codes and messages
4. WHEN external API calls fail, THE Prediction_API_Service SHALL log failure details and retry attempts
5. WHEN database operations fail, THE Prediction_API_Service SHALL log errors but continue serving predictions
6. THE Prediction_API_Service SHALL implement health check endpoint /health returning service status
7. THE Prediction_API_Service SHALL use structured logging with configurable log levels
8. THE Prediction_API_Service SHALL NOT expose sensitive information in error messages or logs

### Requirement 9: Service Configuration Management

**User Story:** As a deployment engineer, I want configurable service parameters, so that I can adjust behavior for different environments without code changes.

#### Acceptance Criteria

1. THE Prediction_API_Service SHALL support configuration of MongoDB connection string via environment variable
2. THE Prediction_API_Service SHALL support configuration of external API endpoints and keys
3. THE Prediction_API_Service SHALL support configuration of mathematical model parameters (EWMA window, degrees of freedom)
4. THE Prediction_API_Service SHALL support configuration of service timeouts and retry limits
5. THE Prediction_API_Service SHALL validate all configuration parameters at startup
6. WHEN required configuration is missing, THE Prediction_API_Service SHALL fail to start with descriptive error
7. THE Prediction_API_Service SHALL provide default values for optional configuration parameters
8. THE Prediction_API_Service SHALL support configuration of log levels and output formats

### Requirement 10: API Response Format Standardization

**User Story:** As an API consumer, I want consistent, well-documented response formats, so that I can reliably parse and use prediction data.

#### Acceptance Criteria

1. THE /predict endpoint SHALL return JSON with fields: current_price, lower_bound, upper_bound, confidence_level, timestamp, prediction_horizon
2. THE /predictions/history endpoint SHALL return JSON array with consistent Prediction_Record format
3. THE Prediction_API_Service SHALL include HTTP status codes: 200 for success, 400 for bad requests, 500 for server errors
4. THE Prediction_API_Service SHALL format all timestamps as ISO 8601 strings with timezone information
5. THE Prediction_API_Service SHALL format all price values as floating point numbers with appropriate precision
6. THE Prediction_API_Service SHALL include response headers: Content-Type application/json, Cache-Control directives
7. WHEN validation errors occur, THE Prediction_API_Service SHALL return 400 with detailed field-level error messages
8. THE Prediction_API_Service SHALL include API version in response headers for future compatibility

### Requirement 11: Frontend Technology Stack

**User Story:** As a frontend developer, I want a modern, maintainable web interface, so that I can efficiently build and extend the dashboard functionality.

#### Acceptance Criteria

1. THE Frontend_Dashboard SHALL use React.js framework for component-based architecture
2. THE Frontend_Dashboard SHALL use a charting library (Chart.js or D3.js) for Historical_Timeline_Chart visualization
3. THE Frontend_Dashboard SHALL use Axios or Fetch API for HTTP requests to backend endpoints
4. THE Frontend_Dashboard SHALL implement responsive CSS framework (Bootstrap or Tailwind CSS)
5. THE Frontend_Dashboard SHALL use modern JavaScript (ES6+) with proper error handling
6. THE Frontend_Dashboard SHALL include build system (Webpack, Vite, or Create React App) for development and production
7. THE Frontend_Dashboard SHALL implement proper state management for prediction data and UI state
8. THE Frontend_Dashboard SHALL include loading indicators and error states for all async operations

### Requirement 12: Database Schema Design

**User Story:** As a database administrator, I want well-designed MongoDB collections with proper indexing, so that I can ensure efficient queries and data integrity.

#### Acceptance Criteria

1. THE MongoDB_Persistence_Layer SHALL use collection named predictions with document schema validation
2. THE predictions collection SHALL enforce required fields: timestamp, current_price, lower_bound, upper_bound
3. THE MongoDB_Persistence_Layer SHALL create compound index on (timestamp, confidence_level) for efficient range queries
4. THE MongoDB_Persistence_Layer SHALL create TTL index to automatically expire old predictions after configurable retention period
5. THE MongoDB_Persistence_Layer SHALL validate that lower_bound is less than upper_bound before insertion
6. THE MongoDB_Persistence_Layer SHALL validate that all price fields are positive numbers
7. THE MongoDB_Persistence_Layer SHALL use appropriate data types: Date for timestamp, Number for prices
8. THE MongoDB_Persistence_Layer SHALL support atomic operations for concurrent prediction storage

### Requirement 13: Service Deployment and Containerization

**User Story:** As a DevOps engineer, I want containerized deployment with clear dependencies, so that I can deploy the service reliably across different environments.

#### Acceptance Criteria

1. THE Prediction_API_Service SHALL provide Dockerfile for containerized deployment
2. THE Prediction_API_Service SHALL include requirements.txt with pinned dependency versions
3. THE Prediction_API_Service SHALL support deployment via Docker Compose with MongoDB service
4. THE Prediction_API_Service SHALL expose configurable port for HTTP traffic (default 8000)
5. THE Prediction_API_Service SHALL include health check endpoint for container orchestration
6. THE Prediction_API_Service SHALL handle graceful shutdown on SIGTERM signals
7. THE Prediction_API_Service SHALL include environment variable documentation for deployment
8. THE Prediction_API_Service SHALL support horizontal scaling with stateless service design

### Requirement 14: API Rate Limiting and Caching

**User Story:** As a service operator, I want rate limiting and caching to prevent abuse and improve performance, so that the service remains stable under load.

#### Acceptance Criteria

1. THE Prediction_API_Service SHALL implement rate limiting on /predict endpoint (maximum 60 requests per minute per IP)
2. THE Prediction_API_Service SHALL cache prediction results for 60 seconds to avoid redundant calculations
3. THE Prediction_API_Service SHALL return HTTP 429 when rate limits are exceeded
4. THE Prediction_API_Service SHALL include rate limit headers in responses (X-RateLimit-Limit, X-RateLimit-Remaining)
5. THE Prediction_API_Service SHALL implement cache invalidation when new historical data significantly changes volatility estimates
6. THE Prediction_API_Service SHALL use in-memory caching (Redis optional) for prediction and price data
7. THE Prediction_API_Service SHALL log rate limit violations for monitoring and analysis
8. THE Prediction_API_Service SHALL allow rate limit configuration via environment variables

### Requirement 15: Monitoring and Observability

**User Story:** As a site reliability engineer, I want comprehensive monitoring and metrics, so that I can ensure service availability and performance.

#### Acceptance Criteria

1. THE Prediction_API_Service SHALL expose metrics endpoint /metrics in Prometheus format
2. THE Prediction_API_Service SHALL track metrics: request count, response time, error rate, prediction generation time
3. THE Prediction_API_Service SHALL monitor external API call success rates and latencies
4. THE Prediction_API_Service SHALL monitor database connection health and query performance
5. THE Prediction_API_Service SHALL implement structured logging with correlation IDs for request tracing
6. THE Prediction_API_Service SHALL alert when prediction generation time exceeds 10 seconds
7. THE Prediction_API_Service SHALL track prediction accuracy metrics over time (when actual prices become available)
8. THE Prediction_API_Service SHALL provide dashboard-friendly endpoints for service status and key metrics