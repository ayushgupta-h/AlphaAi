# Implementation Plan: Bitcoin Prediction API

## Overview

This implementation plan creates a production-ready real-time Bitcoin prediction API service that reuses the proven mathematical engine from the existing bitcoin-probabilistic-forecasting system. The service provides REST API endpoints for live price predictions with MongoDB persistence and a comprehensive React frontend dashboard displaying both current predictions and historical timeline charts.

The implementation follows a microservices architecture with FastAPI backend, MongoDB for persistence, Redis for caching, and React frontend, all containerized for scalable deployment.

## Tasks

- [x] 1. Project Structure and Core Dependencies Setup
  - Create FastAPI project structure with proper module organization
  - Set up requirements.txt with pinned versions for FastAPI, MongoDB, Redis, and mathematical dependencies
  - Configure development environment with Docker Compose for local development
  - Set up pytest configuration with property-based testing support using Hypothesis
  - _Requirements: 9.1, 9.7, 13.1, 13.2_

- [ ] 2. Mathematical Engine Integration and Configuration
  - [x] 2.1 Import and configure existing GBM engine components
    - Import GBMEngine, EWMACalculator, and ForecastConfig from bitcoin_forecasting package
    - Create configuration wrapper to adapt existing config for API service
    - Validate mathematical engine produces consistent results with backtesting system
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ]* 2.2 Write property test for mathematical engine consistency
    - **Property 6: Mathematical Engine Consistency**
    - **Validates: Requirements 4.1, 4.3**

  - [x] 2.3 Implement prediction engine service orchestrator
    - Create PredictionEngineService class to coordinate GBM engine, EWMA calculator, and caching
    - Implement volatility computation using EWMA with configurable lookback window
    - Add drift estimation from recent price returns
    - _Requirements: 4.5, 4.6, 4.7_

  - [ ]* 2.4 Write property tests for EWMA calculation correctness
    - **Property 13: EWMA Calculation Correctness**
    - **Validates: Requirements 4.2, 4.5**

  - [ ]* 2.5 Write property tests for Monte Carlo path generation
    - **Property 14: Monte Carlo Path Generation Consistency**
    - **Validates: Requirements 4.3, 4.8**

- [ ] 3. Real-Time Data Fetcher Implementation
  - [x] 3.1 Implement Binance API integration with retry logic
    - Create RealTimeDataFetcher class with async HTTP client
    - Implement current price fetching from Binance API with 5-second timeout
    - Add exponential backoff retry logic (3 attempts: 1s, 2s, 4s delays)
    - _Requirements: 1.1, 1.2, 1.4, 1.7_

  - [ ]* 3.2 Write property tests for price validation correctness
    - **Property 1: Price Validation Correctness**
    - **Validates: Requirements 1.3**

  - [ ]* 3.3 Write property tests for retry logic consistency
    - **Property 2: Retry Logic Consistency**
    - **Validates: Requirements 1.4**

  - [ ] 3.4 Add CoinGecko API as backup data source
    - Implement fallback mechanism when Binance API fails
    - Add API response validation and error handling
    - Implement price data caching with 60-second TTL
    - _Requirements: 1.2, 1.5, 1.8_

  - [ ]* 3.5 Write property tests for error propagation completeness
    - **Property 3: Error Propagation Completeness**
    - **Validates: Requirements 1.5**

  - [ ] 3.6 Implement historical price fetching for volatility calculation
    - Add method to fetch recent hourly closes (10-24 hour lookback)
    - Validate historical data completeness and quality
    - Handle missing data points gracefully
    - _Requirements: 4.5, 4.6_

- [x] 4. Checkpoint - Ensure data fetching and mathematical engine integration works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. MongoDB Persistence Layer Implementation
  - [ ] 5.1 Set up MongoDB connection and database schema
    - Configure async MongoDB client using Motor driver
    - Create predictions collection with schema validation
    - Implement connection pooling and retry logic for database operations
    - _Requirements: 3.1, 12.1, 12.2_

  - [ ] 5.2 Implement prediction storage with indexing
    - Create store_prediction method with document validation
    - Add compound indexes on (timestamp, confidence_level) for efficient queries
    - Implement TTL index for automatic data retention (configurable)
    - _Requirements: 3.2, 3.3, 12.3, 12.4_

  - [ ]* 5.3 Write property tests for database document structure consistency
    - **Property 10: Database Document Structure Consistency**
    - **Validates: Requirements 3.2, 3.3**

  - [ ] 5.4 Implement historical data querying
    - Create query_predictions_by_date_range method with pagination
    - Add get_latest_prediction method for dashboard display
    - Implement efficient sorting and limiting for large datasets
    - _Requirements: 3.8, 7.2, 7.3, 7.4_

  - [ ]* 5.5 Write property tests for date range query accuracy
    - **Property 12: Date Range Query Accuracy**
    - **Validates: Requirements 3.8**

  - [ ] 5.6 Add numeric validation and error handling
    - Implement validation for finite numeric values before storage
    - Add graceful error handling for database write failures
    - Ensure API continues serving predictions even if database writes fail
    - _Requirements: 3.6, 3.7, 8.5_

  - [ ] 5.8 Implement Historical Reconciliation Job

Create a background task or endpoint to fetch predictions where the hour has closed but actual_price is null.

Fetch the actual closed price from Binance for those specific hours.

Calculate the Winkler score for those rows based on the actual price and the original predicted bounds.

Update the MongoDB documents with the actual_price and winkler_score.
- [ ] 6. FastAPI Application and Endpoint Implementation
  - [ ] 6.1 Create FastAPI application with middleware setup
    - Initialize FastAPI app with CORS, logging, and metrics middleware
    - Configure Pydantic models for request/response validation
    - Set up structured logging with correlation IDs
    - _Requirements: 10.1, 10.3, 8.1, 8.7_

  - [ ] 6.2 Implement /predict endpoint with validation
    - Create prediction controller with confidence level validation (0.5-0.99)
    - Implement complete prediction workflow: fetch data → generate prediction → store → return
    - Add request timeout handling (10-second limit)
    - _Requirements: 2.1, 2.2, 2.3, 2.5, 2.7, 2.8_

  - [ ]* 6.3 Write property tests for API response structure completeness
    - **Property 7: API Response Structure Completeness**
    - **Validates: Requirements 2.4**

  - [ ]* 6.4 Write property tests for confidence level processing accuracy
    - **Property 9: Confidence Level Processing Accuracy**
    - **Validates: Requirements 2.7, 2.8**

  - [ ] 6.5 Implement /predictions/history endpoint
    - Create historical data endpoint with date range filtering
    - Add query parameter validation (start_date, end_date, limit)
    - Implement response formatting and pagination (max 1000 records)
    - _Requirements: 7.1, 7.2, 7.5, 7.6, 7.7, 7.8_

  - [ ] 6.6 Add error handling and standardized responses
    - Implement structured error responses with HTTP status codes
    - Add validation error handling with field-level error messages
    - Ensure consistent JSON response format across all endpoints
    - _Requirements: 2.6, 8.3, 10.4, 10.7_

  - [ ]* 6.7 Write property tests for error response format consistency
    - **Property 8: Error Response Format Consistency**
    - **Validates: Requirements 2.6**

- [ ] 7. Caching Layer Implementation with Redis
  - [ ] 7.1 Set up Redis connection and cache manager
    - Configure async Redis client with connection pooling
    - Implement multi-level caching (in-memory L1 + Redis L2)
    - Add cache health monitoring and fallback mechanisms
    - _Requirements: 14.2, 14.6_

  - [ ] 7.2 Implement prediction result caching
    - Cache prediction results with 60-second TTL to avoid redundant calculations
    - Implement cache key strategy based on confidence level and time window
    - Add cache invalidation logic for significant price changes
    - _Requirements: 14.2, 14.5_

  - [ ]* 7.3 Write property tests for cache behavior correctness
    - **Property 5: Cache Behavior Correctness**
    - **Validates: Requirements 1.8**

  - [ ] 7.4 Add price data caching and cache warming
    - Cache current Bitcoin price with 60-second TTL
    - Implement cache warming for frequently requested data
    - Add cache hit/miss metrics for monitoring
    - _Requirements: 1.8, 14.6_

- [ ] 8. Rate Limiting and Security Implementation
  - [ ] 8.1 Implement rate limiting middleware
    - Add IP-based rate limiting (60 requests per minute default)
    - Return HTTP 429 with appropriate headers when limits exceeded
    - Make rate limits configurable via environment variables
    - _Requirements: 14.1, 14.3, 14.4, 14.7_

  - [ ] 8.2 Add request validation and security headers
    - Implement input sanitization and validation
    - Add security headers (CORS, Content-Type, etc.)
    - Ensure no sensitive information exposure in error messages
    - _Requirements: 8.8, 10.6_

- [ ] 9. Health Checks and Monitoring Implementation
  - [ ] 9.1 Implement comprehensive health check endpoint
    - Create /health endpoint checking all dependencies (MongoDB, Redis, external APIs)
    - Add response time measurements for each dependency
    - Return structured health status with degradation levels
    - _Requirements: 8.6, 13.5_

  - [ ] 9.2 Add Prometheus metrics collection
    - Implement /metrics endpoint in Prometheus format
    - Track request count, response time, error rate, prediction generation time
    - Monitor external API call success rates and database performance
    - _Requirements: 15.1, 15.2, 15.3, 15.4_

  - [ ] 9.3 Implement structured logging with correlation IDs
    - Configure JSON-formatted logging with configurable levels
    - Add correlation IDs for request tracing across components
    - Log all API requests, errors, and performance metrics
    - _Requirements: 8.1, 8.2, 15.5_

- [ ] 10. Checkpoint - Ensure backend API is fully functional
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 11. React Frontend Dashboard Implementation
  - [ ] 11.1 Set up React project with modern tooling
    - Initialize React project with Vite build system
    - Configure Tailwind CSS for responsive design
    - Set up Axios for API communication and error handling
    - _Requirements: 11.1, 11.4, 11.6_

  - [ ] 11.2 Create live prediction display components
    - Build CurrentPredictionCard showing latest forecast with confidence intervals
    - Add PredictionMetrics component displaying volatility and model parameters
    - Implement auto-refresh every 5 minutes with manual refresh button
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ] 11.3 Implement responsive layout and error states
    - Create responsive Header component with navigation and status indicators
    - Add loading indicators and error states for all async operations
    - Implement mobile-friendly design with proper breakpoints
    - _Requirements: 5.6, 5.7, 11.7, 11.8_

  - [ ] 11.4 Add prediction staleness detection
    - Highlight when predictions are stale (older than 10 minutes)
    - Display time since last update prominently
    - Add visual indicators for data freshness
    - _Requirements: 5.8_

- [ ] 12. Historical Charts and Visualization Implementation
  - [ ] 12.1 Set up Chart.js integration for timeline visualization
    - Install and configure Chart.js with React wrapper
    - Create base chart component with responsive configuration
    - Implement zoom and pan functionality for detailed analysis
    - _Requirements: 6.3, 6.7, 11.2_

  - [ ] 12.2 Implement historical timeline chart
    - Create TimelineChart component showing prediction intervals as shaded bands
    - Overlay actual Bitcoin prices as line chart on prediction intervals
    - Add time range selector (1 day, 1 week, 1 month)
    - _Requirements: 6.1, 6.2, 6.4_

  - [ ] 12.3 Add prediction accuracy visualization
    - Highlight predictions where actual price fell outside confidence intervals
    - Calculate and display coverage statistics over time
    - Add visual indicators for prediction accuracy metrics
    - _Requirements: 6.5_

  - [ ] 12.4 Implement data loading and error handling for charts
    - Load historical data from /predictions/history API endpoint
    - Handle insufficient data scenarios with appropriate messaging
    - Add loading states and error recovery for chart data
    - _Requirements: 6.6, 6.8_

- [ ] 13. State Management and API Integration
  - [ ] 13.1 Implement React state management
    - Set up Context API or React Query for global state management
    - Create custom hooks for prediction data fetching and caching
    - Implement optimistic updates and error recovery
    - _Requirements: 11.7_

  - [ ] 13.2 Add API integration with error handling
    - Create API service layer with proper error handling
    - Implement retry logic for failed requests
    - Add request/response interceptors for logging and monitoring
    - _Requirements: 11.5, 11.8_

  - [ ] 13.3 Implement real-time data updates
    - Set up periodic data refresh with configurable intervals
    - Add WebSocket support for real-time updates (optional enhancement)
    - Implement efficient data synchronization between components
    - _Requirements: 5.4_

- [ ] 14. Configuration Management and Environment Setup
  - [ ] 14.1 Create comprehensive configuration system
    - Implement Settings class with Pydantic for environment variable management
    - Add validation for all configuration parameters at startup
    - Create environment-specific configuration files (.env.development, .env.production)
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_

  - [ ] 14.2 Add mathematical model parameter configuration
    - Make EWMA window, degrees of freedom, and simulation count configurable
    - Add default values for all optional parameters
    - Implement configuration validation and error reporting
    - _Requirements: 9.3, 9.7_

  - [ ] 14.3 Configure logging and monitoring settings
    - Set up configurable log levels and output formats
    - Add structured logging configuration for different environments
    - Configure metrics collection and export settings
    - _Requirements: 9.8, 8.7_

- [ ] 15. Docker Containerization and Deployment Setup
  - [ ] 15.1 Create multi-stage Dockerfile for backend
    - Build optimized Docker image with Python dependencies
    - Configure non-root user and security best practices
    - Add health check configuration for container orchestration
    - _Requirements: 13.1, 13.4, 13.5, 13.6_

  - [ ] 15.2 Create Docker Compose for local development
    - Set up complete development environment with MongoDB, Redis, API, and frontend
    - Configure volume mounts for development workflow
    - Add environment variable configuration and networking
    - _Requirements: 13.3_

  - [ ] 15.3 Build frontend Docker container
    - Create optimized production build for React frontend
    - Configure Nginx for static file serving and API proxying
    - Add proper caching headers and compression
    - _Requirements: 11.6_

  - [ ] 15.4 Add production deployment configuration
    - Create Kubernetes deployment manifests with proper resource limits
    - Configure horizontal pod autoscaling based on CPU and memory
    - Add production-ready environment variable management
    - _Requirements: 13.7, 13.8_

- [ ] 16. Integration Testing and End-to-End Validation
  - [ ]* 16.1 Write integration tests for complete prediction workflow
    - Test end-to-end prediction generation from API request to database storage
    - Validate external API integration with real Binance/CoinGecko endpoints
    - Test error handling and recovery scenarios
    - _Requirements: 2.1, 2.2, 2.3, 2.4_

  - [ ]* 16.2 Write integration tests for historical data retrieval
    - Test complete historical data workflow from database query to API response
    - Validate date range filtering and pagination functionality
    - Test performance with large datasets
    - _Requirements: 7.1, 7.2, 7.3, 7.4_

  - [ ]* 16.3 Write integration tests for caching behavior
    - Test multi-level cache functionality under various scenarios
    - Validate cache invalidation and TTL behavior
    - Test cache fallback mechanisms when Redis is unavailable
    - _Requirements: 14.2, 14.5, 14.6_

  - [ ]* 16.4 Write performance and load tests
    - Test API performance under concurrent load (100+ requests/minute)
    - Validate rate limiting behavior and response times
    - Test database and cache performance under stress
    - _Requirements: 14.1, 2.5, 7.8_

- [ ] 17. Final System Integration and Deployment Validation
  - [ ] 17.1 Validate complete system deployment
    - Deploy full system using Docker Compose and verify all components
    - Test frontend-backend integration with real data flows
    - Validate monitoring and logging in deployed environment
    - _Requirements: 13.2, 13.3_

  - [ ] 17.2 Perform end-to-end system testing
    - Test complete user workflows from frontend through to database
    - Validate prediction accuracy and mathematical consistency
    - Test error recovery and graceful degradation scenarios
    - _Requirements: 4.8, 8.4, 8.5_

  - [ ] 17.3 Validate monitoring and observability
    - Verify Prometheus metrics collection and alerting
    - Test health check endpoints and dependency monitoring
    - Validate structured logging and correlation ID tracking
    - _Requirements: 15.1, 15.2, 15.3, 15.4, 15.5_

- [ ] 18. Final checkpoint - Ensure complete system is production-ready
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability and validation
- Property-based tests validate universal correctness properties from the design document
- Integration tests ensure external dependencies work correctly in production scenarios
- The implementation reuses existing mathematical components without modification
- Checkpoints ensure incremental validation and provide opportunities for user feedback
- All components are designed for horizontal scaling and production deployment
- The system maintains mathematical consistency with the existing backtesting system