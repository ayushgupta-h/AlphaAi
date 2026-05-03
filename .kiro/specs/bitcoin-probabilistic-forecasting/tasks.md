# Implementation Plan: Bitcoin Probabilistic Forecasting System

## Overview

This implementation plan breaks down the Bitcoin probabilistic forecasting system into discrete, actionable coding tasks. The system will be implemented in Python with a modular architecture separating data ingestion, mathematical modeling, backtesting, and visualization components.

The implementation follows a dependency-ordered approach: foundational modules first (data ingestion, configuration), then core mathematical engine (EWMA, GBM simulation), followed by backtesting framework, and finally the dashboard interface.

## Tasks

- [x] 1. Set up project structure and dependency management
  - Create project directory structure with separate modules
  - Create requirements.txt with numpy, scipy, pandas, requests, streamlit, and their version specifications
  - Create __init__.py files for Python package structure
  - Set up logging configuration module
  - _Requirements: 11.1, 11.3, 12.1, 12.2, 12.3, 14.4_

- [ ] 2. Implement configuration management module
  - [x] 2.1 Create config.py module with configuration dataclass
    - Define configuration parameters: EWMA lookback window, Student-t degrees of freedom, number of Monte Carlo simulations, confidence level
    - Implement parameter validation with range checks
    - Provide sensible defaults (lookback=24, df=5, n_simulations=10000, confidence=0.95)
    - _Requirements: 13.1, 13.2, 13.3, 13.4, 13.5, 13.6, 13.7_
  
  - [x] 2.2 Write unit tests for configuration validation
    - Test valid configuration acceptance
    - Test invalid parameter rejection with descriptive errors
    - Test default value initialization
    - _Requirements: 13.5, 13.7_

- [ ] 3. Implement data ingestion module
  - [x] 3.1 Create data_ingestion.py with Binance API client
    - Implement fetch_binance_data() function with endpoint https://data-api.binance.vision/api/v3/klines
    - Add parameters: symbol=BTCUSDT, interval=1h, limit=1000
    - Parse JSON response into structured format with timestamp, open, high, low, close, volume
    - Convert timestamps from milliseconds to datetime objects
    - _Requirements: 1.1, 1.2, 1.3, 1.7_
  
  - [x] 3.2 Add retry logic with exponential backoff
    - Implement retry decorator with max 3 attempts
    - Add exponential backoff between retries (1s, 2s, 4s)
    - Raise descriptive error after all retries fail
    - _Requirements: 1.4, 1.5, 14.5_
  
  - [x] 3.3 Add data validation and preprocessing
    - Validate minimum 1000 hourly bars received
    - Sort data by timestamp in ascending order
    - Validate all prices are positive
    - Validate timestamps are in chronological order
    - _Requirements: 1.6, 1.8, 15.1, 15.2_
  
  - [x] 3.4 Write unit tests for data ingestion
    - Test successful API response parsing
    - Test retry logic with mock failures
    - Test data validation edge cases
    - _Requirements: 1.1, 1.4, 15.1, 15.2_

- [x] 4. Checkpoint - Ensure data ingestion works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 5. Implement EWMA volatility calculator
  - [x] 5.1 Create ewma.py module with EWMA calculation function
    - Implement compute_ewma_volatility() accepting price series and lookback window
    - Calculate log returns from consecutive close prices
    - Apply exponential decay weights with configurable decay parameter
    - Compute volatility as square root of EWMA variance
    - Annualize volatility by multiplying by sqrt(8760) for hourly data
    - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_
  
  - [x] 5.2 Add fallback handling and validation
    - Return fallback volatility estimate when insufficient historical data
    - Validate computed volatility is positive and finite
    - Add logging for fallback usage
    - _Requirements: 3.7, 3.8, 14.2, 15.6_
  
  - [x] 5.3 Write unit tests for EWMA calculator
    - Test volatility calculation with known data
    - Test fallback behavior with insufficient data
    - Test validation of positive and finite results
    - _Requirements: 3.6, 3.7, 3.8_

- [ ] 6. Implement GBM mathematical engine
  - [x] 6.1 Create gbm_engine.py with Monte Carlo simulation
    - Implement simulate_gbm() function accepting current_price, drift, volatility, time_horizon, degrees_of_freedom
    - Generate exactly 10,000 random shocks using scipy.stats.t.rvs with df parameter
    - Apply modified GBM formula: S_t = S_0 * exp((drift - 0.5*volatility^2)*dt + volatility*sqrt(dt)*shock)
    - Ensure all simulated prices remain positive
    - Return array of 10,000 terminal prices
    - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8_
  
  - [x] 6.2 Implement prediction interval extraction
    - Extract 2.5th percentile as lower bound using numpy.percentile
    - Extract 97.5th percentile as upper bound
    - Validate lower_bound < upper_bound
    - Validate both bounds are positive
    - Return tuple (lower_bound, upper_bound)
    - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5_
  
  - [x] 6.3 Write property test for GBM simulation
    - **Property: All simulated prices must be positive**
    - **Validates: Requirements 2.7**
    - Generate random valid inputs (price > 0, volatility > 0, df > 2)
    - Assert all 10,000 simulated prices > 0
  
  - [x] 6.4 Write property test for prediction intervals
    - **Property: Lower bound must be less than upper bound**
    - **Validates: Requirements 4.4**
    - Generate random valid simulation results
    - Assert lower_bound < upper_bound for all cases
  
  - [x] 6.5 Write unit tests for GBM engine
    - Test simulation with known parameters
    - Test Student-t distribution usage (not normal)
    - Test interval extraction correctness
    - _Requirements: 2.3, 2.4, 4.1, 4.2_

- [x] 7. Checkpoint - Ensure mathematical engine works
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 8. Implement backtesting framework
  - [x] 8.1 Create backtesting.py with walk-forward validation
    - Implement run_backtest() function accepting full price data and configuration
    - Use first 280 hourly bars for initialization
    - Generate predictions for subsequent 720 hourly bars
    - For each prediction at step i, use only data from rows 0 to i-1
    - Validate predictions are generated in chronological order
    - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.8_
  
  - [x] 8.2 Integrate EWMA and GBM for each prediction
    - For each test timestamp, compute volatility using EWMA on historical data only
    - Estimate drift from recent returns (e.g., mean of last 24 hourly returns)
    - Call GBM simulation with current price, drift, volatility, 1-hour horizon
    - Extract prediction interval from simulation results
    - Store prediction with timestamp, actual_price, lower_bound, upper_bound
    - _Requirements: 5.6, 5.7, 15.4_
  
  - [x] 8.3 Write property test for temporal integrity
    - **Property: No future data leakage in predictions**
    - **Validates: Requirements 5.4**
    - For each prediction at index i, verify only data[0:i] was used
    - Assert prediction timestamp > all training data timestamps
  
  - [x] 8.4 Write unit tests for backtesting
    - Test initialization period handling
    - Test sequential prediction generation
    - Test data isolation per prediction
    - _Requirements: 5.1, 5.3, 5.4_

- [ ] 9. Implement evaluation metrics module
  - [x] 9.1 Create metrics.py with coverage metric calculation
    - Implement compute_coverage() accepting predictions list
    - For each prediction, check if lower_bound <= actual_price <= upper_bound
    - Count predictions with actual price inside bounds
    - Divide by total number of predictions
    - Validate result is between 0 and 1
    - Return coverage with 4 decimal places
    - _Requirements: 6.1, 6.2, 6.3, 6.4, 6.5, 6.6_
  
  - [x] 9.2 Implement average width metric calculation
    - Implement compute_average_width() accepting predictions list
    - For each prediction, compute width = upper_bound - lower_bound
    - Validate all widths are positive
    - Compute mean of all widths
    - Return average width with 2 decimal places
    - _Requirements: 7.1, 7.2, 7.3, 7.4, 7.5_
  
  - [x] 9.3 Implement Winkler score calculation
    - Implement compute_winkler_score() accepting predictions list and alpha=0.05
    - For each prediction: if actual inside interval, score = width
    - If actual < lower_bound, score = width + (2/alpha) * (lower_bound - actual)
    - If actual > upper_bound, score = width + (2/alpha) * (actual - upper_bound)
    - Validate all scores are non-negative
    - Compute mean Winkler score
    - Return mean score with 2 decimal places
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7_
  
  - [x] 9.4 Write property test for coverage metric
    - **Property: Coverage must be between 0 and 1**
    - **Validates: Requirements 6.5**
    - Generate random prediction sets
    - Assert 0 <= coverage <= 1
  
  - [ ] 9.5 Write property test for Winkler score
    - **Property: Winkler score must be non-negative**
    - **Validates: Requirements 8.6**
    - Generate random prediction sets
    - Assert all individual scores >= 0
  
  - [ ] 9.6 Write unit tests for metrics
    - Test coverage with known predictions
    - Test average width calculation
    - Test Winkler score edge cases
    - _Requirements: 6.2, 7.1, 8.2, 8.3, 8.4_

- [ ] 10. Implement results persistence
  - [ ] 10.1 Create persistence.py with JSON Lines writer
    - Implement save_results() accepting predictions list and output filename
    - Format each prediction as JSON object with timestamp, actual_price, lower_bound, upper_bound
    - Convert timestamps to ISO 8601 strings
    - Format all prices as floating point numbers
    - Write one prediction per line to backtest_results.jsonl
    - Validate exactly 720 records written
    - _Requirements: 9.1, 9.2, 9.3, 9.4, 9.5, 9.6_
  
  - [ ] 10.2 Add file validation and error handling
    - Validate output is valid JSON Lines format
    - Raise descriptive error on file write failure
    - Add logging for successful save
    - _Requirements: 9.7, 9.8, 14.5_
  
  - [ ] 10.3 Write unit tests for persistence
    - Test JSON Lines format correctness
    - Test timestamp formatting
    - Test error handling on write failure
    - _Requirements: 9.2, 9.7, 9.8_

- [ ] 11. Checkpoint - Ensure backtesting and metrics work
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 12. Implement main execution script
  - [ ] 12.1 Create main.py orchestrating the full pipeline
    - Load configuration from config module
    - Call data ingestion to fetch Binance data
    - Run walk-forward backtesting
    - Compute all evaluation metrics (coverage, average width, Winkler score)
    - Save results to backtest_results.jsonl
    - Print summary statistics to console
    - _Requirements: 11.1, 11.3, 14.1_
  
  - [ ] 12.2 Add comprehensive logging throughout pipeline
    - Log informational messages for major steps (data fetch, backtest start/end, metrics)
    - Log warnings for fallback values or edge cases
    - Log errors with full context on failures
    - Include timestamps in all log messages
    - _Requirements: 14.1, 14.2, 14.3, 14.7_
  
  - [ ] 12.3 Add input validation at pipeline entry points
    - Validate sufficient historical data before backtesting
    - Validate configuration parameters are in acceptable ranges
    - Validate Monte Carlo results are finite
    - Raise descriptive ValueError on validation failures
    - _Requirements: 15.3, 15.4, 15.5, 15.6, 15.7_
  
  - [ ] 12.4 Write integration tests for full pipeline
    - Test end-to-end execution with mock data
    - Test error propagation and handling
    - Test logging output
    - _Requirements: 14.1, 14.5, 15.5_

- [ ] 13. Implement Streamlit dashboard
  - [ ] 13.1 Create dashboard.py with Streamlit interface
    - Load predictions from backtest_results.jsonl
    - Display error message if file missing with instructions to run backtesting
    - Parse JSON Lines format into DataFrame
    - _Requirements: 10.4, 10.5, 10.8_
  
  - [ ] 13.2 Create time series visualization
    - Plot actual prices as line chart
    - Render prediction intervals as shaded regions (fill_between)
    - Highlight predictions where actual fell outside interval (different color)
    - Enable zoom and pan functionality
    - Add axis labels and title
    - _Requirements: 10.1, 10.2, 10.6, 10.7_
  
  - [ ] 13.3 Display summary statistics
    - Show coverage metric with 4 decimal places
    - Show average width with 2 decimal places
    - Show mean Winkler score with 2 decimal places
    - Format as clear summary cards or metrics
    - _Requirements: 10.3_
  
  - [ ] 13.4 Write integration tests for dashboard
    - Test loading of results file
    - Test error handling for missing file
    - Test data parsing correctness
    - _Requirements: 10.4, 10.8_

- [ ] 14. Add documentation and code quality
  - [ ] 14.1 Add docstrings to all public functions and classes
    - Include parameter descriptions with types
    - Include return value descriptions
    - Include example usage where helpful
    - Follow Google or NumPy docstring style
    - _Requirements: 11.6_
  
  - [ ] 14.2 Add inline comments for mathematical formulas
    - Explain EWMA formula and decay parameter
    - Explain GBM formula and Student-t distribution choice
    - Explain Winkler score penalty calculation
    - Document walk-forward validation logic
    - _Requirements: 11.2_
  
  - [ ] 14.3 Add type hints throughout codebase
    - Add type hints to all function parameters
    - Add type hints to all return values
    - Use typing module for complex types (List, Dict, Tuple, Optional)
    - _Requirements: 11.4_
  
  - [ ] 14.4 Format code according to PEP 8
    - Run black formatter on all Python files
    - Ensure consistent indentation and spacing
    - Ensure line length <= 88 characters (black default)
    - _Requirements: 11.5_
  
  - [ ] 14.5 Create README.md with usage instructions
    - Document installation steps (pip install -r requirements.txt)
    - Document how to run backtesting (python main.py)
    - Document how to launch dashboard (streamlit run dashboard.py)
    - Explain configuration options
    - Include example output and interpretation
    - _Requirements: 11.1, 13.1, 13.2, 13.3, 13.4_

- [ ] 15. Final validation and testing
  - [ ] 15.1 Run full backtesting pipeline on real Binance data
    - Execute main.py and verify successful completion
    - Verify backtest_results.jsonl contains 720 predictions
    - Verify coverage metric is approximately 0.95 (within 0.85-1.0 range)
    - Verify all metrics are computed correctly
    - _Requirements: 1.1, 5.2, 6.7, 9.6_
  
  - [ ] 15.2 Launch and verify Streamlit dashboard
    - Run streamlit run dashboard.py
    - Verify time series plot renders correctly
    - Verify prediction intervals are visible
    - Verify summary statistics display correctly
    - Test zoom and pan functionality
    - _Requirements: 10.1, 10.2, 10.3, 10.6_
  
  - [ ] 15.3 Run all unit and property tests
    - Execute pytest on all test files
    - Verify all tests pass
    - Check test coverage is reasonable (>70%)
    - _Requirements: 11.1_

- [ ] 16. Final checkpoint - Complete system validation
  - Ensure all tests pass, verify dashboard works, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for faster MVP delivery
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation at logical breakpoints
- Property tests validate universal correctness properties from the design
- Unit tests validate specific examples and edge cases
- The implementation uses Python with scientific computing stack (numpy, scipy, pandas)
- Walk-forward validation ensures no look-ahead bias in backtesting
- Student-t distribution captures fat-tailed shocks better than normal distribution
- EWMA provides adaptive volatility estimation that captures clustering
