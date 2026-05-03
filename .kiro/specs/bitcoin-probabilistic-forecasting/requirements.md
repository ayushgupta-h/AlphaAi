# Requirements Document

## Introduction

This document specifies the requirements for a Bitcoin probabilistic forecasting system that predicts next-hour price ranges with 95% confidence intervals. The system implements an advanced Geometric Brownian Motion (GBM) simulator with volatility clustering and fat-tailed shock distributions, validated through rigorous walk-forward backtesting on 30 days of hourly Bitcoin price data from Binance.

The architecture is designed for the AlphaI × Polaris Challenge, emphasizing mathematical rigor, strict temporal validation, and professional quantitative development practices.

## Glossary

- **Forecasting_System**: The complete Bitcoin probabilistic forecasting architecture
- **GBM_Engine**: The mathematical engine implementing modified Geometric Brownian Motion simulation
- **EWMA_Calculator**: Component that computes Exponentially Weighted Moving Average for volatility estimation
- **Monte_Carlo_Simulator**: Component that generates 10,000 price path simulations per prediction
- **Data_Ingestion_Module**: Component responsible for fetching and preprocessing Binance API data
- **Backtesting_Engine**: Component that executes walk-forward validation and computes evaluation metrics
- **Dashboard**: Streamlit-based visualization interface for results and metrics
- **Prediction_Interval**: The range [lower_bound, upper_bound] representing 95% confidence interval
- **Walk_Forward_Validation**: Temporal validation methodology where predictions at step i use only data from rows 0 to i-1
- **Coverage_Metric**: Fraction of predictions where actual price falls within predicted interval (target ~0.95)
- **Winkler_Score**: Interval scoring metric that penalizes both width and coverage violations
- **Volatility_Clustering**: Phenomenon where high volatility periods cluster together, modeled via EWMA
- **Fat_Tailed_Distribution**: Student-t distribution capturing extreme price movements better than normal distribution

## Requirements

### Requirement 1: Data Ingestion from Binance API

**User Story:** As a quantitative analyst, I want to fetch historical Bitcoin price data from Binance, so that I can train and validate the forecasting model with real market data.

#### Acceptance Criteria

1. THE Data_Ingestion_Module SHALL fetch data from the public Binance API endpoint https://data-api.binance.vision/api/v3/klines
2. WHEN fetching data, THE Data_Ingestion_Module SHALL request symbol BTCUSDT with interval 1h and limit 1000
3. THE Data_Ingestion_Module SHALL parse the API response into a structured format containing timestamp, open, high, low, close, and volume fields
4. WHEN API request fails, THE Data_Ingestion_Module SHALL retry up to 3 times with exponential backoff
5. IF all retry attempts fail, THEN THE Data_Ingestion_Module SHALL raise a descriptive error with the failure reason
6. THE Data_Ingestion_Module SHALL validate that fetched data contains at least 1000 hourly bars
7. THE Data_Ingestion_Module SHALL convert timestamps from milliseconds to datetime objects
8. THE Data_Ingestion_Module SHALL sort data by timestamp in ascending order

### Requirement 2: Advanced GBM Mathematical Engine

**User Story:** As a quantitative researcher, I want a modified GBM simulator with realistic market dynamics, so that I can generate probabilistic forecasts that capture volatility clustering and fat-tailed shocks.

#### Acceptance Criteria

1. THE GBM_Engine SHALL implement modified Geometric Brownian Motion simulation for price path generation
2. THE GBM_Engine SHALL use Student-t distribution with configurable degrees of freedom for shock generation
3. WHEN generating shocks, THE GBM_Engine SHALL call scipy.stats.t.rvs with df parameter and size 10000
4. THE GBM_Engine SHALL NOT use normal distribution for shock generation
5. THE GBM_Engine SHALL accept current price, drift, volatility, time horizon, and degrees of freedom as input parameters
6. THE GBM_Engine SHALL generate exactly 10000 price paths per prediction
7. FOR ALL simulations, THE GBM_Engine SHALL ensure generated prices remain positive
8. THE GBM_Engine SHALL return an array of 10000 simulated terminal prices

### Requirement 3: EWMA Volatility Estimation

**User Story:** As a quantitative analyst, I want volatility estimates that adapt to recent market conditions, so that I can capture volatility clustering in Bitcoin markets.

#### Acceptance Criteria

1. THE EWMA_Calculator SHALL compute Exponentially Weighted Moving Average of historical returns
2. THE EWMA_Calculator SHALL use a lookback window between 10 and 24 hourly bars
3. WHEN computing EWMA, THE EWMA_Calculator SHALL calculate log returns from consecutive close prices
4. THE EWMA_Calculator SHALL apply exponential decay weights with configurable decay parameter
5. THE EWMA_Calculator SHALL compute volatility as the square root of EWMA variance
6. THE EWMA_Calculator SHALL annualize volatility by multiplying by the square root of hours per year
7. WHEN insufficient historical data exists, THE EWMA_Calculator SHALL return a fallback volatility estimate
8. THE EWMA_Calculator SHALL validate that computed volatility is positive and finite

### Requirement 4: Prediction Interval Extraction

**User Story:** As a risk manager, I want 95% confidence intervals for price predictions, so that I can quantify forecast uncertainty.

#### Acceptance Criteria

1. WHEN Monte Carlo simulation completes, THE GBM_Engine SHALL extract the 2.5th percentile as the lower bound
2. WHEN Monte Carlo simulation completes, THE GBM_Engine SHALL extract the 97.5th percentile as the upper bound
3. THE GBM_Engine SHALL return a Prediction_Interval containing lower_bound and upper_bound
4. THE GBM_Engine SHALL validate that lower_bound is less than upper_bound
5. THE GBM_Engine SHALL validate that both bounds are positive
6. FOR ALL Prediction_Intervals, the theoretical coverage SHALL be approximately 0.95

### Requirement 5: Walk-Forward Backtesting Methodology

**User Story:** As a quantitative researcher, I want strict temporal validation without look-ahead bias, so that I can trust the backtest results reflect real-world performance.

#### Acceptance Criteria

1. THE Backtesting_Engine SHALL use the first 280 hourly bars for initialization
2. THE Backtesting_Engine SHALL generate predictions for the subsequent 720 hourly bars
3. WHEN generating prediction at step i, THE Backtesting_Engine SHALL use only data from rows 0 to i-1
4. THE Backtesting_Engine SHALL NOT use future data for any prediction
5. THE Backtesting_Engine SHALL iterate through each test timestamp sequentially
6. FOR ALL predictions, THE Backtesting_Engine SHALL compute volatility using only historical data available at prediction time
7. THE Backtesting_Engine SHALL store each prediction with its timestamp, actual price, lower bound, and upper bound
8. THE Backtesting_Engine SHALL validate that predictions are generated in chronological order

### Requirement 6: Coverage Metric Computation

**User Story:** As a model validator, I want to measure how often actual prices fall within predicted intervals, so that I can verify the model's calibration.

#### Acceptance Criteria

1. THE Backtesting_Engine SHALL compute Coverage_Metric as the fraction of predictions where actual price falls within Prediction_Interval
2. WHEN computing coverage, THE Backtesting_Engine SHALL check if lower_bound <= actual_price <= upper_bound
3. THE Backtesting_Engine SHALL count the number of predictions with actual price inside bounds
4. THE Backtesting_Engine SHALL divide the count by total number of predictions
5. THE Backtesting_Engine SHALL validate that Coverage_Metric is between 0 and 1
6. THE Backtesting_Engine SHALL report Coverage_Metric with at least 4 decimal places
7. FOR ALL well-calibrated models, Coverage_Metric SHALL be approximately 0.95

### Requirement 7: Average Width Metric Computation

**User Story:** As a quantitative analyst, I want to measure the average width of prediction intervals, so that I can assess forecast precision.

#### Acceptance Criteria

1. THE Backtesting_Engine SHALL compute the width of each Prediction_Interval as upper_bound minus lower_bound
2. THE Backtesting_Engine SHALL compute Average_Width as the mean of all interval widths
3. THE Backtesting_Engine SHALL validate that all widths are positive
4. THE Backtesting_Engine SHALL report Average_Width in the same units as price (USD)
5. THE Backtesting_Engine SHALL report Average_Width with at least 2 decimal places

### Requirement 8: Winkler Score Computation

**User Story:** As a forecasting researcher, I want a comprehensive interval scoring metric, so that I can balance coverage and precision in model evaluation.

#### Acceptance Criteria

1. THE Backtesting_Engine SHALL compute Winkler_Score for each prediction with alpha parameter 0.05
2. WHEN actual price falls inside Prediction_Interval, THE Backtesting_Engine SHALL set score to interval width
3. WHEN actual price falls below lower_bound, THE Backtesting_Engine SHALL set score to width plus 40 times the difference between lower_bound and actual_price
4. WHEN actual price falls above upper_bound, THE Backtesting_Engine SHALL set score to width plus 40 times the difference between actual_price and upper_bound
5. THE Backtesting_Engine SHALL compute mean Winkler_Score across all predictions
6. THE Backtesting_Engine SHALL validate that all individual scores are non-negative
7. THE Backtesting_Engine SHALL report mean Winkler_Score with at least 2 decimal places

### Requirement 9: Backtest Results Persistence

**User Story:** As a data scientist, I want backtest results saved in a structured format, so that I can perform post-hoc analysis and visualization.

#### Acceptance Criteria

1. THE Backtesting_Engine SHALL save predictions to a file named backtest_results.jsonl
2. THE Backtesting_Engine SHALL format the file as JSON Lines with one prediction per line
3. FOR ALL predictions, THE Backtesting_Engine SHALL include timestamp, actual_price, lower_bound, and upper_bound fields
4. THE Backtesting_Engine SHALL format timestamps as ISO 8601 strings
5. THE Backtesting_Engine SHALL format all price values as floating point numbers
6. THE Backtesting_Engine SHALL write exactly 720 prediction records
7. THE Backtesting_Engine SHALL validate that the file is valid JSON Lines format
8. WHEN file write fails, THE Backtesting_Engine SHALL raise a descriptive error

### Requirement 10: Streamlit Dashboard Visualization

**User Story:** As a portfolio manager, I want an interactive dashboard to visualize forecasts and metrics, so that I can quickly assess model performance.

#### Acceptance Criteria

1. THE Dashboard SHALL display a time series plot of actual prices with prediction intervals
2. THE Dashboard SHALL render prediction intervals as shaded regions around actual prices
3. THE Dashboard SHALL display Coverage_Metric, Average_Width, and mean Winkler_Score as summary statistics
4. THE Dashboard SHALL load predictions from backtest_results.jsonl file
5. THE Dashboard SHALL use Streamlit framework for the user interface
6. THE Dashboard SHALL allow users to zoom and pan the time series plot
7. THE Dashboard SHALL highlight predictions where actual price fell outside the interval
8. WHEN backtest_results.jsonl is missing, THE Dashboard SHALL display an error message instructing the user to run backtesting first

### Requirement 11: Modular Code Architecture

**User Story:** As a software engineer, I want clean, modular code with clear separation of concerns, so that I can maintain and extend the system easily.

#### Acceptance Criteria

1. THE Forecasting_System SHALL organize code into separate modules for data ingestion, mathematical engine, backtesting, and dashboard
2. THE Forecasting_System SHALL include detailed comments explaining mathematical formulas and algorithmic choices
3. THE Forecasting_System SHALL define clear interfaces between modules
4. THE Forecasting_System SHALL use type hints for function parameters and return values
5. THE Forecasting_System SHALL follow PEP 8 style guidelines
6. THE Forecasting_System SHALL include docstrings for all public functions and classes
7. THE Forecasting_System SHALL avoid code duplication through appropriate abstraction

### Requirement 12: Dependency Management

**User Story:** As a deployment engineer, I want explicit dependency specifications, so that I can reproduce the environment reliably.

#### Acceptance Criteria

1. THE Forecasting_System SHALL provide a requirements.txt file listing all Python dependencies
2. THE Forecasting_System SHALL specify exact or minimum versions for critical dependencies
3. THE Forecasting_System SHALL include numpy, scipy, pandas, requests, and streamlit as dependencies
4. THE Forecasting_System SHALL NOT include unnecessary or unused dependencies
5. THE Forecasting_System SHALL validate that all imports in code are listed in requirements.txt

### Requirement 13: Configuration Management

**User Story:** As a quantitative researcher, I want configurable model parameters, so that I can experiment with different settings without modifying code.

#### Acceptance Criteria

1. THE Forecasting_System SHALL support configuration of EWMA lookback window (default 10-24 bars)
2. THE Forecasting_System SHALL support configuration of Student-t degrees of freedom parameter
3. THE Forecasting_System SHALL support configuration of number of Monte Carlo simulations (default 10000)
4. THE Forecasting_System SHALL support configuration of confidence level (default 0.95)
5. THE Forecasting_System SHALL validate that all configuration parameters are within valid ranges
6. THE Forecasting_System SHALL provide sensible defaults for all parameters
7. WHEN invalid configuration is provided, THE Forecasting_System SHALL raise a descriptive error

### Requirement 14: Error Handling and Logging

**User Story:** As a system operator, I want comprehensive error handling and logging, so that I can diagnose issues quickly.

#### Acceptance Criteria

1. THE Forecasting_System SHALL log informational messages for major processing steps
2. THE Forecasting_System SHALL log warnings when using fallback values or encountering edge cases
3. THE Forecasting_System SHALL log errors with full context when operations fail
4. THE Forecasting_System SHALL use Python's logging module with configurable log levels
5. WHEN critical errors occur, THE Forecasting_System SHALL provide actionable error messages
6. THE Forecasting_System SHALL NOT expose sensitive information in log messages
7. THE Forecasting_System SHALL include timestamps in all log messages

### Requirement 15: Input Validation

**User Story:** As a quality assurance engineer, I want robust input validation, so that I can prevent invalid data from corrupting results.

#### Acceptance Criteria

1. WHEN receiving price data, THE Forecasting_System SHALL validate that all prices are positive
2. WHEN receiving timestamps, THE Forecasting_System SHALL validate that they are in chronological order
3. WHEN receiving configuration parameters, THE Forecasting_System SHALL validate that they are within acceptable ranges
4. THE Forecasting_System SHALL validate that sufficient historical data exists before generating predictions
5. WHEN validation fails, THE Forecasting_System SHALL raise a descriptive ValueError with the specific validation failure
6. THE Forecasting_System SHALL validate that Monte Carlo simulations produce finite results
7. THE Forecasting_System SHALL validate that computed metrics are within expected ranges
