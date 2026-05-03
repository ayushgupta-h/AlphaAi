"""
Walk-forward backtesting script for Bitcoin probabilistic forecasting.

This script implements strict walk-forward validation where at step i,
only data from rows 0 to i-1 is used for prediction. Generates the
backtest_results.jsonl file required for Part A.
"""

import json
import pandas as pd
from datetime import datetime, timezone
import logging
from typing import List, Dict, Any

from data_fetcher import fetch_binance_data
from engine import generate_prediction, calculate_coverage, calculate_average_width, calculate_mean_winkler_score

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_walk_forward_backtest(
    burn_in_periods: int = 280,
    test_periods: int = 720,
    confidence_level: float = 0.95,
    lookback_window: int = 24,
    output_file: str = "backtest_results.jsonl"
) -> Dict[str, Any]:
    """
    Run walk-forward backtesting with strict temporal validation.
    
    Args:
        burn_in_periods: Number of periods for initialization (default 280)
        test_periods: Number of periods to test (default 720)
        confidence_level: Confidence level for predictions (default 0.95)
        lookback_window: EWMA lookback window (default 24)
        output_file: Output file for results (default backtest_results.jsonl)
    
    Returns:
        Dictionary with backtest summary statistics
    """
    logger.info("Starting walk-forward backtesting...")
    
    # Fetch data (burn_in + test + some buffer)
    total_periods_needed = burn_in_periods + test_periods + 50  # Buffer for safety
    logger.info(f"Fetching {total_periods_needed} hourly bars from Binance...")
    
    df = fetch_binance_data(limit=min(total_periods_needed, 1000))
    
    if len(df) < burn_in_periods + test_periods:
        raise ValueError(
            f"Insufficient data: got {len(df)} bars, need {burn_in_periods + test_periods}"
        )
    
    logger.info(f"Data fetched successfully: {len(df)} bars available")
    logger.info(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    # Initialize results storage
    predictions = []
    
    # Walk-forward loop: at step i, use only data from rows 0 to i-1
    start_idx = burn_in_periods  # Start predictions after burn-in period
    end_idx = min(start_idx + test_periods, len(df) - 1)  # Ensure we don't go beyond available data
    
    logger.info(f"Running predictions from index {start_idx} to {end_idx-1}")
    logger.info(f"This will generate {end_idx - start_idx} predictions")
    
    for i in range(start_idx, end_idx):
        try:
            # At step i, use only data from rows 0 to i-1 for prediction
            historical_data = df.iloc[:i].copy()  # Rows 0 to i-1
            current_bar = df.iloc[i]  # Row i (the bar we're predicting)
            
            # Extract historical closing prices for the model
            historical_prices = historical_data['close'].tolist()
            current_price = historical_prices[-1]  # Last known price (from row i-1)
            actual_price = current_bar['close']  # Actual price we're trying to predict (row i)
            
            # Generate prediction using only historical data (no look-ahead)
            prediction = generate_prediction(
                current_price=current_price,
                historical_prices=historical_prices[:-1],  # Don't include current_price in history
                confidence_level=confidence_level,
                lookback_window=lookback_window
            )
            
            # Create prediction record
            pred_record = {
                'timestamp': current_bar['timestamp'].isoformat(),
                'current_price': current_price,
                'actual_price': actual_price,
                'lower_bound': prediction['lower_bound'],
                'upper_bound': prediction['upper_bound'],
                'confidence_level': confidence_level,
                'volatility': prediction['volatility'],
                'drift': prediction['drift'],
                'interval_width': prediction['interval_width']
            }
            
            predictions.append(pred_record)
            
            # Log progress every 50 predictions
            if (i - start_idx + 1) % 50 == 0:
                logger.info(f"Completed {i - start_idx + 1}/{end_idx - start_idx} predictions")
            
        except Exception as e:
            logger.error(f"Error at step {i}: {e}")
            continue
    
    logger.info(f"Backtesting completed. Generated {len(predictions)} predictions")
    
    # Save results to JSONL file
    logger.info(f"Saving results to {output_file}")
    with open(output_file, 'w') as f:
        for pred in predictions:
            f.write(json.dumps(pred) + '\n')
    
    # Calculate summary statistics
    if predictions:
        coverage = calculate_coverage(predictions)
        avg_width = calculate_average_width(predictions)
        mean_winkler = calculate_mean_winkler_score(predictions, alpha=1-confidence_level)
        
        summary = {
            'total_predictions': len(predictions),
            'coverage': round(coverage, 4),
            'average_width': round(avg_width, 2),
            'mean_winkler_score': round(mean_winkler, 2),
            'confidence_level': confidence_level,
            'burn_in_periods': burn_in_periods,
            'test_periods': len(predictions),
            'lookback_window': lookback_window,
            'start_date': predictions[0]['timestamp'],
            'end_date': predictions[-1]['timestamp']
        }
        
        logger.info("Backtest Summary:")
        logger.info(f"  Total Predictions: {summary['total_predictions']}")
        logger.info(f"  Coverage: {summary['coverage']:.4f}")
        logger.info(f"  Average Width: ${summary['average_width']:.2f}")
        logger.info(f"  Mean Winkler Score: {summary['mean_winkler_score']:.2f}")
        
        # Save summary
        with open('backtest_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        return summary
    
    else:
        logger.error("No predictions generated!")
        return {}


def validate_backtest_results(results_file: str = "backtest_results.jsonl") -> bool:
    """
    Validate the backtest results file.
    
    Args:
        results_file: Path to results file
    
    Returns:
        True if validation passes, False otherwise
    """
    try:
        predictions = []
        with open(results_file, 'r') as f:
            for line in f:
                pred = json.loads(line.strip())
                predictions.append(pred)
        
        logger.info(f"Loaded {len(predictions)} predictions from {results_file}")
        
        # Validate structure
        required_fields = [
            'timestamp', 'current_price', 'actual_price', 'lower_bound', 
            'upper_bound', 'confidence_level', 'volatility', 'drift'
        ]
        
        for i, pred in enumerate(predictions[:5]):  # Check first 5
            for field in required_fields:
                if field not in pred:
                    logger.error(f"Missing field '{field}' in prediction {i}")
                    return False
        
        # Validate data quality
        for pred in predictions:
            if pred['lower_bound'] >= pred['upper_bound']:
                logger.error("Found prediction with lower_bound >= upper_bound")
                return False
            
            if pred['current_price'] <= 0 or pred['actual_price'] <= 0:
                logger.error("Found prediction with non-positive prices")
                return False
        
        logger.info("Backtest results validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return False


if __name__ == "__main__":
    try:
        # Run the backtest
        summary = run_walk_forward_backtest()
        
        if summary:
            # Validate results
            if validate_backtest_results():
                logger.info("✅ Backtesting completed successfully!")
                logger.info("📁 Generated files:")
                logger.info("   - backtest_results.jsonl (detailed predictions)")
                logger.info("   - backtest_summary.json (summary statistics)")
            else:
                logger.error("❌ Backtest validation failed!")
        else:
            logger.error("❌ Backtesting failed!")
            
    except Exception as e:
        logger.error(f"❌ Backtesting failed with error: {e}")
        raise