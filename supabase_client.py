"""
Supabase client for cloud persistence of Bitcoin predictions.

This module handles the connection to Supabase (PostgreSQL) for storing
and retrieving prediction data for Part C tiebreaker requirements.
"""

import os
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
import logging

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("Supabase not available. Install with: pip install supabase")

logger = logging.getLogger(__name__)


class SupabaseClient:
    """Client for interacting with Supabase database."""
    
    def __init__(self):
        """Initialize Supabase client."""
        self.client: Optional[Client] = None
        self.connected = False
        
        if SUPABASE_AVAILABLE:
            self._connect()
    
    def _connect(self):
        """Connect to Supabase using environment variables or Streamlit secrets."""
        if not SUPABASE_AVAILABLE:
            logger.info("Supabase package not installed. Cloud features disabled.")
            return
            
        try:
            # Try to get credentials from Streamlit secrets first
            url = None
            key = None
            
            try:
                import streamlit as st
                if hasattr(st, 'secrets'):
                    url = st.secrets["SUPABASE_URL"]
                    key = st.secrets["SUPABASE_KEY"]
            except Exception as e:
                logger.debug(f"Could not read Streamlit secrets: {e}")
                pass
            
            # Fallback to environment variables
            if not url or not key:
                url = os.getenv("SUPABASE_URL")
                key = os.getenv("SUPABASE_KEY")
            
            if not url or not key:
                logger.info("Supabase credentials not found. Cloud features disabled.")
                return
            
            self.client = create_client(url, key)
            self.connected = True
            logger.info("Successfully connected to Supabase")
            
            # Ensure predictions table exists
            self._ensure_table_exists()
            
        except Exception as e:
            logger.warning(f"Failed to connect to Supabase: {e}")
            self.connected = False
    
    def _ensure_table_exists(self):
        """Ensure the predictions table exists with correct schema."""
        if not self.connected:
            return
        
        try:
            # Try to query the table to see if it exists
            result = self.client.table('predictions').select('id').limit(1).execute()
            logger.info("Predictions table exists")
        except Exception as e:
            logger.warning(f"Predictions table may not exist: {e}")
            # Note: In a real deployment, you would create the table here
            # For this demo, we'll assume it exists or will be created manually
    
    def store_prediction(self, prediction: Dict[str, Any]) -> bool:
        """
        Store a prediction in Supabase.
        
        Args:
            prediction: Dictionary containing prediction data
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            logger.warning("Not connected to Supabase, skipping storage")
            return False
        
        try:
            # Prepare data for insertion
            data = {
                'timestamp': prediction['timestamp'],
                'current_price': prediction['current_price'],
                'lower_bound': prediction['lower_bound'],
                'upper_bound': prediction['upper_bound'],
                'confidence_level': prediction['confidence_level'],
                'volatility': prediction['volatility'],
                'drift': prediction['drift'],
                'interval_width': prediction['interval_width'],
                'actual_price': None,  # Will be filled later during reconciliation
                'winkler_score': None,  # Will be calculated later
                'created_at': datetime.now(timezone.utc).isoformat()
            }
            
            result = self.client.table('predictions').insert(data).execute()
            
            if result.data:
                logger.info(f"Successfully stored prediction for {prediction['timestamp']}")
                return True
            else:
                logger.error("Failed to store prediction - no data returned")
                return False
                
        except Exception as e:
            logger.error(f"Error storing prediction: {e}")
            return False
    
    def get_predictions_for_reconciliation(self) -> List[Dict[str, Any]]:
        """
        Get predictions that need reconciliation (actual_price is null).
        
        Returns:
            List of predictions needing reconciliation
        """
        if not self.connected:
            return []
        
        try:
            result = self.client.table('predictions').select('*').is_('actual_price', 'null').execute()
            
            if result.data:
                logger.info(f"Found {len(result.data)} predictions needing reconciliation")
                return result.data
            else:
                logger.info("No predictions need reconciliation")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching predictions for reconciliation: {e}")
            return []
    
    def update_prediction_with_actual(self, prediction_id: int, actual_price: float, winkler_score: float) -> bool:
        """
        Update a prediction with actual price and Winkler score.
        
        Args:
            prediction_id: ID of the prediction to update
            actual_price: Actual observed price
            winkler_score: Calculated Winkler score
        
        Returns:
            True if successful, False otherwise
        """
        if not self.connected:
            return False
        
        try:
            result = self.client.table('predictions').update({
                'actual_price': actual_price,
                'winkler_score': winkler_score,
                'updated_at': datetime.now(timezone.utc).isoformat()
            }).eq('id', prediction_id).execute()
            
            if result.data:
                logger.info(f"Successfully updated prediction {prediction_id} with actual price ${actual_price:.2f}")
                return True
            else:
                logger.error(f"Failed to update prediction {prediction_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error updating prediction {prediction_id}: {e}")
            return False
    
    def get_all_predictions(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Get all predictions for display.
        
        Args:
            limit: Maximum number of predictions to return
        
        Returns:
            List of all predictions
        """
        if not self.connected:
            return []
        
        try:
            result = self.client.table('predictions').select('*').order('timestamp', desc=True).limit(limit).execute()
            
            if result.data:
                logger.info(f"Retrieved {len(result.data)} predictions from database")
                return result.data
            else:
                logger.info("No predictions found in database")
                return []
                
        except Exception as e:
            logger.error(f"Error fetching all predictions: {e}")
            return []
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Get connection status information.
        
        Returns:
            Dictionary with connection status
        """
        return {
            'connected': self.connected,
            'supabase_available': SUPABASE_AVAILABLE,
            'client_initialized': self.client is not None
        }


# Global instance
supabase_client = SupabaseClient()


def reconcile_predictions():
    """
    Reconcile predictions with actual prices.
    
    This function implements the "magic trick" - it finds predictions
    where the hour has closed and updates them with actual prices.
    """
    if not supabase_client.connected:
        logger.warning("Supabase not connected, skipping reconciliation")
        return
    
    # Get predictions needing reconciliation
    predictions = supabase_client.get_predictions_for_reconciliation()
    
    if not predictions:
        logger.info("No predictions need reconciliation")
        return
    
    # Import here to avoid circular imports
    from data_fetcher import fetch_binance_data
    from engine import calculate_winkler_score
    
    try:
        # Fetch recent historical data
        recent_data = fetch_binance_data(limit=100)
        
        reconciled_count = 0
        
        for pred in predictions:
            try:
                # Parse prediction timestamp
                pred_time = datetime.fromisoformat(pred['timestamp'].replace('Z', '+00:00'))
                
                # Check if the hour has closed (at least 1 hour ago)
                now = datetime.now(timezone.utc)
                if (now - pred_time).total_seconds() < 3600:  # Less than 1 hour
                    continue
                
                # Find the corresponding actual price
                # Look for the bar that corresponds to the prediction hour
                target_hour = pred_time.replace(minute=0, second=0, microsecond=0)
                
                # Find matching bar in historical data
                actual_price = None
                for _, row in recent_data.iterrows():
                    bar_time = row['timestamp'].replace(minute=0, second=0, microsecond=0)
                    if bar_time == target_hour:
                        actual_price = row['close']
                        break
                
                if actual_price is None:
                    logger.warning(f"Could not find actual price for prediction at {pred_time}")
                    continue
                
                # Calculate Winkler score
                winkler_score = calculate_winkler_score(
                    actual_price=actual_price,
                    lower_bound=pred['lower_bound'],
                    upper_bound=pred['upper_bound'],
                    alpha=1 - pred['confidence_level']
                )
                
                # Update the prediction
                success = supabase_client.update_prediction_with_actual(
                    prediction_id=pred['id'],
                    actual_price=actual_price,
                    winkler_score=winkler_score
                )
                
                if success:
                    reconciled_count += 1
                    logger.info(f"Reconciled prediction {pred['id']}: actual=${actual_price:.2f}, score={winkler_score:.2f}")
                
            except Exception as e:
                logger.error(f"Error reconciling prediction {pred.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"Reconciliation complete: {reconciled_count} predictions updated")
        
    except Exception as e:
        logger.error(f"Error during reconciliation: {e}")


# SQL for creating the predictions table (for reference)
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMPTZ NOT NULL,
    current_price DECIMAL(12,2) NOT NULL,
    lower_bound DECIMAL(12,2) NOT NULL,
    upper_bound DECIMAL(12,2) NOT NULL,
    confidence_level DECIMAL(5,4) NOT NULL,
    volatility DECIMAL(8,6) NOT NULL,
    drift DECIMAL(8,6) NOT NULL,
    interval_width DECIMAL(12,2) NOT NULL,
    actual_price DECIMAL(12,2),
    winkler_score DECIMAL(12,2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX IF NOT EXISTS idx_predictions_actual_price ON predictions(actual_price);
"""