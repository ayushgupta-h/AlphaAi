"""
Bitcoin Probabilistic Forecasting Dashboard

A Streamlit dashboard for live Bitcoin price predictions with historical
backtesting results, interactive visualizations, and cloud persistence.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.graph_objects as go
from datetime import datetime, timezone, timedelta
import logging
from typing import Dict, Any, List

from data_fetcher import get_live_data_for_prediction, fetch_current_price
from engine import generate_prediction
from supabase_client import supabase_client, reconcile_predictions

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Bitcoin Probabilistic Forecasting",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #00ff88;
    }
    .metric-container {
        background-color: #262730;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #00ff88;
    }
    .prediction-box {
        background-color: #1e1e2e;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border: 2px solid #00ff88;
        margin: 1rem 0;
    }
    .price-display {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
    }
    .interval-display {
        font-size: 1.2rem;
        text-align: center;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_backtest_results() -> Dict[str, Any]:
    """Load backtest results from JSON files."""
    try:
        # Load summary statistics
        with open('backtest_summary.json', 'r') as f:
            summary = json.load(f)
        
        # Load detailed predictions
        predictions = []
        with open('backtest_results.jsonl', 'r') as f:
            for line in f:
                pred = json.loads(line.strip())
                predictions.append(pred)
        
        return {
            'summary': summary,
            'predictions': predictions
        }
    except Exception as e:
        st.error(f"Failed to load backtest results: {e}")
        return {'summary': {}, 'predictions': []}


@st.cache_data(ttl=60)  # Cache for 1 minute
def get_live_prediction() -> Dict[str, Any]:
    """Generate live prediction for next hour."""
    try:
        # Get live data
        live_data = get_live_data_for_prediction(lookback_hours=500)
        
        # Generate prediction
        prediction = generate_prediction(
            current_price=live_data['current_price'],
            historical_prices=live_data['historical_prices'],
            confidence_level=0.95
        )
        
        # Add metadata
        prediction['timestamp'] = datetime.now(timezone.utc).isoformat()
        prediction['data_timestamp'] = live_data['data_timestamp'].isoformat()
        
        # Store prediction in Supabase (Part C) - only if connected
        if supabase_client.connected:
            try:
                supabase_client.store_prediction(prediction)
                st.success("✅ Prediction stored in cloud database", icon="☁️")
            except Exception as e:
                st.warning(f"⚠️ Cloud storage failed: {e}")
        else:
            st.info("💡 Cloud storage available - add Supabase credentials to enable", icon="☁️")
        
        return prediction
        
    except Exception as e:
        st.error(f"Failed to generate live prediction: {e}")
        return {}


def display_backtest_metrics(summary: Dict[str, Any]):
    """Display Part A backtest metrics in columns."""
    if not summary:
        st.warning("No backtest summary available")
        return
    
    st.markdown("### 📊 Backtest Performance (Part A)")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Coverage Rate",
            value=f"{summary.get('coverage', 0):.4f}",
            help="Fraction of predictions where actual price fell within bounds"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Average Width",
            value=f"${summary.get('average_width', 0):.2f}",
            help="Average width of prediction intervals"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Mean Winkler Score",
            value=f"{summary.get('mean_winkler_score', 0):.2f}",
            help="Comprehensive interval scoring metric (lower is better)"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="Total Predictions",
            value=f"{summary.get('total_predictions', 0):,}",
            help="Number of predictions in backtest"
        )
        st.markdown('</div>', unsafe_allow_html=True)


def display_live_prediction(prediction: Dict[str, Any]):
    """Display live prediction prominently."""
    if not prediction:
        st.warning("No live prediction available")
        return
    
    st.markdown("### 🔮 Live Prediction (Part B)")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        
        # Current price
        st.markdown(
            f'<div class="price-display">Current Price: ${prediction["current_price"]:,.2f}</div>',
            unsafe_allow_html=True
        )
        
        # Prediction interval
        st.markdown(
            f'<div class="interval-display">'
            f'Next Hour 95% Prediction: '
            f'${prediction["lower_bound"]:,.2f} - ${prediction["upper_bound"]:,.2f}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Interval width
        st.markdown(
            f'<div class="interval-display">'
            f'Interval Width: ${prediction["interval_width"]:,.2f}'
            f'</div>',
            unsafe_allow_html=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        # Additional metrics
        st.markdown("**Model Parameters:**")
        st.write(f"• Volatility: {prediction.get('volatility', 0):.4f}")
        st.write(f"• Drift: {prediction.get('drift', 0):.4f}")
        st.write(f"• Confidence: {prediction.get('confidence_level', 0.95):.1%}")
        
        # Timestamp info
        pred_time = datetime.fromisoformat(prediction['timestamp'].replace('Z', '+00:00'))
        st.write(f"• Generated: {pred_time.strftime('%H:%M:%S UTC')}")


def create_prediction_chart(prediction: Dict[str, Any], historical_data: pd.DataFrame = None):
    """Create interactive Plotly chart with prediction interval."""
    if not prediction:
        st.warning("No prediction data for chart")
        return
    
    # If no historical data provided, fetch some for the chart
    if historical_data is None:
        try:
            from data_fetcher import fetch_binance_data
            historical_data = fetch_binance_data(limit=50)
        except Exception as e:
            st.error(f"Failed to fetch chart data: {e}")
            return
    
    # Take last 50 points for chart
    chart_data = historical_data.tail(50).copy()
    
    fig = go.Figure()
    
    # Historical price line
    fig.add_trace(go.Scatter(
        x=chart_data['timestamp'],
        y=chart_data['close'],
        mode='lines',
        name='Historical Price',
        line=dict(color='#00ff88', width=2)
    ))
    
    # Current price point
    current_time = datetime.now(timezone.utc)
    fig.add_trace(go.Scatter(
        x=[current_time],
        y=[prediction['current_price']],
        mode='markers',
        name='Current Price',
        marker=dict(color='#ff6b6b', size=10, symbol='circle')
    ))
    
    # Prediction interval for next hour
    next_hour = current_time + timedelta(hours=1)
    
    # Create prediction interval as filled area
    fig.add_trace(go.Scatter(
        x=[current_time, next_hour, next_hour, current_time, current_time],
        y=[
            prediction['lower_bound'],
            prediction['lower_bound'],
            prediction['upper_bound'],
            prediction['upper_bound'],
            prediction['lower_bound']
        ],
        fill='toself',
        fillcolor='rgba(255, 107, 107, 0.2)',
        line=dict(color='rgba(255, 107, 107, 0.8)', width=2),
        name='95% Prediction Interval',
        hovertemplate='<b>Prediction Interval</b><br>' +
                     'Lower: $%{y:,.2f}<br>' +
                     'Upper: $%{customdata:,.2f}<extra></extra>',
        customdata=[prediction['upper_bound']] * 5
    ))
    
    # Update layout
    fig.update_layout(
        title='Bitcoin Price with Next-Hour Prediction',
        xaxis_title='Time (UTC)',
        yaxis_title='Price (USD)',
        template='plotly_dark',
        height=500,
        showlegend=True,
        hovermode='x unified'
    )
    
    # Format y-axis as currency
    fig.update_layout(yaxis_tickformat='$,.0f')
    
    st.plotly_chart(fig, use_container_width=True)


def display_cloud_predictions():
    """Display historical predictions from cloud database (Part C)."""
    st.markdown("### ☁️ Cloud Prediction History (Part C)")
    
    # Get connection status
    status = supabase_client.get_connection_status()
    
    if not status['connected']:
        st.warning("⚠️ Cloud database not connected. Using demo mode.")
        
        # Create expandable section with setup instructions
        with st.expander("🔧 Click here to enable cloud persistence"):
            st.markdown("""
            **To enable Part C cloud persistence:**
            
            1. **Create Supabase Account**: Go to [supabase.com](https://supabase.com) and create a free account
            
            2. **Create New Project**: Click "New Project" and wait for setup to complete
            
            3. **Get Credentials**: 
               - Go to Settings → API
               - Copy your Project URL and anon/public key
            
            4. **Add to Streamlit**:
               - Create `.streamlit/secrets.toml` file
               - Add your credentials:
               ```toml
               SUPABASE_URL = "https://your-project.supabase.co"
               SUPABASE_KEY = "your-anon-key-here"
               ```
            
            5. **Create Database Table**: Run this SQL in your Supabase SQL editor:
               ```sql
               CREATE TABLE predictions (
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
                   created_at TIMESTAMPTZ DEFAULT NOW()
               );
               ```
            
            6. **Restart Streamlit**: The cloud features will activate automatically!
            """)
        
        # Show demo data instead
        st.info("📊 Showing demo mode - cloud features will activate when Supabase is configured")
        
        # Create some demo data to show what it would look like
        demo_data = {
            'timestamp': ['2026-05-03 12:00 UTC', '2026-05-03 11:00 UTC', '2026-05-03 10:00 UTC'],
            'current_price': ['$78,397.14', '$78,245.67', '$78,156.23'],
            'lower_bound': ['$78,182.24', '$78,030.45', '$77,941.12'],
            'upper_bound': ['$78,626.06', '$78,474.89', '$78,385.34'],
            'actual_price': ['N/A', '$78,312.45', '$78,201.67'],
            'winkler_score': ['N/A', '1,245.67', '1,189.23']
        }
        
        st.dataframe(
            demo_data,
            use_container_width=True,
            hide_index=True
        )
        
        st.caption("👆 This is demo data. Real predictions will appear here when cloud database is connected.")
        return
    
    # If connected, run the real cloud functionality
    with st.spinner("Reconciling predictions with actual prices..."):
        reconcile_predictions()
    
    # Fetch predictions from cloud
    try:
        predictions = supabase_client.get_all_predictions(limit=100)
        
        if not predictions:
            st.info("No predictions found in cloud database yet. Generate some predictions to see them here!")
            return
        
        # Convert to DataFrame for display
        df = pd.DataFrame(predictions)
        
        # Format columns for display
        display_df = df.copy()
        
        # Format timestamps
        if 'timestamp' in display_df.columns:
            display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M UTC')
        
        # Format prices
        price_cols = ['current_price', 'lower_bound', 'upper_bound', 'actual_price']
        for col in price_cols:
            if col in display_df.columns:
                display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
        
        # Format scores
        if 'winkler_score' in display_df.columns:
            display_df['winkler_score'] = display_df['winkler_score'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
        
        # Select columns for display
        display_cols = ['timestamp', 'current_price', 'lower_bound', 'upper_bound', 'actual_price', 'winkler_score']
        display_cols = [col for col in display_cols if col in display_df.columns]
        
        # Display the dataframe
        st.dataframe(
            display_df[display_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Summary statistics
        if 'actual_price' in df.columns:
            reconciled = df['actual_price'].notna().sum()
            total = len(df)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Predictions", total)
            
            with col2:
                st.metric("Reconciled", reconciled)
            
            with col3:
                if reconciled > 0:
                    avg_score = df['winkler_score'].dropna().mean()
                    st.metric("Avg Winkler Score", f"{avg_score:.2f}" if pd.notna(avg_score) else "N/A")
                else:
                    st.metric("Avg Winkler Score", "N/A")
        
    except Exception as e:
        st.error(f"Failed to load cloud predictions: {e}")


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">₿ Bitcoin Probabilistic Forecasting Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Load backtest results
    backtest_data = load_backtest_results()
    
    # Display Part A metrics
    display_backtest_metrics(backtest_data['summary'])
    
    st.markdown("---")
    
    # Generate and display live prediction
    with st.spinner("Generating live prediction..."):
        live_prediction = get_live_prediction()
    
    display_live_prediction(live_prediction)
    
    st.markdown("---")
    
    # Interactive chart
    st.markdown("### 📈 Interactive Price Chart")
    
    if live_prediction:
        create_prediction_chart(live_prediction)
    
    st.markdown("---")
    
    # Cloud predictions (Part C)
    display_cloud_predictions()
    
    # Refresh button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Refresh All Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Footer with info
    st.markdown("---")
    st.markdown("""
    **Model Information:**
    - Uses EWMA volatility estimation with 24-hour lookback
    - Student-t distributed shocks (5 degrees of freedom)
    - 10,000 Monte Carlo simulations per prediction
    - Strict walk-forward validation (no look-ahead bias)
    - Cloud persistence with automatic reconciliation
    """)


if __name__ == "__main__":
    main()