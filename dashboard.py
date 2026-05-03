"""
Streamlit dashboard for Bitcoin Probabilistic Forecasting System.

Displays backtesting results loaded from backtest_results.jsonl with:
- Time series visualization of actual prices vs prediction intervals
- Highlighted violations (where actual price fell outside interval)
- Summary statistics cards (coverage, average width, Winkler score)

Usage:
    streamlit run dashboard.py

Requirements:
    Run main.py first to generate backtest_results.jsonl.
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page configuration is called inside run_dashboard() to avoid
# triggering Streamlit at import time (which breaks pytest collection).
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RESULTS_FILE = "backtest_results.jsonl"
INSIDE_COLOR = "rgba(99, 179, 237, 0.20)"  # light blue for covered intervals
VIOLATION_COLOR = "rgba(252, 129, 74, 0.35)"  # orange for violations


# ---------------------------------------------------------------------------
# Data loading helpers (Requirement 13.1)
# ---------------------------------------------------------------------------


def load_predictions(filename: str = RESULTS_FILE) -> List[Dict[str, Any]]:
    """
    Load predictions from a JSON Lines file.

    Parameters
    ----------
    filename : str
        Path to JSON Lines results file.

    Returns
    -------
    List[Dict[str, Any]]
        List of prediction dicts with keys: timestamp, actual_price,
        lower_bound, upper_bound.

    Raises
    ------
    FileNotFoundError
        If the results file does not exist.
    ValueError
        If the file contains invalid JSON Lines format.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)

    records = []
    with open(filename, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} in '{filename}': {e}"
                ) from e

    return records


def parse_to_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Parse prediction records into a Pandas DataFrame.

    Converts ISO 8601 timestamp strings to datetime objects and
    adds a boolean 'covered' column indicating whether the actual
    price falls within the prediction interval.

    Parameters
    ----------
    records : List[Dict[str, Any]]
        Prediction records loaded from JSONL file.

    Returns
    -------
    pd.DataFrame
        DataFrame sorted by timestamp with columns:
        timestamp, actual_price, lower_bound, upper_bound, covered.
    """
    df = pd.DataFrame(records)

    # Parse timestamps
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Ensure prices are floats
    for col in ("actual_price", "lower_bound", "upper_bound"):
        df[col] = df[col].astype(float)

    # Add coverage flag (Requirement 13.2)
    df["covered"] = (df["lower_bound"] <= df["actual_price"]) & (
        df["actual_price"] <= df["upper_bound"]
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Metric computation helpers (Requirement 13.3)
# ---------------------------------------------------------------------------


def _compute_coverage(df: pd.DataFrame) -> float:
    return round(df["covered"].mean(), 4)


def _compute_average_width(df: pd.DataFrame) -> float:
    return round((df["upper_bound"] - df["lower_bound"]).mean(), 2)


def _compute_winkler_score(df: pd.DataFrame, alpha: float = 0.05) -> float:
    penalty_factor = 2.0 / alpha
    scores = []
    for _, row in df.iterrows():
        width = row["upper_bound"] - row["lower_bound"]
        if row["covered"]:
            score = width
        elif row["actual_price"] < row["lower_bound"]:
            score = width + penalty_factor * (row["lower_bound"] - row["actual_price"])
        else:
            score = width + penalty_factor * (row["actual_price"] - row["upper_bound"])
        scores.append(score)
    return round(sum(scores) / len(scores), 2)


# ---------------------------------------------------------------------------
# Custom CSS for premium styling
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 2rem;
    border: 1px solid rgba(99,179,237,0.2);
}

.main-header h1 {
    color: #e2e8f0;
    font-size: 2rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: #94a3b8;
    margin: 0.5rem 0 0 0;
    font-size: 0.95rem;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #1a2744 100%);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    text-align: center;
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,179,237,0.15);
}

.metric-label {
    color: #94a3b8;
    font-size: 0.82rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 0.5rem;
}

.metric-value {
    color: #63b3ed;
    font-size: 1.85rem;
    font-weight: 700;
    line-height: 1.1;
}

.metric-sub {
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 0.3rem;
}

.quality-badge {
    display: inline-block;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.3px;
}

.quality-excellent { background: rgba(72,187,120,0.2); color: #68d391; }
.quality-good      { background: rgba(99,179,237,0.2); color: #63b3ed; }
.quality-fair      { background: rgba(237,180,99,0.2); color: #f6ad55; }
.quality-poor      { background: rgba(252,129,74,0.2); color: #fc8150; }

.section-title {
    color: #e2e8f0;
    font-size: 1.1rem;
    font-weight: 600;
    margin: 1.5rem 0 1rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(99,179,237,0.2);
}

.info-box {
    background: rgba(99,179,237,0.08);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    color: #94a3b8;
    font-size: 0.9rem;
    line-height: 1.6;
}

.stApp {
    background: #0d1117;
}
</style>
"""


# ---------------------------------------------------------------------------
# Chart rendering (Requirement 13.2)
# ---------------------------------------------------------------------------


def render_time_series_chart(df: pd.DataFrame) -> None:
    """
    Render the time series chart using Plotly via Streamlit.

    Plots:
    - Shaded prediction intervals (blue for covered, orange for violations)
    - Actual price line
    - Violation scatter markers

    Enables zoom and pan natively via Plotly.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: timestamp, actual_price,
        lower_bound, upper_bound, covered.
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.error("plotly is required for charting. Install with: pip install plotly")
        return

    fig = go.Figure()

    # Split into covered and violated rows for separate shading
    covered_df = df[df["covered"]]
    violated_df = df[~df["covered"]]

    # --- Shaded bands: covered (blue) ---
    if not covered_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.concat(
                    [covered_df["timestamp"], covered_df["timestamp"].iloc[::-1]]
                ),
                y=pd.concat(
                    [covered_df["upper_bound"], covered_df["lower_bound"].iloc[::-1]]
                ),
                fill="toself",
                fillcolor=INSIDE_COLOR,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Interval (covered)",
            )
        )

    # --- Shaded bands: violations (orange) ---
    if not violated_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pd.concat(
                    [violated_df["timestamp"], violated_df["timestamp"].iloc[::-1]]
                ),
                y=pd.concat(
                    [violated_df["upper_bound"], violated_df["lower_bound"].iloc[::-1]]
                ),
                fill="toself",
                fillcolor=VIOLATION_COLOR,
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=True,
                name="95% Interval (violation)",
            )
        )

    # --- Actual price line ---
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["actual_price"],
            mode="lines",
            name="Actual Price",
            line=dict(color="#63b3ed", width=1.5),
            hovertemplate="<b>%{x}</b><br>Actual: $%{y:,.2f}<extra></extra>",
        )
    )

    # --- Violation markers ---
    if not violated_df.empty:
        fig.add_trace(
            go.Scatter(
                x=violated_df["timestamp"],
                y=violated_df["actual_price"],
                mode="markers",
                name="Violations",
                marker=dict(color="#fc8150", size=5, symbol="x"),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Actual: $%{y:,.2f}<br>"
                    "<span style='color:#fc8150'>Outside interval</span>"
                    "<extra></extra>"
                ),
            )
        )

    # --- Layout ---
    fig.update_layout(
        title=dict(
            text="BTC/USDT — Actual Prices vs 95% Prediction Intervals",
            font=dict(color="#e2e8f0", size=16),
        ),
        xaxis=dict(
            title="Date / Time",
            title_font=dict(color="#94a3b8"),
            tickfont=dict(color="#94a3b8"),
            gridcolor="rgba(148,163,184,0.1)",
            rangeslider=dict(visible=True),
        ),
        yaxis=dict(
            title="Price (USD)",
            title_font=dict(color="#94a3b8"),
            tickfont=dict(color="#94a3b8"),
            tickprefix="$",
            tickformat=",.0f",
            gridcolor="rgba(148,163,184,0.1)",
        ),
        legend=dict(
            font=dict(color="#94a3b8"),
            bgcolor="rgba(30,41,59,0.8)",
            bordercolor="rgba(99,179,237,0.2)",
        ),
        paper_bgcolor="rgba(13,17,23,0)",
        plot_bgcolor="rgba(30,41,59,0.4)",
        hovermode="x unified",
        height=520,
        margin=dict(l=10, r=10, t=50, b=10),
    )

    # Zoom/pan enabled by default in Plotly
    st.plotly_chart(fig, width="stretch", key="btc_backtest_chart")


# ---------------------------------------------------------------------------
# Summary statistics display (Requirement 13.3)
# ---------------------------------------------------------------------------


def render_metrics(df: pd.DataFrame) -> None:
    """
    Display summary statistics as styled metric cards.

    Shows:
    - Coverage with 4 decimal places
    - Average width with 2 decimal places
    - Mean Winkler score with 2 decimal places

    Parameters
    ----------
    df : pd.DataFrame
        Predictions DataFrame.
    """
    coverage = _compute_coverage(df)
    avg_width = _compute_average_width(df)
    winkler = _compute_winkler_score(df)

    violations = int((~df["covered"]).sum())
    n = len(df)

    # Coverage quality badge
    if coverage >= 0.93:
        q_class, q_label = "quality-excellent", "Excellent"
    elif coverage >= 0.85:
        q_class, q_label = "quality-good", "Good"
    elif coverage >= 0.75:
        q_class, q_label = "quality-fair", "Fair"
    else:
        q_class, q_label = "quality-poor", "Poor"

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Coverage</div>
                <div class="metric-value">{coverage:.4f}</div>
                <div class="metric-sub">{coverage * 100:.2f}% of actuals inside interval</div>
                <div style="margin-top:0.5rem">
                    <span class="quality-badge {q_class}">{q_label}</span>
                </div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Average Width</div>
                <div class="metric-value">${avg_width:,.2f}</div>
                <div class="metric-sub">Mean interval width in USD</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col3:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Winkler Score</div>
                <div class="metric-value">${winkler:,.2f}</div>
                <div class="metric-sub">Lower = better calibration</div>
            </div>""",
            unsafe_allow_html=True,
        )

    with col4:
        st.markdown(
            f"""<div class="metric-card">
                <div class="metric-label">Violations</div>
                <div class="metric-value">{violations}</div>
                <div class="metric-sub">out of {n} predictions</div>
            </div>""",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Main dashboard entry point
# ---------------------------------------------------------------------------


def run_dashboard() -> None:
    """
    Main function that renders the full Streamlit dashboard.

    Loads predictions from backtest_results.jsonl, displays error
    instructions if file is missing, otherwise renders charts and
    summary statistics.
    """
    # Page configuration — must be the very first Streamlit call
    st.set_page_config(
        page_title="Bitcoin Probabilistic Forecasting Dashboard",
        page_icon="\u20bf",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    # Inject custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Header
    st.markdown(
        """<div class="main-header">
            <h1>₿ Bitcoin Probabilistic Forecasting Dashboard</h1>
            <p>Walk-forward backtest · GBM + EWMA volatility · 95% prediction intervals</p>
        </div>""",
        unsafe_allow_html=True,
    )

    # --- Load data (Requirement 13.1) ---
    try:
        records = load_predictions(RESULTS_FILE)
        df = parse_to_dataframe(records)
    except FileNotFoundError:
        # Requirement 13.1: Display error message if file missing
        st.error(
            f"**Results file not found**: `{RESULTS_FILE}`\n\n"
            "Run the backtesting pipeline first:\n"
            "```bash\npython main.py\n```\n"
            "Then relaunch this dashboard."
        )
        return
    except ValueError as e:
        st.error(f"**Failed to parse results file**: {e}")
        return

    # --- Dataset info sidebar ---
    with st.sidebar:
        st.header("Dataset Info")
        st.metric("Total Predictions", len(df))
        st.metric("Date Range Start", df["timestamp"].min().strftime("%Y-%m-%d %H:%M"))
        st.metric("Date Range End", df["timestamp"].max().strftime("%Y-%m-%d %H:%M"))
        st.metric("Covered Predictions", int(df["covered"].sum()))
        st.metric("Violations", int((~df["covered"]).sum()))
        st.markdown("---")
        st.caption("Data source: Binance BTCUSDT 1h klines")

    # --- Summary metrics (Requirement 13.3) ---
    st.markdown(
        '<div class="section-title">📊 Summary Metrics</div>', unsafe_allow_html=True
    )
    render_metrics(df)

    # --- Time series chart (Requirement 13.2) ---
    st.markdown(
        '<div class="section-title">📈 Time Series Visualization</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """<div class="info-box">
            <b>Chart guide:</b> The shaded <span style="color:#63b3ed">blue regions</span>
            show 95% prediction intervals where the actual price fell <em>inside</em>.
            <span style="color:#fc8150">Orange regions</span> and ✕ markers indicate
            <em>violations</em> — where the actual price fell outside the predicted interval.
            Use the range slider below the chart to zoom into specific time windows.
        </div>""",
        unsafe_allow_html=True,
    )

    render_time_series_chart(df)

    # --- Raw data table ---
    with st.expander("🗂 View Raw Predictions Data"):
        display_df = df.copy()
        display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        display_df["covered"] = display_df["covered"].map({True: "✅", False: "❌"})
        display_df = display_df.rename(
            columns={
                "timestamp": "Timestamp",
                "actual_price": "Actual Price ($)",
                "lower_bound": "Lower Bound ($)",
                "upper_bound": "Upper Bound ($)",
                "covered": "Inside Interval",
            }
        )
        st.dataframe(
            display_df[
                [
                    "Timestamp",
                    "Actual Price ($)",
                    "Lower Bound ($)",
                    "Upper Bound ($)",
                    "Inside Interval",
                ]
            ],
            width="stretch",
            height=400,
        )

    # Footer
    st.markdown("---")
    st.caption(
        "Bitcoin Probabilistic Forecasting System · "
        "AlphaI × Polaris Challenge · "
        f"Results from `{RESULTS_FILE}`"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
# Streamlit executes this file as a script (not __main__), so we call
# run_dashboard() unconditionally.  We guard with a try/except so that
# plain Python imports (e.g., during pytest collection) do not fail.
try:
    import streamlit.runtime.scriptrunner as _sr

    if _sr.get_script_run_ctx() is not None:
        run_dashboard()
except Exception:
    pass

if __name__ == "__main__":
    run_dashboard()
