# AlphaAI - Bitcoin Probabilistic Forecasting

A production-ready Bitcoin price prediction system using probabilistic forecasting with EWMA volatility estimation and Student-t distributed Monte Carlo simulations.

## 🎯 Features

### Part A: Backtesting Engine
- **Walk-forward validation** with strict no-look-ahead bias
- **EWMA volatility** estimation (24-hour lookback)
- **Student-t distributed shocks** (5 degrees of freedom)
- **10,000 Monte Carlo simulations** per prediction
- **95% confidence intervals**
- Comprehensive metrics: Coverage Rate, Average Width, Winkler Score

### Part B: Live Dashboard
- **Real-time Bitcoin price predictions** from Binance API
- **Interactive Plotly charts** with prediction intervals
- **Streamlit-based UI** for instant deployment
- **Auto-refresh** with configurable cache TTL
- Live metrics display with model parameters

### Part C: Cloud Persistence
- **Supabase (PostgreSQL)** integration for prediction storage
- **Automatic reconciliation** of predictions with actual prices
- **Historical timeline** with Winkler score tracking
- **Demo mode** when cloud database is not configured

## 📊 Performance Metrics

Based on 719 historical predictions:
- **Coverage Rate**: 92.63%
- **Average Interval Width**: $1,030.48
- **Mean Winkler Score**: 1,714.22

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AlphaAi.git
cd AlphaAi
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the backtest (optional):
```bash
python backtest.py
```

4. Launch the dashboard:
```bash
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`

## ☁️ Cloud Setup (Optional)

To enable Part C cloud persistence:

1. Create a free [Supabase](https://supabase.com) account
2. Create a new project and get your credentials
3. Create `.streamlit/secrets.toml`:
```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_KEY = "your-anon-key-here"
```

4. Run this SQL in Supabase SQL Editor:
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
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ
);

CREATE INDEX idx_predictions_timestamp ON predictions(timestamp);
CREATE INDEX idx_predictions_actual_price ON predictions(actual_price);
```

5. Restart Streamlit - cloud features will activate automatically!

## 📁 Project Structure

```
AlphaAi/
├── app.py                      # Main Streamlit dashboard
├── engine.py                   # Core mathematical functions
├── data_fetcher.py            # Binance API integration
├── backtest.py                # Walk-forward backtesting script
├── supabase_client.py         # Cloud persistence layer
├── backtest_results.jsonl     # Historical predictions (719 records)
├── backtest_summary.json      # Performance metrics
├── requirements.txt           # Python dependencies
├── .streamlit/
│   ├── config.toml           # Streamlit configuration
│   └── secrets.toml.example  # Template for secrets
└── README.md
```

## 🔬 Technical Details

### Volatility Estimation
- Uses Exponentially Weighted Moving Average (EWMA)
- 24-hour lookback window
- Captures recent market dynamics

### Monte Carlo Simulation
- Student-t distribution with 5 degrees of freedom
- Accounts for fat tails in crypto returns
- 10,000 simulation paths per prediction

### Validation Methodology
- Strict walk-forward validation
- At step i, only uses data from rows 0 to i-1
- No look-ahead bias
- Realistic out-of-sample performance

### Evaluation Metrics
- **Coverage Rate**: Fraction of actuals within predicted intervals
- **Average Width**: Mean width of prediction intervals
- **Winkler Score**: Penalizes both width and coverage misses

## 🛠️ API Integration

Uses Binance public API:
- Endpoint: `https://api.binance.com/api/v3/klines`
- Symbol: BTCUSDT
- Interval: 1 hour
- No authentication required

## 📝 License

MIT License - feel free to use this for your own projects!

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

**Built with ❤️ for quantitative finance and probabilistic forecasting**
