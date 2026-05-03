# Design Document: Bitcoin Probabilistic Forecasting System

## Overview

This document specifies the technical design for a Bitcoin probabilistic forecasting system that generates next-hour price predictions with 95% confidence intervals. The system implements a modified Geometric Brownian Motion (GBM) simulator enhanced with volatility clustering (EWMA) and fat-tailed shock distributions (Student-t), validated through rigorous walk-forward backtesting on 30 days of hourly Bitcoin price data.

### Design Goals

1. **Mathematical Rigor**: Implement advanced stochastic modeling with realistic market dynamics
2. **Temporal Integrity**: Ensure strict walk-forward validation without look-ahead bias
3. **Modularity**: Clean separation between data ingestion, mathematical engine, backtesting, and visualization
4. **Reproducibility**: Deterministic results with explicit dependency management
5. **Professional Quality**: Type hints, comprehensive documentation, robust error handling

### System Context

The system fetches historical Bitcoin price data from Binance, computes adaptive volatility estimates, generates 10,000 Monte Carlo price paths per prediction, and evaluates forecast quality through coverage metrics, average width, and Winkler scores. Results are persisted in JSON Lines format and visualized through an interactive Streamlit dashboard.

## Architecture

### High-Level System Architecture

```mermaid
graph TB
    subgraph "Data Layer"
        A[Binance API] -->|HTTPS| B[Data Ingestion Module]
        B -->|Validated DataFrame| C[Data Store]
    end
    
    subgraph "Mathematical Engine"
        C -->|Historical Prices| D[EWMA Calculator]
        D -->|Volatility Estimate| E[GBM Engine]
        E -->|10,000 Paths| F[Percentile Extractor]
        F -->|Prediction Interval| G[Results Aggregator]
    end
    
    subgraph "Validation Layer"
        G -->|Predictions| H[Backtesting Engine]
        H -->|Metrics| I[Evaluation Module]
        I -->|Coverage/Width/Winkler| J[Results Persistence]
    end
    
    subgraph "Presentation Layer"
        J -->|JSON Lines| K[Streamlit Dashboard]
        K -->|Interactive Viz| L[User]
    end
    
    style B fill:#e1f5ff
    style E fill:#ffe1e1
    style H fill:#e1ffe1
    style K fill:#fff5e1
