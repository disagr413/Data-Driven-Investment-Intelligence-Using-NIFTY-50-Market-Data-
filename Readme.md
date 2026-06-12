# Data-Driven Investment Intelligence Using NIFTY 50 Market Data

**Cult Quant 2026** is an institutional-grade, end-to-end algorithmic investment platform built for the NIFTY 50 equity universe. It bridges the gap between raw market data and actionable portfolio decisions by combining advanced Machine Learning with Modern Portfolio Theory, a framework often called "Quantamental" investing.

---

## Key Results at a Glance

| Metric | Value |
|---|---|
| Top Pick | **NESTLEIND** with an expected return of **15.10%** and an AI score of **91.5 out of 100** |
| Lowest Risk Stock | **INFY** with a forecasted volatility of **0.18%** |
| Sector Leader | **Consumer Goods** |
| Balanced Portfolio Sharpe Ratio | **0.98** |
| Conservative Portfolio under Severe Crash (-30%) | Drawdown limited to **-21.99%** |
| ML Anomalies Detected | **3** cross-sectional anomalies in the current market |
| Total Stocks Analyzed | **50** (full NIFTY 50 universe) |

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture and Pipeline](#architecture-and-pipeline)
3. [Repository Structure](#repository-structure)
4. [Research Pipeline (Notebooks)](#research-pipeline-notebooks)
5. [ML Models and Features](#ml-models-and-features)
6. [Portfolio Optimization](#portfolio-optimization)
7. [Risk Analytics and Stress Testing](#risk-analytics-and-stress-testing)
8. [Anomaly Detection](#anomaly-detection)
9. [Investment Recommendations Output](#investment-recommendations-output)
10. [Sector Analysis](#sector-analysis)
11. [Streamlit Dashboard](#streamlit-dashboard)
12. [Output Files](#output-files)
13. [Environment Setup and Installation](#environment-setup-and-installation)
14. [Running the Application](#running-the-application)
15. [Dev Container and GitHub Codespaces](#dev-container-and-github-codespaces)
16. [Contributors](#contributors)

---

## Project Overview

Traditional heuristic trading models and linear factor models often fail to capture the non-linear, heteroskedastic nature of modern financial markets. This project tackles that problem head-on through three core capabilities.

**Predicting Alpha.** Histogram-based gradient boosting ensembles (LightGBM and XGBoost) are used to forecast next-day expected returns, trained on a rich set of engineered momentum, trend, and volatility features.

**Managing Risk.** Monte Carlo simulations across 10,000 portfolio combinations are used to construct an Empirical Efficient Frontier and isolate three mathematically optimal portfolio types (Conservative, Balanced, and Aggressive).

**Ensuring Transparency.** Explainable AI overlays are applied throughout, including SHAP value decomposition for feature importance, Isolation Forest for cross-sectional anomaly detection, and Natural Language Generation (NLG) to produce human-readable trade rationales for every stock.

All results are surfaced through an interactive 8-page Streamlit dashboard.

### The Dataset

- **Source:** [Kaggle NIFTY50 Stock Market Data](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data) by Rohan Rao and [Index Datasets](https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020)
- **Universe:** All 50 NIFTY 50 constituent stocks
- **Training Period:** Historical data through end of 2017
- **Test Period:** January 2018 onwards
- **Features Engineered:** 50+ technical indicators plus cross-sectional and lag-based features

---

## Architecture and Pipeline

```
Raw NIFTY 50 CSV Data (Kaggle)
            |
            v
+----------------------------------+
|  Notebook 1: Data Cleaning       |  Missing values, corporate action
|  notebook1_data_cleaning         |  adjustment, schema unification
+----------------------------------+
            |
            v
+----------------------------------+
|  Notebook 2: EDA + Feature Engg  |  RSI, MACD, ATR, BB, OBV, Stoch,
|  notebook2_eda_feature_engg      |  volatility clustering, SHAP,
+----------------------------------+  lag and cross-sectional features
            |
            v
+----------------------------------+
|  Notebook 3: Model Training      |  LightGBM + XGBoost + RandomForest
|  notebook3_model_training        |  Optuna tuning, TimeSeriesSplit CV,
+----------------------------------+  SHAP explainability
            |
            v
+----------------------------------+
|  Notebook 4: Portfolio Optim.    |  Monte Carlo (10,000 simulations)
|  notebook4_portfolio_optimiz.    |  Efficient Frontier, 3 optimal
+----------------------------------+  portfolio types, scoring engine
            |
            v
+----------------------------------+
|  Notebook 5: Advanced Intel.     |  Isolation Forest anomaly detection
|  notebook5_advanced_intelligence |  Stress testing, VaR, EWMA vol,
+----------------------------------+  NLG trade rationales
            |
            v
+----------------------------------+
|  app.py (Streamlit Dashboard)    |  8-page interactive dashboard
|  Runs on port 8501               |  Plotly visualizations
+----------------------------------+
```

---

## Repository Structure

```
Data-Driven-Investment-Intelligence-Using-NIFTY-50-Market-Data-/
|
+-- .devcontainer/
|   +-- devcontainer.json               # GitHub Codespaces configuration
|
+-- app.py                              # Main Streamlit dashboard application
+-- requirements.txt                    # Project dependencies
+-- README.md                           # Project documentation
|
+-- notebook1_data_cleaning.ipynb       # Stage 1: Data ingestion and cleaning
+-- notebook2_eda_feature_engg.ipynb    # Stage 2: EDA and feature engineering
+-- notebook3_model_training.ipynb      # Stage 3: ML model training and evaluation
+-- notebook4_portfolio_optimization.ipynb  # Stage 4: Portfolio construction
+-- notebook5_advanced_intelligence.ipynb   # Stage 5: Risk and anomaly intelligence
|
+-- final_opportunities.csv             # Master output: all 50 stocks ranked
+-- investment_recommendations.csv      # AI-scored recommendations with explanations
+-- executive_summary.csv               # Top conviction picks summary
+-- portfolio_allocations.csv           # Weights for all 3 portfolio types
+-- portfolio_risk_summary.csv          # Risk-return profile per portfolio
+-- efficient_frontier.csv              # 10,000 Monte Carlo simulated portfolios
+-- anomalies.csv                       # Historical extreme return events
+-- stress_test_results.csv             # Drawdown simulation results
+-- sector_analytics.csv                # Sector-level aggregated metrics
+-- predictions.csv                     # Full test-period ML predictions (2018+)
+-- report_insights.txt                 # Plain-text executive summary
```

---

## Research Pipeline (Notebooks)

### Notebook 1 - Data Cleaning (`notebook1_data_cleaning.ipynb`)

This notebook handles raw NIFTY 50 data ingestion from Kaggle using `kagglehub`. It loads individual stock CSVs alongside the combined `NIFTY50_all.csv`, unifies the schema across all 50 symbols, treats missing values with forward-fill strategies, normalizes dates to `YYYY-MM-DD` format, and applies corporate action adjustments. The cleaned outputs are `master_features.csv`, `pivot_close.csv`, and `pivot_returns.csv`.

### Notebook 2 - EDA and Feature Engineering (`notebook2_eda_feature_engg.ipynb`)

This notebook constructs the full feature matrix from cleaned OHLCV data. Features are organized into four groups.

**Technical Indicators** cover RSI, MACD, MACD Histogram, OBV, Stochastic-K, ATR, Bollinger Band percentage, Volume Ratio, ROC-10, ROC-20, and Annualized Volatility.

**Lag Features** apply each of the 13 base indicators at 1-day, 2-day, and 3-day lookbacks with strict no-leakage enforcement (look-back only, never forward).

**Slope Features** capture the rate of change in MACD Histogram, RSI, OBV, and Volume ratio across consecutive lags.

**Cross-Sectional Features** include market-relative return, sector-relative return, market-relative 5-day return, return rank, RSI sector relative, and volume rank. SHAP value analysis is also performed here for early feature importance insights, alongside volatility clustering using EWMA.

### Notebook 3 - Model Training (`notebook3_model_training.ipynb`)

An ensemble of five models is trained with a hard time-series cutoff: all data before January 2018 is used for training and everything after is held out for testing. Cross-validation uses 5-fold TimeSeriesSplit to prevent any future data leakage.

**Regression models** include LightGBM (600 estimators, learning rate 0.04, max depth 7, num leaves 80), XGBoost (Optuna-tuned), and RandomForest (Optuna-tuned). These predict the next-day closing price.

**Classification models** include LightGBM Classifier and XGBoost Classifier, which predict the directional movement (up or down) with AUC as the primary optimization target.

Evaluation uses RMSE in INR, MAPE percentage, R-squared, Direction Accuracy, AUC, and F1-Score. The final ensemble prediction is a weighted blend of all three regression models. SHAP decomposition is applied per stock and per feature to explain individual predictions.

### Notebook 4 - Portfolio Optimization (`notebook4_portfolio_optimization.ipynb`)

This notebook takes the latest model predictions and builds investable portfolios. An investment scoring engine normalizes six metrics (Expected Return, Sharpe, ROC-20, RSI, Volatility, and Max Drawdown) using MinMaxScaler and combines them into a composite Future Score. A Monte Carlo simulation across 10,000 random portfolio weight combinations is then run to map the full Efficient Frontier. Three optimal portfolios are extracted: the Conservative portfolio (minimum volatility point), the Balanced portfolio (maximum Sharpe ratio), and the Aggressive portfolio (maximum expected return). NLG-based rationales are generated per stock explaining the recommendation in plain English.

### Notebook 5 - Advanced Intelligence (`notebook5_advanced_intelligence.ipynb`)

Adds institutional-grade risk overlays on top of the portfolio outputs. Anomaly detection runs across three methods: Z-score flagging for extreme daily return events (beyond 3 standard deviations), drawdown anomaly detection for any stock breaching -20% from its rolling peak, and Isolation Forest for cross-sectional anomaly detection on the current snapshot of investment metrics. EWMA-based forward volatility forecasting is continuously maintained across all 50 stocks. Stress testing simulates portfolio drawdown under three macro scenarios using portfolio beta. Value at Risk is computed using both historical and parametric methods. All signals are merged into the final `final_opportunities.csv` output.

---

## ML Models and Features

### Model Summary

| Model | Task | Key Parameters |
|---|---|---|
| LightGBM Regressor | Next-day price forecast | 600 estimators, lr 0.04, max depth 7, num leaves 80 |
| XGBoost Regressor | Next-day price validation | Optuna-tuned |
| RandomForest Regressor | Ensemble baseline | Optuna-tuned |
| LightGBM Classifier | Directional movement (up/down) | AUC-optimized |
| XGBoost Classifier | Directional validation | AUC-optimized |

### Feature Categories (50+ total)

| Category | Features |
|---|---|
| Momentum | RSI, ROC-10, ROC-20, Stochastic-K |
| Trend | MACD, MACD Histogram, MACD slope |
| Volatility | ATR, Bollinger Band %, Annualized Vol, Vol Ratio |
| Volume | OBV, OBV slope, Volume Rank |
| Lag Features | All above indicators at t-1, t-2, and t-3 |
| Cross-Sectional | Market Relative Return, Sector Relative Return, Return Rank, RSI Sector Relative |

### Train/Test Split

- **Training set:** Historical data up to December 31, 2017
- **Test set:** January 1, 2018 onwards
- **Validation:** 5-fold TimeSeriesSplit with no future leakage

---

## Portfolio Optimization

Three optimal portfolios are derived from the Efficient Frontier produced by 10,000 Monte Carlo simulations:

| Portfolio | Expected Return | Volatility | Sharpe Ratio |
|---|---|---|---|
| Conservative (Min Vol) | 18.70% | 15.87% | 0.8635 |
| Balanced (Max Sharpe) | 21.03% | 16.34% | 0.9806 |
| Aggressive (Max Return) | 22.95% | 18.55% | 0.9680 |

### Sample Allocation for NESTLEIND (Top Ranked Stock)

| Portfolio | Weight |
|---|---|
| Conservative | 17.47% |
| Balanced | 17.34% |
| Aggressive | 13.86% |

---

## Risk Analytics and Stress Testing

Portfolio drawdown is simulated across three market crash scenarios using each portfolio's beta:

| Scenario | Conservative | Balanced | Aggressive |
|---|---|---|---|
| Moderate Correction (-10%) | -7.33% | -7.33% | -8.29% |
| Bear Market (-20%) | -14.66% | -14.66% | -16.58% |
| Severe Crash (-30%) | -21.99% | -21.99% | -24.87% |

Portfolio betas are 0.73 for Conservative and Balanced, and 0.83 for Aggressive. EWMA volatility forecasting runs continuously across all 50 stocks, and individual maximum drawdown is tracked for every stock in `investment_recommendations.csv`.

---

## Anomaly Detection

Three complementary methods are used to detect market anomalies.

**Extreme Return Events** flag any daily return with an absolute Z-score above 3 standard deviations from the historical mean, applied across the full NIFTY 50 time series.

**Drawdown Anomalies** flag any stock experiencing a cumulative drawdown below -20% from its rolling peak price.

**ML Cross-Sectional Anomalies (Isolation Forest)** detect stocks with unusual combinations of Expected Return, Sharpe, Volatility, and Max Drawdown in the current market snapshot. The model uses a contamination threshold of 5% and currently identifies **3 anomalies** in the NIFTY 50 universe.

All anomaly flags are stored in `final_opportunities.csv` under the `Anomaly_Flag` column, labeled as either `Normal` or `Anomaly`.

---

## Investment Recommendations Output

The master output file `final_opportunities.csv` covers all 50 NIFTY 50 stocks with 15 columns per row:

| Column | Description |
|---|---|
| `Rank` | AI-ranked position (1 is highest conviction) |
| `Symbol` | NSE stock ticker |
| `Expected_Return` | ML-predicted return in percentage |
| `Sharpe` | Historical Sharpe ratio |
| `Volatility` | Annualized historical volatility in percentage |
| `Sortino` | Sortino ratio measuring downside-only risk |
| `Max_Drawdown` | Maximum historical drawdown in percentage |
| `Recommendation` | Rule-based signal: BUY, HOLD, REDUCE, or SELL |
| `Confidence_Score` | AI composite score from 0 to 100 |
| `Explanation` | NLG-generated key factors driving the rating |
| `Anomaly_Flag` | Isolation Forest result: Normal or Anomaly |
| `Forecast_Volatility` | EWMA-forecasted forward volatility |
| `Future_Score` | Final composite ranking score from 0 to 100 |
| `Final_Recommendation` | Strong Buy, Buy, Hold, Reduce, or Sell |
| `AI_Explainability` | Full human-readable NLG trade rationale |

### Top Stocks by AI Score

| Rank | Symbol | Expected Return | Sharpe | Final Recommendation | AI Score |
|---|---|---|---|---|---|
| 1 | NESTLEIND | 15.10% | 0.57 | Strong Buy | 91.50 |
| 2 | SHREECEM | 2.75% | 0.83 | Buy | 63.17 |
| 3 | MARUTI | 1.54% | 0.60 | Buy | 60.58 |
| 4 | EICHERMOT | 5.99% | 0.38 | Buy | 60.20 |

---

## Sector Analysis

Aggregated metrics across 12 sectors drawn from `sector_analytics.csv`:

| Rank | Sector | Expected Return | Sharpe | Stocks |
|---|---|---|---|---|
| 1 | Consumer Goods | 3.17% | 0.32 | 6 |
| 2 | Energy | 1.70% | 0.05 | 8 |
| 3 | Media | 1.69% | -0.01 | 1 |
| 4 | Chemicals | 1.56% | 0.19 | 1 |
| 5 | Automobile | 1.44% | 0.37 | 6 |
| 6 | IT | 0.96% | 0.10 | 5 |
| 7 | Telecom | 0.42% | 0.39 | 1 |
| 8 | Pharma | 0.37% | 0.12 | 3 |
| 9 | Materials | 0.15% | 0.39 | 7 |
| 10 | Banking | -0.15% | 0.39 | 6 |
| 11 | Infrastructure | -0.66% | 0.17 | 2 |
| 12 | Financial Services | -2.43% | 0.57 | 3 |

---

## Streamlit Dashboard

`app.py` is an 8-page interactive dashboard built with Streamlit and Plotly. It reads all output CSVs directly and renders live visualizations. No re-running of notebooks is needed since all CSV outputs are included in the repository.

### Pages

| Page | What it shows |
|---|---|
| 1. Overview | Market-wide summary metrics and a top opportunities bar chart ranking stocks by Future Score and Expected Return |
| 2. Stock Explorer | Per-stock deep-dive with an AI gauge meter, key metrics (return, volatility, Sharpe), and the full NLG rationale |
| 3. Portfolio Builder | Risk-return table for all three portfolios and interactive donut charts for Conservative, Balanced, and Aggressive allocations |
| 4. Risk Dashboard | Efficient Frontier scatter plot colored by Sharpe ratio, stress test results table, and max drawdown bar chart |
| 5. Forecast Dashboard | Risk-return scatter plot colored by recommendation category and a full sector outlook table |
| 6. Anomaly Dashboard | Cross-sectional anomaly table, historical extreme events log, and a monthly anomaly frequency timeline |
| 7. Investment Recommendations | Full filterable stock table covering all 50 stocks and all 15 metrics |
| 8. Executive Summary | Top conviction picks with AI scores and one-line NLG rationales |

### Tech Stack

| Layer | Library |
|---|---|
| Dashboard | streamlit |
| Visualization | plotly.express and plotly.graph_objects |
| Data handling | pandas |
| ML models (notebooks) | lightgbm, xgboost, scikit-learn |
| Optimization | scipy.optimize, optuna |
| Explainability | shap |
| Anomaly detection | sklearn.ensemble.IsolationForest |

---

## Output Files

| File | Description | Approx. Rows |
|---|---|---|
| `final_opportunities.csv` | Master ranked output for all 50 NIFTY stocks | 50 |
| `investment_recommendations.csv` | AI-scored ranking with NLG explanations | 50 |
| `executive_summary.csv` | Top conviction picks with Symbol, Score, Return, and Volatility | 10 |
| `portfolio_allocations.csv` | Conservative, Balanced, and Aggressive weights per stock | 50 |
| `portfolio_risk_summary.csv` | Return, volatility, and Sharpe per portfolio type | 3 |
| `efficient_frontier.csv` | Monte Carlo portfolio universe (Volatility, Return, Sharpe per point) | 10,000 |
| `anomalies.csv` | Historical extreme return events by Date and Symbol | 100,000+ |
| `stress_test_results.csv` | Drawdown under 3 crash scenarios for each portfolio | 3 |
| `sector_analytics.csv` | Sector-aggregated return, volatility, Sharpe, and rank | 12 |
| `predictions.csv` | Full test-set ML predictions from 2018 onwards | 100,000+ |
| `report_insights.txt` | Plain-text executive summary of key findings | n/a |

---

## Environment Setup and Installation

### Prerequisites

- Python 3.11 or higher
- pip

### Step 1: Clone the Repository

```bash
git clone https://github.com/disagr413/Data-Driven-Investment-Intelligence-Using-NIFTY-50-Market-Data-.git
cd Data-Driven-Investment-Intelligence-Using-NIFTY-50-Market-Data-
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

The `requirements.txt` file covers the dashboard dependencies:

```
streamlit
pandas
plotly
```

To reproduce the full research pipeline by re-running the notebooks, install the additional ML dependencies:

```bash
pip install lightgbm xgboost scikit-learn shap optuna scipy seaborn matplotlib joblib kagglehub
```

---

## Running the Application

```bash
streamlit run app.py
```

The dashboard will open at **https://5tfnxejxrhpepbm84gmtcs.streamlit.app/**

All CSV output files are already included in the repository, so the dashboard runs immediately without needing to re-execute the notebooks.

---

## Dev Container and GitHub Codespaces

This repository is fully configured for GitHub Codespaces with zero local setup required. The `.devcontainer/devcontainer.json` file handles everything automatically:

- **Base image:** `mcr.microsoft.com/devcontainers/python:1-3.11-bookworm`
- All packages from `requirements.txt` are installed during container build
- The Streamlit app launches automatically when you attach to the container
- Port 8501 is forwarded and opened in the browser preview
- VS Code extensions for Python and Pylance are pre-installed
- `README.md` and `app.py` open automatically on start

### Launching in Codespaces

1. Click the green **Code** button on GitHub
2. Select **Codespaces** and then **Create codespace on main**
3. The Streamlit dashboard launches automatically in the built-in browser preview

---

## Contributors

| Name | GitHub |
|---|---|
| Disha Agrawal 24113039| [@disagr413](https://github.com/disagr413) |
| Ayush Aary 24113028| [@ayushaary](https://github.com/ayushaary) |

---
