# KOSPI Index Prediction

Next-day KOSPI closing price prediction using regression models.

## Results

| | Part 1 (OHLCV only) | Part 2 (+ Technical Indicators) |
|---|---|---|
| Best Model | Ridge Regression | Linear Regression |
| R² | 0.8315 | 0.8971 |
| RMSE | 33.41 | 24.23 |
| MAE | 26.51 | 18.22 |

**R² improved from 0.83 → 0.90 by adding technical indicators.**

## Features

**Part 1** — OHLCV lag features (1, 2, 3, 5 days)

**Part 2** — Part 1 + MA5/20/60, RSI, Bollinger Bands, 
Momentum, Daily Return, Volatility

## Models Compared
Linear Regression, Ridge, Lasso, ElasticNet, 
Random Forest, Gradient Boosting, XGBoost

## Validation
TimeSeriesSplit cross-validation (prevents data leakage)

## Tech
Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib
