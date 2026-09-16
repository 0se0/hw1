# KOSPI Index Prediction

Next-day KOSPI closing price prediction using regression models.

## Results

| | Part 1 (OHLCV only) | Part 2 (+ Technical Indicators) | Part 3 (Tuned + Ensemble) |
|---|---|---|---|
| Best Model | Ridge Regression | Linear Regression | Lasso Regression (tuned) |
| R² | 0.8315 | 0.8971 | 0.8965 |
| RMSE | 33.41 | 24.23 | 24.30 |
| MAE | 26.51 | 18.22 | 18.39 |

**R² improved from 0.83 → 0.90 by adding technical indicators.**
Adding more features (EMA/MACD, day-of-week, return lags), hyperparameter
tuning (`RandomizedSearchCV`), and a `VotingRegressor` ensemble in Part 3
did **not** beat Part 2's plain Linear Regression — the top 4 models in
Part 2/3 all land within R² 0.895–0.897, suggesting the OHLCV + technical
indicator feature set is already close to the ceiling for a linear
next-day-close model on this data. Walk-forward (expanding-window)
validation on the Part 3 model confirms this is a stable result over
2023, not an artifact of the single train/test split (R² 0.897).

## Features

**Part 1** — OHLCV lag features (1, 2, 3, 5 days)

**Part 2** — Part 1 + MA5/20/60, RSI, Bollinger Bands, 
Momentum, Daily Return, Volatility

**Part 3** — Part 2 + EMA12/26, MACD, day-of-week dummies, return lags (1, 2, 3 days)

## Models Compared
Linear Regression, Ridge, Lasso, ElasticNet, 
Random Forest, Gradient Boosting, XGBoost, and a Voting ensemble of the
top 3 tuned models (Part 3)

## Validation
- TimeSeriesSplit cross-validation (prevents data leakage)
- `RandomizedSearchCV` hyperparameter tuning within each CV fold (Part 3)
- Walk-forward (expanding-window) validation over the 2023 test period (Part 3)
- Residual-based 95% prediction interval (empirical coverage: 96.2%)

## Tech
Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib
