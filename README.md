# KOSPI Index Prediction

Next-day KOSPI closing price prediction using regression models.

## Results

| | Naive Baseline (Closeₜ = Closeₜ₊₁) | Part 1 (OHLCV only) | Part 2 (+ Technical Indicators) | Part 3 (Tuned + Ensemble) |
|---|---|---|---|---|
| Best Model | — (persistence) | Ridge Regression | Linear Regression | Lasso Regression (tuned) |
| R² | 0.9002 | 0.8315 | 0.8971 | 0.8965 |
| RMSE | 23.86 | 33.41 | 24.23 | 24.30 |
| MAE | 17.64 | 26.51 | 18.22 | 18.39 |

**R² improved from 0.83 → 0.90 by adding technical indicators (Part 1 → Part 2).**
But **none of the 9 trained models beat a trivial naive baseline** that just
predicts "tomorrow's close = today's close" (RMSE 23.86 vs. best model's
24.23). Next-day closing *price level* is close to a random walk, so a
persistence baseline already explains most of the variance — a high R² here
is not by itself evidence of real predictive skill.

Adding more features (EMA/MACD, day-of-week, return lags), hyperparameter
tuning (`RandomizedSearchCV`), and a `VotingRegressor` ensemble in Part 3
did **not** beat Part 2's plain Linear Regression either — the top 4 models
in Part 2/3 all land within R² 0.895–0.897. Most of Part 3's additional
features are highly correlated with what Part 2 already has (EMA vs. MA,
MACD derived from EMA, return lags vs. Daily_Return), so they add little
new information; the real bottleneck is that no model here is beating the
persistence baseline. Walk-forward (expanding-window) validation on the
Part 3 model confirms its performance is stable over 2023 (R² 0.897), not
an artifact of the single train/test split — but stable-and-still-worse-than-naive
is the more accurate framing.

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
- Naive persistence baseline as a sanity check against all trained models (Part 3)

## Tech
Python, Scikit-learn, XGBoost, Pandas, NumPy, Matplotlib
