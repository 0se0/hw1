# KOSPI Index Prediction

Predicting the next trading day's KOSPI closing price with regression models,
built incrementally across three stages — from a bare OHLCV feature set to
a tuned, ensembled model with proper time-series validation.

## Overview

- **Task**: predict tomorrow's KOSPI closing price from historical price/volume data
- **Data**: daily KOSPI OHLCV, 2019–2022 for training (926 usable rows), 2023 held out for testing (243 rows)
- **Approach**: three stages of increasing model complexity. Models are selected by
  `TimeSeriesSplit` cross-validation on the training data only; the 2023 test set is
  used for the final report, never for choosing a model.
- **Comparable by construction**: every part and the baseline are evaluated on the
  exact same train/test rows, and rolling indicators are computed over the full
  history using past values only (no look-ahead).

## Key Insight

None of the models selected by cross-validation beats a trivial baseline that
predicts "tomorrow's close = today's close" (test RMSE 24.09 vs. 24.35–24.55).
Daily closing price is close to a random walk, so a persistence forecast already
explains ~92% of the variance — a high R² on price *level* is not, by itself,
evidence that a model has learned anything useful. This project reports the
baseline next to every model as a required sanity check.

Two things the baseline made visible:

- **Information parity matters.** An earlier version of this pipeline left out the
  current day's OHLCV (predicting `t+1` from `t-1` and earlier), which made the
  models look much worse than the baseline (R² 0.83) and made technical indicators
  look like a big win. Once the same information the baseline uses (day `t`) is
  included, plain linear models land on top of the baseline.
- **Feature engineering barely moves the needle.** Adding technical indicators
  (Part 2) changes test R² by about +0.001, and the extra features, tuning, and
  ensembling in Part 3 do not improve on it.

![Model comparison against the naive baseline](figures/part3_model_comparison.png)

## Results

Test set (2023, 243 rows). Each part's model is the one with the lowest
`TimeSeriesSplit` CV MSE; all three happen to be Lasso.

| | Naive Baseline (Closeₜ₊₁ = Closeₜ) | Part 1 (OHLCV) | Part 2 (+ Technical Indicators) | Part 3 (+ Extra Features, Tuned) |
|---|---|---|---|---|
| Selected Model | — (persistence) | Lasso | Lasso | Lasso (tuned) |
| R² | 0.9235 | 0.9209 | 0.9219 | 0.9206 |
| RMSE | 24.09 | 24.49 | 24.35 | 24.55 |
| MAE | 18.20 | 18.60 | 18.46 | 18.44 |

- All CV-selected models trail the baseline by ~1–2% RMSE.
- A few *non-selected* linear models edge past it on the test set (Part 2 Ridge
  RMSE 23.94, Part 2 Linear 24.05), but by well under 1% — noise-level differences,
  and picking them would mean selecting on the test set.
- Part 3's `VotingRegressor` ensemble (R² 0.9209) does not beat its best member.
  Most of Part 3's new features are nearly redundant with Part 2's (e.g. EMA12 vs.
  MA5 correlation 0.998, EMA26 vs. MA20 0.999; MACD is derived from the EMAs).
- Walk-forward (expanding-window) validation of the Part 3 model gives R² 0.9217 /
  RMSE 24.37 — consistent with the single split, and still a hair behind the baseline.
- Tree models (Random Forest, Gradient Boosting, XGBoost) do notably worse than
  linear models here (likely because they cannot extrapolate beyond the price range
  seen in training; not tested separately).

![Part 3 predictions with a 95% residual-based interval](figures/part3_prediction_interval.png)

## Methodology

**Part 1** — OHLCV for the current day and lags of 1, 2, 3, 5 days.

**Part 2** — Part 1 + technical indicators: MA5/20/60, RSI, Bollinger Bands,
Momentum, Daily Return, Volatility, consecutive up/down-day counts.

**Part 3** — Part 2 + EMA12/26, MACD and signal line, day-of-week dummies, return
lags (1, 2, 3 days), `RandomizedSearchCV` tuning, and a `VotingRegressor` ensemble
of the top 3 tuned models.

**Models compared**: Linear Regression, Ridge, Lasso, Elastic Net, Random Forest,
Gradient Boosting, XGBoost, plus a Voting ensemble in Part 3.

**Validation**:
- Model selection by `TimeSeriesSplit` cross-validation MSE (training data only)
- `RandomizedSearchCV` hyperparameter tuning with `TimeSeriesSplit` as the CV scheme (Part 3)
- Identical train/test rows for every part and the baseline
- Naive persistence baseline as a sanity check against every model
- Walk-forward (expanding-window) validation over the 2023 test period (Part 3)
- Approximate 95% prediction interval sized from out-of-sample CV residuals on the
  training data (half-width ±73.7). It covers 99.6% of the test points, i.e. it is
  conservative: early CV folds train on little data and inflate the residual spread.

![Residual diagnostics: distribution and Q-Q plot](figures/part3_residual_diagnostics.png)

## Project Structure

```
machine_learning_hw1.ipynb   # full pipeline: data loading -> Part 1 -> Part 2 -> Part 3
README.md
data/
  kospi_train.csv             # not tracked in git, see below
  kospi_test.csv               # not tracked in git, see below
models/
  kospi_part{1,2,3}_model.pkl  # CV-selected model per part
  kospi_part{1,2,3}_scaler.pkl # matching StandardScaler per part
figures/
  *.png                        # correlation, predictions, residuals, comparisons
```

## Running It

1. Place `kospi_train.csv` and `kospi_test.csv` (daily `Date, Open, Low,
   High, Close, Volume`) under `data/` — these aren't committed to git.
2. Install dependencies: `pandas`, `numpy`, `matplotlib`, `seaborn`,
   `scikit-learn`, `xgboost`, `scipy`, `joblib`.
3. Run `machine_learning_hw1.ipynb` top to bottom. Models/scalers are saved
   to `models/` and plots to `figures/`.

## Tech

Python, scikit-learn, XGBoost, pandas, NumPy, Matplotlib, seaborn
