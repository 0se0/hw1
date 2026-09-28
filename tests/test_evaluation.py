import numpy as np
from sklearn.linear_model import Ridge

from src.evaluation import (
    cv_mse_constant, return_metrics, scale_features, walk_forward_validate,
)


def test_scaler_is_fit_on_train_only():
    rng = np.random.default_rng(0)
    X_tr = rng.normal(5, 2, (100, 3))
    X_te = rng.normal(50, 2, (30, 3))
    _, X_te_s, scaler = scale_features(X_tr, X_te)
    np.testing.assert_allclose(scaler.mean_, X_tr.mean(axis=0))
    assert X_te_s.mean() > 10   # far from train distribution, so it is not re-centred on test


def test_return_metrics_perfect_prediction():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    close_t = np.array([100.0, 101.0, 99.0, 102.0])
    close_next = close_t * (1 + y)
    m = return_metrics(y, y, close_t, close_next)
    assert m['rmse'] == 0 and m['r2_os'] == 1 and m['dir_acc'] == 1 and m['price_rmse'] < 1e-9


def test_zero_forecast_has_zero_oos_r2():
    y = np.array([0.01, -0.02, 0.03, -0.01])
    close = np.full(4, 100.0)
    assert return_metrics(y, np.zeros(4), close, close * (1 + y))['r2_os'] == 0


def test_cv_mse_constant_zero_forecast_is_mean_squared_target():
    y = np.random.default_rng(1).normal(0, 1, 120)
    manual = []
    from sklearn.model_selection import TimeSeriesSplit
    for _, va in TimeSeriesSplit(n_splits=5).split(y):
        manual.append(np.mean(y[va] ** 2))
    assert np.isclose(cv_mse_constant(y, lambda v: 0.0), np.mean(manual))


def test_walk_forward_uses_only_past_targets():
    """Predictions for a block must not depend on targets from that block or later."""
    rng = np.random.default_rng(2)
    X_tr, y_tr = rng.normal(size=(80, 4)), rng.normal(size=80)
    X_te, y_te = rng.normal(size=(60, 4)), rng.normal(size=60)
    base = walk_forward_validate(Ridge(), X_tr, y_tr, X_te, y_te, step=20)

    y_changed = y_te.copy()
    y_changed[20:] += 100          # alter targets from block 2 onward
    changed = walk_forward_validate(Ridge(), X_tr, y_tr, X_te, y_changed, step=20)

    np.testing.assert_allclose(base[:20], changed[:20])   # block 1 unaffected
    np.testing.assert_allclose(base[20:40], changed[20:40])   # block 2 only sees block-1 targets
    assert not np.allclose(base[40:], changed[40:])       # block 3 sees changed block-2 targets
