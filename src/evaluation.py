import numpy as np
import scipy.stats as st
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"{model_name} 성능:")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R^2: {r2:.4f}")

    return {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}


def time_series_cv(model, X, y, n_splits=5):
    """Mean MSE over TimeSeriesSplit folds."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
    return -cv_scores.mean()


def cv_mse_constant(y, fit_const, n_splits=5):
    """CV MSE of a constant forecast (zero, train mean, ...) under the same folds."""
    errs = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits).split(y):
        errs.append(np.mean((y[va_idx] - fit_const(y[tr_idx])) ** 2))
    return np.mean(errs)


def return_metrics(y_true, y_pred, close_t, close_next):
    """RMSE, out-of-sample R^2 against a zero forecast, direction accuracy, rank IC, price RMSE."""
    sse = np.sum((y_true - y_pred) ** 2)
    return {
        'rmse': np.sqrt(np.mean((y_true - y_pred) ** 2)),
        'r2_os': 1 - sse / np.sum(y_true ** 2),
        'dir_acc': np.mean(np.sign(y_pred) == np.sign(y_true)),
        'ic': st.spearmanr(y_true, y_pred).correlation,
        'price_rmse': np.sqrt(np.mean((close_t * (1 + y_pred) - close_next) ** 2)),
    }


def walk_forward_validate(model, X_train, y_train, X_test, y_test, step=21):
    """Expanding-window walk-forward: predict `step` rows, add the realized rows to history, refit."""
    X_hist = np.asarray(X_train)
    y_hist = np.asarray(y_train)
    X_test = np.asarray(X_test)
    y_test_arr = np.asarray(y_test)

    n_test = len(X_test)
    preds = np.zeros(n_test)

    for start in range(0, n_test, step):
        end = min(start + step, n_test)
        step_model = clone(model)
        step_model.fit(X_hist, y_hist)
        preds[start:end] = step_model.predict(X_test[start:end])

        X_hist = np.vstack([X_hist, X_test[start:end]])
        y_hist = np.concatenate([y_hist, y_test_arr[start:end]])

    return preds


def cv_residuals(model, X, y, n_splits=5):
    """Out-of-sample residuals from TimeSeriesSplit folds (used to size prediction intervals)."""
    X = np.asarray(X)
    y = np.asarray(y)
    res = []
    for tr_idx, va_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        fold_model = clone(model).fit(X[tr_idx], y[tr_idx])
        res.append(y[va_idx] - fold_model.predict(X[va_idx]))
    return np.concatenate(res)
