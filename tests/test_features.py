import numpy as np
import pandas as pd
import pytest

from src.features import (
    FEATURE_COLUMNS, build_features, build_full_frame, prepare_features,
)

PARTS = [1, 2, 3, 4]


@pytest.mark.parametrize('part', PARTS)
def test_features_do_not_look_ahead(part, train_df, test_df):
    """Changing prices after row k must not change any feature at or before row k."""
    full = build_full_frame(train_df, test_df)
    full['Volume'] = full['Volume'].astype(float)
    k = 400
    altered = full.copy()
    rng = np.random.default_rng(42)
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        altered.loc[k + 1:, col] = altered.loc[k + 1:, col] * rng.uniform(0.5, 1.5, len(altered) - k - 1)

    cols = FEATURE_COLUMNS[part]
    a = build_features(part, full.copy()).loc[:k, cols]
    b = build_features(part, altered).loc[:k, cols]
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize('part', PARTS)
def test_train_test_are_disjoint_and_ordered(part, train_df, test_df):
    X_tr, y_tr, X_te, y_te, tr, te = prepare_features(part, train_df, test_df)
    assert tr['Date'].max() < te['Date'].min()
    assert set(X_tr.index).isdisjoint(X_te.index)


@pytest.mark.parametrize('part', PARTS)
def test_no_training_target_falls_in_test_period(part, train_df, test_df):
    """A training row's target comes from the next trading day, which must precede the test period."""
    full = build_full_frame(train_df, test_df)
    dates = full['Date'].tolist()
    next_date = dict(zip(dates[:-1], dates[1:]))
    _, _, _, _, tr, te = prepare_features(part, train_df, test_df)
    split_date = te['Date'].min()
    assert all(next_date[d] < split_date for d in tr['Date'])


def test_parts_share_identical_rows(train_df, test_df):
    frames = [prepare_features(p, train_df, test_df) for p in PARTS]
    for X_tr, _, X_te, _, _, _ in frames[1:]:
        assert X_tr.index.equals(frames[0][0].index)
        assert X_te.index.equals(frames[0][2].index)


@pytest.mark.parametrize('part', [1, 2, 3])
def test_price_target_is_next_close(part, train_df, test_df):
    _, y_tr, _, y_te, tr, te = prepare_features(part, train_df, test_df)
    full = build_full_frame(train_df, test_df)
    next_close = full['Close'].shift(-1)
    pd.testing.assert_series_equal(y_te, next_close.loc[y_te.index], check_names=False)
    pd.testing.assert_series_equal(y_tr, next_close.loc[y_tr.index], check_names=False)


def test_return_target_is_next_day_return(train_df, test_df):
    _, y_tr, _, y_te, tr, te = prepare_features(4, train_df, test_df)
    full = build_full_frame(train_df, test_df)
    ret = full['Close'].shift(-1) / full['Close'] - 1
    pd.testing.assert_series_equal(y_te, ret.loc[y_te.index], check_names=False)


@pytest.mark.parametrize('part', PARTS)
def test_no_missing_values_after_split(part, train_df, test_df):
    X_tr, y_tr, X_te, y_te, _, _ = prepare_features(part, train_df, test_df)
    for obj in (X_tr, y_tr, X_te, y_te):
        assert not obj.isna().any().any() if hasattr(obj, 'columns') else not obj.isna().any()
