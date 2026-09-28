import numpy as np
import pandas as pd
import pytest


def _make_frame(n, start, seed):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    close = 2000 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.003, n)))
    volume = rng.integers(200_000, 500_000, n)
    return pd.DataFrame({'Date': dates, 'Open': open_, 'Low': low, 'High': high,
                         'Close': close, 'Volume': volume})


@pytest.fixture
def train_df():
    return _make_frame(500, '2019-01-01', seed=0)


@pytest.fixture
def test_df(train_df):
    return _make_frame(150, train_df['Date'].iloc[-1] + pd.offsets.BDay(1), seed=1)


def write_csvs(directory, train_df, test_df):
    directory.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(directory / 'kospi_train.csv', index=False)
    test_df.to_csv(directory / 'kospi_test.csv', index=False)
    return directory
