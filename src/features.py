"""Feature construction and train/test splitting shared by every part.

All features at row t use only data up to t. The target is built from t+1 and the last
training row is dropped, so no training target lies in the test period.
"""
import numpy as np
import pandas as pd

LAG_COLUMNS = ['Open', 'High', 'Low', 'Close', 'Volume']
LAGS = [0, 1, 2, 3, 5]   # lag 0 = today's values, known when predicting tomorrow
WARMUP_ROWS = 59         # MA60 warm-up; every part drops the same rows so metrics are comparable


def build_full_frame(train_df, test_df):
    """Concatenate train and test in time order so rolling features have history at the
    start of the test period."""
    full = pd.concat([train_df, test_df], ignore_index=True)
    full['Date'] = pd.to_datetime(full['Date'])
    return full.sort_values('Date').reset_index(drop=True)


def add_lag_features(df):
    for lag in LAGS:
        for col in LAG_COLUMNS:
            df[f'{col}_lag_{lag}'] = df[col].shift(lag)
    return df


def split_features(full, feature_cols, split_date, target=None):
    """Build the next-day target and split into train/test rows shared by all parts."""
    full = full.copy()
    full['Target'] = full['Close'].shift(-1) if target is None else target
    full = full.iloc[WARMUP_ROWS:]
    train = full[full['Date'] < split_date].iloc[:-1]   # its target would be the first test-day close
    test = full[full['Date'] >= split_date]
    train = train.dropna(subset=feature_cols + ['Target'])
    test = test.dropna(subset=feature_cols + ['Target'])
    return train[feature_cols], train['Target'], test[feature_cols], test['Target'], train, test


def create_technical_features(df):
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()

    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].rolling(window=20).std()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    df['Momentum'] = df['Close'].pct_change(periods=20)

    df['Consecutive_Up_Days'] = df['Close'].diff().gt(0).rolling(window=10).sum()
    df['Consecutive_Down_Days'] = df['Close'].diff().lt(0).rolling(window=10).sum()

    df['Bollinger_Middle'] = df['Close'].rolling(window=20).mean()
    df['Bollinger_Upper'] = df['Bollinger_Middle'] + 2 * df['Close'].rolling(window=20).std()
    df['Bollinger_Lower'] = df['Bollinger_Middle'] - 2 * df['Close'].rolling(window=20).std()

    return df


def create_extended_features(df):
    """Technical indicators + EMA/MACD, day-of-week dummies, return lags."""
    df = create_technical_features(df)

    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    dow = df['Date'].dt.dayofweek
    for i, name in zip(range(1, 5), ['Tue', 'Wed', 'Thu', 'Fri']):
        df[f'DOW_{name}'] = (dow == i).astype(int)

    for lag in [1, 2, 3]:
        df[f'Return_lag_{lag}'] = df['Daily_Return'].shift(lag)

    return df


def create_return_features(df):
    """Stationary features for return prediction."""
    df = create_extended_features(df)

    ret = df['Close'].pct_change()
    for k in range(0, 5):
        df[f'Ret_lag_{k}'] = ret.shift(k)
    df['Mom5'] = df['Close'].pct_change(5)
    df['Mom20'] = df['Close'].pct_change(20)
    for w in (5, 20, 60):
        df[f'MA_gap_{w}'] = df['Close'] / df[f'MA{w}'] - 1
    df['MACD_rel'] = df['MACD'] / df['Close']
    df['BB_pctB'] = (df['Close'] - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'])
    df['Vol_ratio'] = np.log(df['Volume'] / df['Volume'].rolling(20).mean())
    df['HL_range'] = (df['High'] - df['Low']) / df['Close']
    df['Gap_open'] = df['Open'] / df['Close'].shift(1) - 1
    return df


PART1_FEATURES = [f'{col}_lag_{lag}' for lag in LAGS for col in LAG_COLUMNS]

PART2_EXTRA_FEATURES = [
    'MA5', 'MA20', 'MA60',
    'Daily_Return', 'Volatility',
    'RSI',
    'Momentum',
    'Consecutive_Up_Days', 'Consecutive_Down_Days',
    'Bollinger_Middle', 'Bollinger_Upper', 'Bollinger_Lower'
]
PART2_FEATURES = PART1_FEATURES + PART2_EXTRA_FEATURES

PART3_EXTRA_FEATURES = PART2_EXTRA_FEATURES + [
    'EMA12', 'EMA26', 'MACD', 'MACD_Signal',
    'DOW_Tue', 'DOW_Wed', 'DOW_Thu', 'DOW_Fri',
    'Return_lag_1', 'Return_lag_2', 'Return_lag_3',
]
PART3_FEATURES = PART1_FEATURES + PART3_EXTRA_FEATURES

PART4_FEATURES = (
    [f'Ret_lag_{k}' for k in range(0, 5)]
    + ['Mom5', 'Mom20', 'MA_gap_5', 'MA_gap_20', 'MA_gap_60', 'MACD_rel', 'BB_pctB',
       'Vol_ratio', 'HL_range', 'Gap_open', 'Volatility', 'RSI',
       'DOW_Tue', 'DOW_Wed', 'DOW_Thu', 'DOW_Fri']
)

FEATURE_COLUMNS = {1: PART1_FEATURES, 2: PART2_FEATURES, 3: PART3_FEATURES, 4: PART4_FEATURES}


def build_features(part, full):
    """Feature frame (no target) for the given part, computed on a time-ordered frame."""
    if part == 1:
        return add_lag_features(full)
    if part == 2:
        return add_lag_features(create_technical_features(full))
    if part == 3:
        return add_lag_features(create_extended_features(full))
    if part == 4:
        return create_return_features(full)
    raise ValueError(f'unknown part: {part}')


def prepare_features(part, train_df, test_df):
    full = build_features(part, build_full_frame(train_df, test_df))
    target = full['Close'].shift(-1) / full['Close'] - 1 if part == 4 else None
    return split_features(full, FEATURE_COLUMNS[part], test_df['Date'].min(), target=target)


def prepare_part1_features(train_df, test_df):
    return prepare_features(1, train_df, test_df)


def prepare_part2_features(train_df, test_df):
    return prepare_features(2, train_df, test_df)


def prepare_part3_features(train_df, test_df):
    return prepare_features(3, train_df, test_df)


def prepare_part4_features(train_df, test_df):
    return prepare_features(4, train_df, test_df)
