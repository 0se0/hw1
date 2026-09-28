"""Predict from a saved model using the latest row of a price history CSV.

    python -m src.predict --part 2 --history data/kospi_test.csv [--model-dir models]

Parts 1-3 output the next trading day's close; part 4 outputs the next-day return.
The history needs about 60 trading days so the rolling indicators are defined.
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd

from .data import read_data
from .features import FEATURE_COLUMNS, build_features


def predict_next(part, history_df, model_dir='models'):
    model_dir = Path(model_dir)
    model = joblib.load(model_dir / f'kospi_part{part}_model.pkl')
    scaler = joblib.load(model_dir / f'kospi_part{part}_scaler.pkl')

    history = history_df.copy()
    history['Date'] = pd.to_datetime(history['Date'])
    history = history.sort_values('Date').reset_index(drop=True)

    cols = FEATURE_COLUMNS[part]
    latest = build_features(part, history).iloc[[-1]]
    if latest[cols].isna().any().any():
        raise ValueError('not enough history to compute all features; provide at least ~60 trading days')

    pred = float(model.predict(scaler.transform(latest[cols]))[0])
    last_close = float(latest['Close'].iloc[0])
    result = {'as_of': latest['Date'].iloc[0].date(), 'last_close': last_close}
    if part == 4:
        result['predicted_return'] = pred
        result['implied_next_close'] = last_close * (1 + pred)
    else:
        result['predicted_next_close'] = pred
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--part', type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument('--history', required=True, help='CSV with Date, Open, Low, High, Close, Volume')
    parser.add_argument('--model-dir', default='models')
    args = parser.parse_args()
    for key, value in predict_next(args.part, read_data(args.history), args.model_dir).items():
        print(f"{key}: {value}")


if __name__ == '__main__':
    main()
