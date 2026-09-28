"""Retrain one part end to end and save the CV-selected model and its scaler.

    python -m src.train --part 2 [--data-dir data] [--out-dir models]

Parts 1-2 compare the base models; part 3 tunes six models on price; part 4 tunes six models
on next-day return. The model is chosen by TimeSeriesSplit CV MSE on the training data only.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np

from .data import read_data
from .evaluation import evaluate_model, return_metrics, scale_features
from .features import prepare_features
from .models import PART3_SEARCH_SPACE, PART4_SEARCH_SPACE, train_and_evaluate_models, tune_models


def train(part, data_dir='data', out_dir='models'):
    data_dir, out_dir = Path(data_dir), Path(out_dir)
    df_train = read_data(data_dir / 'kospi_train.csv')
    df_test = read_data(data_dir / 'kospi_test.csv')

    X_tr, y_tr, X_te, y_te, _, te = prepare_features(part, df_train, df_test)
    X_tr_s, X_te_s, scaler = scale_features(X_tr, X_te)
    print(f"Part {part}: {len(X_tr)} train rows, {len(X_te)} test rows, {X_tr.shape[1]} features")

    if part in (1, 2):
        _, (name, model) = train_and_evaluate_models(X_tr_s, y_tr, X_te_s, y_te)
    else:
        space = PART3_SEARCH_SPACE if part == 3 else PART4_SEARCH_SPACE
        estimators, cv_mse = tune_models(space, X_tr_s, y_tr)
        name = min(cv_mse, key=cv_mse.get)
        model = estimators[name]
        y_pred = model.predict(X_te_s)
        if part == 3:
            evaluate_model(y_te, y_pred, name)
        else:
            close_t = te['Close'].values
            metrics = return_metrics(y_te.values, y_pred, close_t, close_t * (1 + y_te.values))
            print({k: round(float(v), 5) for k, v in metrics.items()})

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out_dir / f'kospi_part{part}_model.pkl')
    joblib.dump(scaler, out_dir / f'kospi_part{part}_scaler.pkl')
    print(f"\nSelected by CV: {name} -> saved to {out_dir}/kospi_part{part}_model.pkl")
    return name, model, scaler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--part', type=int, choices=[1, 2, 3, 4], required=True)
    parser.add_argument('--data-dir', default='data')
    parser.add_argument('--out-dir', default='models')
    args = parser.parse_args()
    train(args.part, args.data_dir, args.out_dir)


if __name__ == '__main__':
    main()
