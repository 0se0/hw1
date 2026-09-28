import joblib
import numpy as np
from sklearn.linear_model import Ridge

from src.evaluation import scale_features
from src.features import FEATURE_COLUMNS, build_features, build_full_frame, prepare_features
from src.predict import predict_next


def _train_and_save(part, train_df, test_df, tmp_path):
    X_tr, y_tr, X_te, _, _, _ = prepare_features(part, train_df, test_df)
    X_tr_s, _, scaler = scale_features(X_tr, X_te)
    model = Ridge(alpha=10.0).fit(X_tr_s, y_tr)
    joblib.dump(model, tmp_path / f'kospi_part{part}_model.pkl')
    joblib.dump(scaler, tmp_path / f'kospi_part{part}_scaler.pkl')
    return model, scaler


def test_predict_matches_training_pipeline(train_df, test_df, tmp_path):
    """Inference must build features exactly as training did (no train/serve skew)."""
    for part in (2, 4):
        model, scaler = _train_and_save(part, train_df, test_df, tmp_path)
        history = build_full_frame(train_df, test_df)
        latest = build_features(part, history.copy()).iloc[[-1]]
        expected = float(model.predict(scaler.transform(latest[FEATURE_COLUMNS[part]]))[0])

        out = predict_next(part, history, tmp_path)
        key = 'predicted_return' if part == 4 else 'predicted_next_close'
        assert np.isclose(out[key], expected)


def test_predict_rejects_short_history(train_df, test_df, tmp_path):
    _train_and_save(2, train_df, test_df, tmp_path)
    try:
        predict_next(2, train_df.iloc[:20], tmp_path)
    except ValueError as e:
        assert 'not enough history' in str(e)
    else:
        raise AssertionError('expected ValueError')
