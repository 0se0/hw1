import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.features import FEATURE_COLUMNS
from src.train import train
from tests.conftest import _make_frame, write_csvs

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('part', [1, 2, 3, 4])
def test_train_saves_a_usable_model_and_scaler(part, train_df, test_df, tmp_path):
    data_dir = write_csvs(tmp_path / 'data', train_df, test_df)
    out_dir = tmp_path / 'models'
    name, model, scaler = train(part, data_dir, out_dir, n_iter=1)

    saved_model = joblib.load(out_dir / f'kospi_part{part}_model.pkl')
    saved_scaler = joblib.load(out_dir / f'kospi_part{part}_scaler.pkl')
    assert type(saved_model) is type(model)
    assert saved_scaler.n_features_in_ == len(FEATURE_COLUMNS[part])

    X = np.zeros((1, len(FEATURE_COLUMNS[part])))
    assert np.isfinite(saved_model.predict(saved_scaler.transform(X))).all()


@pytest.mark.parametrize('part', [1, 2])
def test_model_selection_does_not_depend_on_test_data(part, train_df, test_df, tmp_path):
    """Selection uses training-data CV only, so a different test set must not change the model."""
    other_test = _make_frame(len(test_df), test_df['Date'].iloc[0], seed=99)
    a = write_csvs(tmp_path / 'a', train_df, test_df)
    b = write_csvs(tmp_path / 'b', train_df, other_test)

    name_a, model_a, _ = train(part, a, tmp_path / 'ma')
    name_b, model_b, _ = train(part, b, tmp_path / 'mb')

    assert name_a == name_b
    if hasattr(model_a, 'coef_'):
        np.testing.assert_allclose(model_a.coef_, model_b.coef_)


def test_train_cli_runs_end_to_end(train_df, test_df, tmp_path):
    data_dir = write_csvs(tmp_path / 'data', train_df, test_df)
    out_dir = tmp_path / 'models'
    result = subprocess.run(
        [sys.executable, '-m', 'src.train', '--part', '2', '--data-dir', str(data_dir), '--out-dir', str(out_dir)],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / 'kospi_part2_model.pkl').exists()
    assert 'Selected by CV' in result.stdout
