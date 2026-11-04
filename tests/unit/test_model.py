from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from app.db import latest_features
from app.model import ModelUnavailable, PredictionError, RulModel, load_model, predict_rul
from tests.conftest import Artifacts


class _Stub:
    """Stands in for a trained model that always returns the same number."""

    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, rows: Any) -> np.ndarray:
        return np.array([self.value])


def _stub_model(value: float) -> RulModel:
    return RulModel(estimator=_Stub(value), feature_columns=["a"], rul_clip=125)


def test_load_model_reads_the_bundle(artifacts: Artifacts) -> None:
    model = load_model(artifacts.model_path)
    assert model.rul_clip == 125
    assert len(model.feature_columns) == 14 * 5
    assert model.estimator.get_params()["n_jobs"] == 1


def test_missing_model_file_is_reported(tmp_path: Path) -> None:
    with pytest.raises(ModelUnavailable, match="not found"):
        load_model(tmp_path / "nope.joblib")


def test_corrupt_model_file_is_reported(tmp_path: Path) -> None:
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a joblib file")
    with pytest.raises(ModelUnavailable):
        load_model(bad)


def test_bundle_without_the_expected_keys_is_reported(tmp_path: Path) -> None:
    odd = tmp_path / "odd.joblib"
    joblib.dump({"something": "else"}, odd)
    with pytest.raises(ModelUnavailable):
        load_model(odd)


def test_prediction_is_in_range_and_rounded_to_one_decimal(artifacts: Artifacts) -> None:
    model = load_model(artifacts.model_path)
    latest = latest_features(artifacts.db_path, 1)
    assert latest is not None
    rul = predict_rul(model, latest.values)
    assert 0.0 <= rul <= 125.0
    assert rul == round(rul, 1)


@pytest.mark.parametrize(("raw", "expected"), [(500.0, 125.0), (-5.0, 0.0), (42.34, 42.3)])
def test_prediction_is_kept_within_the_training_range(raw: float, expected: float) -> None:
    assert predict_rul(_stub_model(raw), {"a": 1.0}) == expected


def test_missing_stored_feature_is_an_error() -> None:
    with pytest.raises(PredictionError, match="missing"):
        predict_rul(_stub_model(10.0), {"b": 1.0})


def test_nan_prediction_is_an_error() -> None:
    with pytest.raises(PredictionError, match="NaN"):
        predict_rul(_stub_model(float("nan")), {"a": 1.0})
