from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pytest

from app.db import feature_history, latest_features
from app.model import (
    ModelUnavailable,
    PredictionError,
    RulModel,
    load_model,
    predict_rul,
    predict_rul_series,
)
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


def test_bundle_whose_model_is_not_an_estimator_is_reported(tmp_path: Path) -> None:
    """The right keys but the wrong object must be ModelUnavailable, not an AttributeError."""
    odd = tmp_path / "wrong-object.joblib"
    joblib.dump({"model": object(), "feature_columns": ["a"], "rul_clip": 125}, odd)
    with pytest.raises(ModelUnavailable, match="get_params"):
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


class _Values:
    """Stands in for a trained model that returns a fixed list of numbers, one per row."""

    def __init__(self, values: list[float]) -> None:
        self.values = values

    def predict(self, rows: Any) -> np.ndarray:
        return np.array(self.values)


def _series_model(values: list[float]) -> RulModel:
    return RulModel(estimator=_Values(values), feature_columns=["a"], rul_clip=125)


def test_series_matches_one_prediction_per_row(artifacts: Artifacts) -> None:
    model = load_model(artifacts.model_path)
    rows = [f.values for f in feature_history(artifacts.db_path, 1)]
    assert predict_rul_series(model, rows) == [predict_rul(model, values) for values in rows]


def test_series_matches_the_estimator_called_directly(artifacts: Artifacts) -> None:
    """Compare with scikit-learn's own output, not with another path through our code."""
    model = load_model(artifacts.model_path)
    rows = [f.values for f in feature_history(artifacts.db_path, 1)]
    frame = pd.DataFrame(rows)[model.feature_columns]
    expected = [
        round(min(max(float(raw), 0.0), model.rul_clip), 1)
        for raw in model.estimator.predict(frame)
    ]
    assert len(expected) == len(rows)
    assert predict_rul_series(model, rows) == expected


def test_series_of_nothing_is_empty() -> None:
    assert predict_rul_series(_series_model([]), []) == []


def test_series_is_kept_within_zero_and_the_cap() -> None:
    model = _series_model([-5.0, 60.04, 400.0])
    assert predict_rul_series(model, [{"a": 1.0}] * 3) == [0.0, 60.0, 125.0]


def test_series_reports_missing_inputs() -> None:
    with pytest.raises(PredictionError, match="missing"):
        predict_rul_series(_series_model([1.0, 2.0]), [{"a": 1.0}, {"b": 2.0}])


def test_series_reports_a_nan_prediction() -> None:
    with pytest.raises(PredictionError, match="NaN"):
        predict_rul_series(_series_model([10.0, float("nan")]), [{"a": 1.0}] * 2)
