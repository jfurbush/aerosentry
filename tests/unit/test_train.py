import json
from pathlib import Path

import joblib
import pytest

from pipeline.features import feature_columns
from pipeline.train import rmse, train


def _run(raw_dir: Path, out_dir: Path) -> dict:
    return train(
        raw_dir=raw_dir,
        model_path=out_dir / "model.joblib",
        metrics_path=out_dir / "metrics.json",
        n_estimators=20,
    )


def test_rmse_known_value() -> None:
    assert rmse([0, 0, 0, 0], [3, 3, 3, 3]) == pytest.approx(3.0)
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_train_writes_model_and_metrics(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    metrics = _run(synthetic_raw_dir, tmp_path)
    assert (tmp_path / "model.joblib").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved == metrics
    for key in ["rmse_cycles", "baseline_rmse_cycles", "met_target", "seed", "windows"]:
        assert key in saved
    assert saved["n_test_engines"] == 3
    assert saved["n_train_engines"] == 6


def test_model_beats_the_mean_baseline_on_synthetic_data(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    metrics = _run(synthetic_raw_dir, tmp_path)
    assert metrics["rmse_cycles"] < metrics["baseline_rmse_cycles"]


def test_saved_bundle_has_what_the_service_needs(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    _run(synthetic_raw_dir, tmp_path)
    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["feature_columns"] == feature_columns(bundle["sensors"], bundle["windows"])
    assert bundle["rul_clip"] == 125
    assert hasattr(bundle["model"], "predict")


def test_training_is_reproducible(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    first = _run(synthetic_raw_dir, tmp_path / "a")
    second = _run(synthetic_raw_dir, tmp_path / "b")
    assert first["rmse_cycles"] == second["rmse_cycles"]


def test_prediction_is_within_the_clipped_range(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    _run(synthetic_raw_dir, tmp_path)
    bundle = joblib.load(tmp_path / "model.joblib")
    from pipeline.data import load_cmapss
    from pipeline.features import add_rolling_features

    test_df = add_rolling_features(
        load_cmapss(synthetic_raw_dir / "test_FD001.txt"), bundle["sensors"], bundle["windows"]
    )
    predictions = bundle["model"].predict(test_df[bundle["feature_columns"]])
    assert predictions.min() >= 0
    assert predictions.max() <= 125


def test_command_line_entry_point_trains_and_prints_metrics(
    synthetic_raw_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import sys

    from pipeline import train as train_module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--raw-dir", str(synthetic_raw_dir),
            "--model-path", str(tmp_path / "m.joblib"),
            "--metrics-path", str(tmp_path / "m.json"),
        ],
    )  # fmt: skip
    monkeypatch.setattr(train_module, "RMSE_TARGET", 0.0)  # force the "above target" note
    train_module.main()
    output = capsys.readouterr().out
    assert '"rmse_cycles"' in output
    assert "above the" in output
    assert (tmp_path / "m.joblib").exists()
