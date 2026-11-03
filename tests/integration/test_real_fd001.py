"""Checks against the real FD001 files committed in data/raw. Skipped if they are absent."""

from pathlib import Path

import pytest

from pipeline.data import CONSTANT_SENSORS, add_rul_target, load_cmapss, load_test_rul
from pipeline.train import DEFAULT_RAW_DIR, train

pytestmark = pytest.mark.skipif(
    not (DEFAULT_RAW_DIR / "train_FD001.txt").exists(),
    reason="FD001 files not present in data/raw",
)


def test_fd001_has_the_documented_shape() -> None:
    train_df = load_cmapss(DEFAULT_RAW_DIR / "train_FD001.txt")
    test_df = load_cmapss(DEFAULT_RAW_DIR / "test_FD001.txt")
    assert train_df.shape == (20631, 26)
    assert test_df.shape == (13096, 26)
    assert train_df["engine_id"].nunique() == 100
    assert test_df["engine_id"].nunique() == 100
    assert len(load_test_rul(DEFAULT_RAW_DIR / "RUL_FD001.txt")) == 100


def test_the_sensors_we_drop_really_are_constant() -> None:
    train_df = load_cmapss(DEFAULT_RAW_DIR / "train_FD001.txt")
    for sensor in CONSTANT_SENSORS:
        assert train_df[sensor].nunique() <= 2


def test_rul_target_on_real_data_is_clipped() -> None:
    df = add_rul_target(load_cmapss(DEFAULT_RAW_DIR / "train_FD001.txt"))
    assert df["rul"].max() == 125
    assert df["rul"].min() == 0


def test_model_beats_baseline_on_real_test_set(tmp_path: Path) -> None:
    metrics = train(
        model_path=tmp_path / "model.joblib",
        metrics_path=tmp_path / "metrics.json",
        n_estimators=50,
    )
    assert metrics["n_test_engines"] == 100
    for view in ["vs_clipped_actual", "vs_unclipped_actual"]:
        assert metrics[view]["rmse"] < metrics[view]["baseline_rmse"]
