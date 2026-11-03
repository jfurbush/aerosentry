import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

from pipeline import train as train_module
from pipeline.features import feature_columns
from pipeline.train import (
    align_labels,
    bootstrap_rmse_interval,
    last_cycle_rows,
    rmse,
    score,
    train,
)

VIEWS = ["vs_clipped_actual", "vs_unclipped_actual"]
VIEW_KEYS = {"rmse", "rmse_95_interval", "baseline_rmse", "mae", "mean_error", "max_abs_error"}


def _run(raw_dir: Path, out_dir: Path) -> dict:
    return train(
        raw_dir=raw_dir,
        model_path=out_dir / "model.joblib",
        metrics_path=out_dir / "metrics.json",
        n_estimators=20,
    )


# --- error measures -------------------------------------------------------------------


def test_rmse_known_value() -> None:
    assert rmse([0, 0, 0, 0], [3, 3, 3, 3]) == pytest.approx(3.0)
    assert rmse([1, 2, 3], [1, 2, 3]) == 0.0


def test_score_matches_hand_calculation() -> None:
    result = score(actual=[10, 20], predicted=[15, 25], baseline_predicted=[0, 0])
    assert result["rmse"] == pytest.approx(5.0)
    assert result["mae"] == pytest.approx(5.0)
    assert result["mean_error"] == pytest.approx(5.0)  # overestimates remaining life
    assert result["max_abs_error"] == pytest.approx(5.0)
    assert result["baseline_rmse"] == pytest.approx(15.8113883)  # sqrt((100 + 400) / 2)


def test_mean_error_is_negative_when_life_is_underestimated() -> None:
    result = score(actual=[10, 20], predicted=[8, 15], baseline_predicted=[0, 0])
    assert result["mean_error"] == pytest.approx(-3.5)


def test_bootstrap_interval_of_identical_errors_is_a_single_value() -> None:
    low, high = bootstrap_rmse_interval([0, 0, 0, 0], [3, 3, 3, 3])
    assert low == pytest.approx(3.0)
    assert high == pytest.approx(3.0)


def test_bootstrap_interval_is_ordered_and_repeatable() -> None:
    actual = list(range(0, 100, 5))
    predicted = [a + (i % 7) - 3 for i, a in enumerate(actual)]
    first = bootstrap_rmse_interval(actual, predicted)
    assert 0 <= first[0] < first[1]
    assert first == bootstrap_rmse_interval(actual, predicted)


# --- choosing and matching the evaluation rows ------------------------------------------


def test_last_cycle_rows_does_not_depend_on_row_order() -> None:
    frame = pd.DataFrame(
        {"engine_id": [1, 1, 1, 2, 2], "cycle": [1, 3, 2, 2, 1], "s2": [10, 30, 20, 200, 100]}
    )
    last = last_cycle_rows(frame)
    assert last["cycle"].to_dict() == {1: 3, 2: 2}
    assert last["s2"].to_dict() == {1: 30, 2: 200}
    shuffled = last_cycle_rows(frame.sample(frac=1.0, random_state=3))
    assert shuffled.sort_index().equals(last.sort_index())


def test_align_labels_returns_labels_in_engine_order() -> None:
    last_rows = pd.DataFrame({"cycle": [5, 6, 7]}, index=pd.Index([3, 1, 2], name="engine_id"))
    labels = pd.Series([10, 20, 30], index=pd.Index([1, 2, 3], name="engine_id"))
    assert align_labels(last_rows, labels).tolist() == [30, 10, 20]


@pytest.mark.parametrize("label_ids", [[1, 2], [1, 2, 3, 4], [1, 2, 9]])
def test_align_labels_rejects_a_different_set_of_engines(label_ids: list[int]) -> None:
    last_rows = pd.DataFrame({"cycle": [5, 6, 7]}, index=pd.Index([1, 2, 3], name="engine_id"))
    labels = pd.Series(range(len(label_ids)), index=pd.Index(label_ids, name="engine_id"))
    with pytest.raises(ValueError, match="does not match"):
        align_labels(last_rows, labels)


def test_align_labels_rejects_repeated_engine_ids() -> None:
    last_rows = pd.DataFrame({"cycle": [5, 6]}, index=pd.Index([1, 2], name="engine_id"))
    labels = pd.Series([10, 20, 30], index=pd.Index([1, 2, 2], name="engine_id"))
    with pytest.raises(ValueError, match="more than once"):
        align_labels(last_rows, labels)


def test_train_rejects_a_rul_file_for_other_engines(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    (synthetic_raw_dir / "RUL_FD001.txt").write_text("10\n20\n")
    with pytest.raises(ValueError, match="does not match"):
        _run(synthetic_raw_dir, tmp_path)
    assert not (tmp_path / "model.joblib").exists()


# --- the training run -------------------------------------------------------------------


def test_train_writes_model_and_metrics(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    metrics = _run(synthetic_raw_dir, tmp_path)
    assert (tmp_path / "model.joblib").exists()
    assert json.loads((tmp_path / "metrics.json").read_text()) == metrics
    for view in VIEWS:
        assert set(metrics[view]) == VIEW_KEYS
    for key in ["met_target_vs_clipped_actual", "met_target_vs_unclipped_actual", "seed"]:
        assert key in metrics
    assert metrics["n_test_engines"] == 3
    assert metrics["n_train_engines"] == 6


@pytest.mark.parametrize("view", VIEWS)
def test_model_beats_the_mean_baseline_on_synthetic_data(
    synthetic_raw_dir: Path, tmp_path: Path, view: str
) -> None:
    metrics = _run(synthetic_raw_dir, tmp_path)
    assert metrics[view]["rmse"] < metrics[view]["baseline_rmse"]


def test_views_agree_when_no_true_rul_exceeds_the_cap(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    # The synthetic labels are 60, 50 and 60, all below the 125 cap.
    metrics = _run(synthetic_raw_dir, tmp_path)
    assert metrics["vs_clipped_actual"] == metrics["vs_unclipped_actual"]


def test_views_differ_when_a_true_rul_exceeds_the_cap(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    (synthetic_raw_dir / "RUL_FD001.txt").write_text("200\n50\n60\n")
    metrics = _run(synthetic_raw_dir, tmp_path)
    assert metrics["vs_unclipped_actual"]["rmse"] != metrics["vs_clipped_actual"]["rmse"]
    assert metrics["vs_unclipped_actual"]["max_abs_error"] > 75  # 200 against a model capped at 125


def test_saved_bundle_has_what_the_service_needs(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    _run(synthetic_raw_dir, tmp_path)
    bundle = joblib.load(tmp_path / "model.joblib")
    assert bundle["feature_columns"] == feature_columns(bundle["sensors"], bundle["windows"])
    assert bundle["rul_clip"] == 125
    assert hasattr(bundle["model"], "predict")


def test_training_is_reproducible(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    first = _run(synthetic_raw_dir, tmp_path / "a")
    second = _run(synthetic_raw_dir, tmp_path / "b")
    assert first == second


def test_prediction_is_within_the_clipped_range(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    from pipeline.data import load_cmapss
    from pipeline.features import add_rolling_features

    _run(synthetic_raw_dir, tmp_path)
    bundle = joblib.load(tmp_path / "model.joblib")
    test_df = add_rolling_features(
        load_cmapss(synthetic_raw_dir / "test_FD001.txt"), bundle["sensors"], bundle["windows"]
    )
    predictions = bundle["model"].predict(test_df[bundle["feature_columns"]])
    assert predictions.min() >= 0
    assert predictions.max() <= 125


def test_failed_save_keeps_the_previous_model_and_leaves_no_temp_file(
    synthetic_raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _run(synthetic_raw_dir, tmp_path)
    before = (tmp_path / "model.joblib").read_bytes()

    def boom(*args: object, **kwargs: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(joblib, "dump", boom)
    with pytest.raises(OSError, match="disk full"):
        _run(synthetic_raw_dir, tmp_path)
    assert (tmp_path / "model.joblib").read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_command_line_entry_point_trains_and_prints_metrics(
    synthetic_raw_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
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
    assert '"vs_clipped_actual"' in output
    assert "above the" in output
    assert (tmp_path / "m.joblib").exists()
