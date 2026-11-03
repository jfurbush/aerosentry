"""Train the RUL regression model, save it as .joblib and report test RMSE (FR-4, NFR-1, NFR-6).

Run as: python -m pipeline.train
Writes models/rul_model.joblib and models/metrics.json.

Evaluation: each FD001 test engine is cut off before failure, and the model is scored on the
last recorded cycle of each test engine against the true RUL in RUL_FD001.txt. This measures
accuracy at that last recorded cycle, not accuracy over an engine's whole life.

Model settings are fixed in code and were chosen before looking at test results. If tuning is
ever added, do it with cross-validation over the training engines, never on the test set.
"""

import argparse
import json
import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from numpy.typing import ArrayLike
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor

from pipeline.data import (
    INFORMATIVE_SENSORS,
    RUL_CLIP,
    add_rul_target,
    load_cmapss,
    load_test_rul,
)
from pipeline.features import DEFAULT_WINDOWS, add_rolling_features, feature_columns

SEED = 42
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_MODEL_PATH = ROOT / "models" / "rul_model.joblib"
DEFAULT_METRICS_PATH = ROOT / "models" / "metrics.json"
RMSE_TARGET = 25.0
BOOTSTRAP_RESAMPLES = 2000


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root mean squared error in cycles."""
    diff = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def bootstrap_rmse_interval(
    actual: ArrayLike,
    predicted: ArrayLike,
    seed: int = SEED,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """95% interval for the RMSE, from resampling the engines with replacement.

    With only 100 test engines the RMSE is uncertain, so the second decimal means little.
    """
    a = np.asarray(actual, dtype=float)
    p = np.asarray(predicted, dtype=float)
    picks = np.random.default_rng(seed).integers(0, len(a), size=(resamples, len(a)))
    scores = np.sqrt(np.mean((a[picks] - p[picks]) ** 2, axis=1))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)


def score(actual: ArrayLike, predicted: ArrayLike, baseline_predicted: ArrayLike) -> dict[str, Any]:
    """Error measures of a prediction against one version of the true RUL, in cycles.

    `mean_error` is predicted minus actual, so a positive value means the model overestimates
    remaining life. For maintenance planning that is the risky direction.
    """
    a = np.asarray(actual, dtype=float)
    error = np.asarray(predicted, dtype=float) - a
    low, high = bootstrap_rmse_interval(a, predicted)
    return {
        "rmse": rmse(a, predicted),
        "rmse_95_interval": [low, high],
        "baseline_rmse": rmse(a, baseline_predicted),
        "mae": float(np.mean(np.abs(error))),
        "mean_error": float(np.mean(error)),
        "max_abs_error": float(np.max(np.abs(error))),
    }


def last_cycle_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Return each engine's row with the highest cycle number, indexed by engine ID.

    The result does not depend on the order of the rows in `df`.
    """
    df = df.reset_index(drop=True)
    return df.loc[df.groupby("engine_id")["cycle"].idxmax()].set_index("engine_id")


def align_labels(last_rows: pd.DataFrame, actual_rul: pd.Series) -> pd.Series:
    """Match RUL labels to test engines by engine ID.

    Raises ValueError if the label file and the test file do not describe the same engines.
    """
    if not actual_rul.index.is_unique:
        raise ValueError("RUL file lists the same engine ID more than once")
    engines, labels = set(last_rows.index), set(actual_rul.index)
    if engines != labels:
        raise ValueError(
            f"RUL file does not match the test file: {len(labels)} labels, {len(engines)} engines"
        )
    return actual_rul.loc[last_rows.index]


def _write_atomically(target: Path, write: Callable[[Path], object]) -> None:
    """Write to a temporary file, then move it into place, so a failure leaves no partial file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    try:
        write(tmp)
        os.replace(tmp, target)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _rounded(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _rounded(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_rounded(item) for item in value]
    if isinstance(value, float):
        return round(value, 3)
    return value


def train(
    raw_dir: Path = DEFAULT_RAW_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    metrics_path: Path = DEFAULT_METRICS_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
    n_estimators: int = 200,
) -> dict[str, Any]:
    """Fit the model on FD001 training data, score it on the test set, and save both outputs."""
    sensors = list(INFORMATIVE_SENSORS)
    columns = feature_columns(sensors, windows)

    train_df = add_rolling_features(
        add_rul_target(load_cmapss(raw_dir / "train_FD001.txt")), sensors, windows
    )
    x_train, y_train = train_df[columns], train_df["rul"]

    test_df = add_rolling_features(load_cmapss(raw_dir / "test_FD001.txt"), sensors, windows)
    last_rows = last_cycle_rows(test_df)
    actual_rul = align_labels(last_rows, load_test_rul(raw_dir / "RUL_FD001.txt"))
    x_test = last_rows[columns]

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=3,
        max_features=0.5,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    baseline = DummyRegressor(strategy="mean").fit(x_train, y_train)

    predicted = model.predict(x_test)
    baseline_predicted = baseline.predict(x_test)
    vs_clipped = score(actual_rul.clip(upper=RUL_CLIP), predicted, baseline_predicted)
    vs_unclipped = score(actual_rul, predicted, baseline_predicted)

    metrics: dict[str, Any] = {
        "units": "operating cycles",
        "evaluated_on": (
            "last recorded cycle of each FD001 test engine, against the labels in "
            "RUL_FD001.txt. This is accuracy at the last recorded cycle, not over a whole life"
        ),
        "vs_clipped_actual": _rounded(vs_clipped),
        "vs_unclipped_actual": _rounded(vs_unclipped),
        "mean_error_note": "predicted minus actual; positive means remaining life is overestimated",
        "target_rmse": RMSE_TARGET,
        "met_target_vs_clipped_actual": bool(vs_clipped["rmse"] <= RMSE_TARGET),
        "met_target_vs_unclipped_actual": bool(vs_unclipped["rmse"] <= RMSE_TARGET),
        "n_train_rows": int(len(x_train)),
        "n_train_engines": int(train_df["engine_id"].nunique()),
        "n_test_engines": int(len(x_test)),
        "model": "RandomForestRegressor",
        "n_estimators": n_estimators,
        "rul_clip": RUL_CLIP,
        "windows": list(windows),
        "seed": SEED,
        "sklearn_version": sklearn.__version__,
    }

    bundle = {
        "model": model,
        "feature_columns": columns,
        "sensors": sensors,
        "windows": list(windows),
        "rul_clip": RUL_CLIP,
    }
    _write_atomically(model_path, lambda tmp: joblib.dump(bundle, tmp))
    _write_atomically(
        metrics_path, lambda tmp: tmp.write_text(json.dumps(metrics, indent=2) + "\n")
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AeroSentry RUL model.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    metrics = train(args.raw_dir, args.model_path, args.metrics_path)
    print(json.dumps(metrics, indent=2))
    if not (metrics["met_target_vs_clipped_actual"] and metrics["met_target_vs_unclipped_actual"]):
        print(f"Note: RMSE is above the {RMSE_TARGET} cycle target for at least one comparison.")


if __name__ == "__main__":
    main()
