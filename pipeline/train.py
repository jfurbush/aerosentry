"""Train the RUL regression model, save it as .joblib and report test RMSE (FR-4, NFR-1, NFR-6).

Run as: python -m pipeline.train
Writes models/rul_model.joblib and models/metrics.json.

Evaluation follows the usual FD001 protocol: each test engine is cut off before failure,
and the model is scored on its last recorded cycle against the value in RUL_FD001.txt.
"""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
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


def rmse(actual: ArrayLike, predicted: ArrayLike) -> float:
    """Root mean squared error in cycles."""
    diff = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


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
    last_rows = test_df.groupby("engine_id").tail(1).set_index("engine_id")
    actual_rul = load_test_rul(raw_dir / "RUL_FD001.txt").loc[last_rows.index]
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
    clipped_actual = actual_rul.clip(upper=RUL_CLIP)

    metrics: dict[str, Any] = {
        "rmse_cycles": round(rmse(clipped_actual, predicted), 3),
        "baseline_rmse_cycles": round(rmse(clipped_actual, baseline_predicted), 3),
        "rmse_vs_unclipped_actual_cycles": round(rmse(actual_rul, predicted), 3),
        "target_rmse_cycles": RMSE_TARGET,
        "met_target": bool(rmse(clipped_actual, predicted) <= RMSE_TARGET),
        "evaluated_on": "last recorded cycle of each FD001 test engine",
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
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the AeroSentry RUL model.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--metrics-path", type=Path, default=DEFAULT_METRICS_PATH)
    args = parser.parse_args()

    metrics = train(args.raw_dir, args.model_path, args.metrics_path)
    print(json.dumps(metrics, indent=2))
    if not metrics["met_target"]:
        print(f"Note: RMSE {metrics['rmse_cycles']} is above the {RMSE_TARGET} cycle target.")


if __name__ == "__main__":
    main()
