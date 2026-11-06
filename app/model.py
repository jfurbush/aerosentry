"""Load the trained RUL model and make single-engine predictions (FR-4, FR-7).

The .joblib file is a pickle, so only load files this project built itself. In the Docker
image the file is created at build time and nothing at runtime can replace it.
"""

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class ModelUnavailable(Exception):
    """The model file is missing, unreadable, or not in the expected format."""


class PredictionError(Exception):
    """The model could not produce a valid prediction from the stored features."""


@dataclass(frozen=True)
class RulModel:
    estimator: Any
    feature_columns: list[str]
    rul_clip: int


def load_model(path: Path) -> RulModel:
    """Load the model bundle written by `python -m pipeline.train`."""
    if not path.is_file():
        raise ModelUnavailable(f"model file not found: {path}")
    try:
        bundle = joblib.load(path)
        estimator = bundle["model"]
        columns = list(bundle["feature_columns"])
        clip = int(bundle["rul_clip"])
        # One row at a time is faster on a single thread than spinning up a worker pool.
        if "n_jobs" in estimator.get_params():
            estimator.set_params(n_jobs=1)
    except Exception as exc:  # a bad file can fail in many ways, all of which mean "unusable"
        raise ModelUnavailable(f"cannot load model {path}: {exc!r}") from exc
    return RulModel(estimator=estimator, feature_columns=columns, rul_clip=clip)


def predict_rul(model: RulModel, values: dict[str, float]) -> float:
    """Predict remaining useful life in cycles, kept within 0 and the training cap."""
    missing = [name for name in model.feature_columns if name not in values]
    if missing:
        raise PredictionError(f"stored features are missing model inputs: {missing[:3]}...")
    row = pd.DataFrame(
        [[values[name] for name in model.feature_columns]], columns=model.feature_columns
    )
    raw = float(model.estimator.predict(row)[0])
    if math.isnan(raw):
        raise PredictionError("model returned NaN")
    return round(min(max(raw, 0.0), float(model.rul_clip)), 1)
