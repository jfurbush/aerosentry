"""Shared fixtures: a small synthetic CMAPSS-style dataset so tests do not need FD001."""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest

from pipeline.data import SENSOR_COLUMNS
from pipeline.load_db import build_database
from pipeline.train import train


def _engine_rows(engine_id: int, n_cycles: int, rng: np.random.Generator) -> list[str]:
    lines = []
    for cycle in range(1, n_cycles + 1):
        wear = cycle / n_cycles  # 0 at start, 1 at failure
        values = [float(engine_id), float(cycle), 0.0, 0.0, 100.0]
        for i, _ in enumerate(SENSOR_COLUMNS, start=1):
            if i in (2, 3, 4, 7, 11, 12):  # a few sensors drift with wear
                values.append(500.0 + 40.0 * wear + rng.normal(0, 1.0))
            else:
                values.append(500.0 + rng.normal(0, 0.1))
        fields = [str(int(values[0])), str(int(values[1]))] + [f"{v:.4f}" for v in values[2:]]
        lines.append(" ".join(fields))
    return lines


def write_synthetic_raw_dir(root: Path) -> Path:
    """Write train_FD001.txt, test_FD001.txt and RUL_FD001.txt for a tiny fake fleet."""
    rng = np.random.default_rng(0)
    root.mkdir(parents=True, exist_ok=True)

    train_lines: list[str] = []
    for engine_id, n_cycles in enumerate([160, 180, 200, 170, 190, 210], start=1):
        train_lines += _engine_rows(engine_id, n_cycles, rng)
    (root / "train_FD001.txt").write_text("\n".join(train_lines) + "\n")

    # Test engines are cut off before failure; RUL file says how many cycles were left.
    test_lines: list[str] = []
    rul_values = []
    for engine_id, (full, kept) in enumerate([(180, 120), (200, 150), (160, 100)], start=1):
        rows = _engine_rows(engine_id, full, rng)[:kept]
        test_lines += rows
        rul_values.append(full - kept)
    (root / "test_FD001.txt").write_text("\n".join(test_lines) + "\n")
    (root / "RUL_FD001.txt").write_text("\n".join(str(v) for v in rul_values) + "\n")

    return root


@pytest.fixture
def synthetic_raw_dir(tmp_path: Path) -> Path:
    return write_synthetic_raw_dir(tmp_path / "raw")


class Artifacts(NamedTuple):
    db_path: Path
    model_path: Path


@pytest.fixture
def artifacts(synthetic_raw_dir: Path, tmp_path: Path) -> Artifacts:
    """A small database and a small trained model built by the real pipeline code."""
    db_path = tmp_path / "aerosentry.db"
    model_path = tmp_path / "rul_model.joblib"
    build_database(synthetic_raw_dir, db_path)
    train(
        raw_dir=synthetic_raw_dir,
        model_path=model_path,
        metrics_path=tmp_path / "metrics.json",
        n_estimators=10,
    )
    return Artifacts(db_path=db_path, model_path=model_path)
