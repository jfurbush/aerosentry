"""Load FD001 test engines and their precomputed features into SQLite (FR-1, FR-5).

Run as: python -m pipeline.load_db
Writes data/aerosentry.db

Tables:
  engines   one row per engine: engine_id, n_cycles, last_cycle, actual_rul
  readings  every raw reading (settings and all 21 sensors) per engine and cycle
  features  the model inputs per engine and cycle (raw informative sensors + rolling stats)

The service looks up the latest `features` row for an engine, so nothing is computed
at request time. `actual_rul` comes from RUL_FD001.txt and is only for comparison.
"""

import argparse
import os
import sqlite3
from collections.abc import Sequence
from contextlib import closing
from pathlib import Path

from pipeline.data import INFORMATIVE_SENSORS, load_cmapss, load_test_rul
from pipeline.features import DEFAULT_WINDOWS, add_rolling_features, feature_columns

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_DB_PATH = ROOT / "data" / "aerosentry.db"


def build_database(
    raw_dir: Path = DEFAULT_RAW_DIR,
    db_path: Path = DEFAULT_DB_PATH,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> dict[str, int]:
    """Create the SQLite database from the FD001 test files. Returns row counts per table.

    The file is built under a temporary name and then moved into place, so a failed
    run never leaves a half-written database behind.
    """
    readings = load_cmapss(raw_dir / "test_FD001.txt")
    actual_rul = load_test_rul(raw_dir / "RUL_FD001.txt")

    engine_ids = sorted(readings["engine_id"].unique())
    if list(actual_rul.index) != engine_ids:
        raise ValueError(
            f"RUL file has {len(actual_rul)} engines but the test file has {len(engine_ids)}"
        )

    sensors = list(INFORMATIVE_SENSORS)
    with_features = add_rolling_features(readings, sensors, windows)
    features = with_features[["engine_id", "cycle", *feature_columns(sensors, windows)]]

    engines = (
        readings.groupby("engine_id")["cycle"]
        .agg(n_cycles="count", last_cycle="max")
        .join(actual_rul)
        .reset_index()
    )

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".db.tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(tmp_path)) as conn:
            engines.to_sql("engines", conn, index=False)
            readings.to_sql("readings", conn, index=False)
            features.to_sql("features", conn, index=False)
            conn.execute("CREATE UNIQUE INDEX idx_engines_id ON engines (engine_id)")
            conn.execute("CREATE UNIQUE INDEX idx_readings_key ON readings (engine_id, cycle)")
            conn.execute("CREATE UNIQUE INDEX idx_features_key ON features (engine_id, cycle)")
            conn.commit()
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, db_path)

    return {"engines": len(engines), "readings": len(readings), "features": len(features)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AeroSentry SQLite database.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    args = parser.parse_args()

    counts = build_database(args.raw_dir, args.db_path)
    print(f"Wrote {args.db_path}: {counts}")


if __name__ == "__main__":
    main()
