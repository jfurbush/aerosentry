"""Read-only access to the AeroSentry SQLite database (FR-5, FR-6, FR-7).

The service never changes data, so the database is opened read-only. That also means a
missing file is an error, instead of SQLite quietly creating an empty one.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class DatabaseUnavailable(Exception):
    """The database file is missing, unreadable, or does not have the expected tables."""


@dataclass(frozen=True)
class EngineRow:
    engine_id: int
    n_cycles: int


@dataclass(frozen=True)
class LatestFeatures:
    engine_id: int
    cycle: int
    values: dict[str, float]


@contextmanager
def connect(db_path: Path) -> Iterator[sqlite3.Connection]:
    """Open the database read-only and close it afterwards."""
    if not db_path.is_file():
        raise DatabaseUnavailable(f"database file not found: {db_path}")
    uri = f"{db_path.resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
    except sqlite3.Error as exc:
        raise DatabaseUnavailable(f"cannot open database {db_path}: {exc}") from exc
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def list_engines(db_path: Path) -> list[EngineRow]:
    """Return every engine with its number of recorded cycles, ordered by engine ID."""
    try:
        with connect(db_path) as conn:
            rows = conn.execute(
                "SELECT engine_id, n_cycles FROM engines ORDER BY engine_id"
            ).fetchall()
    except sqlite3.Error as exc:
        raise DatabaseUnavailable(f"cannot read engines from {db_path}: {exc}") from exc
    return [EngineRow(engine_id=int(r["engine_id"]), n_cycles=int(r["n_cycles"])) for r in rows]


def latest_features(db_path: Path, engine_id: int) -> LatestFeatures | None:
    """Return the precomputed features for an engine's latest cycle, or None if unknown."""
    try:
        with connect(db_path) as conn:
            row = conn.execute(
                "SELECT * FROM features WHERE engine_id = ? ORDER BY cycle DESC LIMIT 1",
                (engine_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise DatabaseUnavailable(f"cannot read features from {db_path}: {exc}") from exc
    if row is None:
        return None
    values = {key: row[key] for key in row.keys() if key not in ("engine_id", "cycle")}  # noqa: SIM118
    return LatestFeatures(engine_id=int(row["engine_id"]), cycle=int(row["cycle"]), values=values)

def get_engine_history(db_path: Path, engine_id: int) -> list[dict[str, Any]]:
    """Retrieve all historical readings for a specific engine ordered by cycle."""
    import sqlite3
    sensor_cols = [f"s{i}" for i in range(1, 22)]
    cols_query = ", ".join(sensor_cols)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            f"SELECT cycle, {cols_query} FROM readings WHERE engine_id = ? ORDER BY cycle ASC",  # noqa: S608
            (engine_id,),
        )
        return [dict(row) for row in cursor.fetchall()]
