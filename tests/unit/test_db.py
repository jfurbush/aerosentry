import sqlite3
from pathlib import Path

import pytest

from app.db import (
    SENSOR_COLUMNS,
    DatabaseUnavailable,
    connect,
    feature_history,
    get_engine_history,
    latest_features,
    list_engines,
)
from pipeline.features import feature_columns
from tests.conftest import Artifacts


def test_list_engines_returns_ids_and_cycle_counts_in_order(artifacts: Artifacts) -> None:
    rows = list_engines(artifacts.db_path)
    assert [(r.engine_id, r.n_cycles) for r in rows] == [(1, 120), (2, 150), (3, 100)]


def test_latest_features_is_the_last_cycle_with_every_model_input(artifacts: Artifacts) -> None:
    latest = latest_features(artifacts.db_path, 2)
    assert latest is not None
    assert (latest.engine_id, latest.cycle) == (2, 150)
    assert set(latest.values) == set(feature_columns())


def test_latest_features_for_unknown_engine_is_none(artifacts: Artifacts) -> None:
    assert latest_features(artifacts.db_path, 999) is None


def test_missing_database_is_reported_and_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(DatabaseUnavailable):
        list_engines(missing)
    assert not missing.exists()


def test_file_that_is_not_a_database_is_reported(tmp_path: Path) -> None:
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not a sqlite file" * 50)
    with pytest.raises(DatabaseUnavailable):
        list_engines(junk)


def test_database_without_the_expected_tables_is_reported(tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    with pytest.raises(DatabaseUnavailable):
        latest_features(empty, 1)


def test_connection_is_read_only(artifacts: Artifacts) -> None:
    with connect(artifacts.db_path) as conn, pytest.raises(sqlite3.OperationalError):
        conn.execute("CREATE TABLE should_fail (a INTEGER)")


def test_history_has_every_cycle_in_order_with_all_21_sensors(artifacts: Artifacts) -> None:
    history = get_engine_history(artifacts.db_path, 2)
    assert [row["cycle"] for row in history] == list(range(1, 151))
    assert set(history[0]) == {"cycle", *SENSOR_COLUMNS}
    assert len(SENSOR_COLUMNS) == 21


def test_history_for_unknown_engine_is_empty(artifacts: Artifacts) -> None:
    assert get_engine_history(artifacts.db_path, 999) == []


def test_history_on_missing_database_is_reported_and_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(DatabaseUnavailable):
        get_engine_history(missing, 1)
    assert not missing.exists()


def test_history_without_the_readings_table_is_reported(tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    with pytest.raises(DatabaseUnavailable):
        get_engine_history(empty, 1)


def test_feature_history_has_every_cycle_in_order_and_ends_at_the_latest(
    artifacts: Artifacts,
) -> None:
    history = feature_history(artifacts.db_path, 2)
    assert [f.cycle for f in history] == list(range(1, 151))
    assert {f.engine_id for f in history} == {2}
    assert set(history[0].values) == set(feature_columns())
    assert history[-1] == latest_features(artifacts.db_path, 2)


def test_feature_history_for_unknown_engine_is_empty(artifacts: Artifacts) -> None:
    assert feature_history(artifacts.db_path, 999) == []


def test_feature_history_on_missing_database_is_reported_and_not_created(tmp_path: Path) -> None:
    missing = tmp_path / "nope.db"
    with pytest.raises(DatabaseUnavailable):
        feature_history(missing, 1)
    assert not missing.exists()


def test_feature_history_without_the_features_table_is_reported(tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()
    with pytest.raises(DatabaseUnavailable):
        feature_history(empty, 1)
