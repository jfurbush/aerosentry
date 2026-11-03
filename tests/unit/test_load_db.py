import sqlite3
import sys
from contextlib import closing
from pathlib import Path

import pytest

from pipeline import load_db as load_db_module
from pipeline.features import feature_columns
from pipeline.load_db import build_database


def _query(db: Path, sql: str) -> list[tuple]:
    with closing(sqlite3.connect(db)) as conn:
        return conn.execute(sql).fetchall()


def test_build_database_reports_row_counts(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    counts = build_database(synthetic_raw_dir, tmp_path / "t.db")
    # Test engines keep 120, 150 and 100 cycles in the synthetic data.
    assert counts == {"engines": 3, "readings": 370, "features": 370}


def test_engines_table_has_cycles_and_actual_rul(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    build_database(synthetic_raw_dir, db)
    rows = _query(db, "SELECT engine_id, n_cycles, last_cycle, actual_rul FROM engines")
    assert rows == [(1, 120, 120, 60), (2, 150, 150, 50), (3, 100, 100, 60)]


def test_features_table_has_every_model_column(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    build_database(synthetic_raw_dir, db)
    with closing(sqlite3.connect(db)) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(features)")]
    assert columns == ["engine_id", "cycle", *feature_columns()]


def test_readings_table_keeps_all_21_sensors(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    build_database(synthetic_raw_dir, db)
    with closing(sqlite3.connect(db)) as conn:
        columns = [row[1] for row in conn.execute("PRAGMA table_info(readings)")]
    assert len(columns) == 26
    assert "s1" in columns and "s21" in columns


def test_latest_feature_row_lookup_works(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    build_database(synthetic_raw_dir, db)
    rows = _query(db, "SELECT cycle FROM features WHERE engine_id = 2 ORDER BY cycle DESC LIMIT 1")
    assert rows == [(150,)]


def test_rebuilding_replaces_the_old_database(synthetic_raw_dir: Path, tmp_path: Path) -> None:
    db = tmp_path / "t.db"
    build_database(synthetic_raw_dir, db)
    build_database(synthetic_raw_dir, db)
    assert _query(db, "SELECT COUNT(*) FROM engines") == [(3,)]
    assert not (tmp_path / "t.db.tmp").exists()


def test_mismatched_rul_file_is_rejected_and_leaves_no_database(
    synthetic_raw_dir: Path, tmp_path: Path
) -> None:
    (synthetic_raw_dir / "RUL_FD001.txt").write_text("10\n20\n")
    db = tmp_path / "t.db"
    with pytest.raises(ValueError, match="RUL file has 2 engines"):
        build_database(synthetic_raw_dir, db)
    assert not db.exists()


def test_command_line_entry_point(
    synthetic_raw_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    db = tmp_path / "cli.db"
    monkeypatch.setattr(
        sys, "argv", ["load_db", "--raw-dir", str(synthetic_raw_dir), "--db-path", str(db)]
    )
    load_db_module.main()
    assert "Wrote" in capsys.readouterr().out
    assert db.exists()


def test_failed_write_cleans_up_the_temp_file(
    synthetic_raw_dir: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr("pandas.DataFrame.to_sql", boom)
    db = tmp_path / "t.db"
    with pytest.raises(RuntimeError, match="disk full"):
        build_database(synthetic_raw_dir, db)
    assert not db.exists()
    assert not (tmp_path / "t.db.tmp").exists()
