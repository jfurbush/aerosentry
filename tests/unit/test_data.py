from pathlib import Path

import pandas as pd
import pytest

from pipeline.data import (
    ALL_COLUMNS,
    INFORMATIVE_SENSORS,
    RUL_CLIP,
    add_rul_target,
    load_cmapss,
    load_test_rul,
)


def test_load_cmapss_returns_named_columns_sorted(synthetic_raw_dir: Path) -> None:
    df = load_cmapss(synthetic_raw_dir / "train_FD001.txt")
    assert list(df.columns) == ALL_COLUMNS
    assert df["engine_id"].nunique() == 6
    assert df.equals(df.sort_values(["engine_id", "cycle"]).reset_index(drop=True))


def test_load_cmapss_rejects_wrong_column_count(tmp_path: Path) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("1 1 0.1 0.2\n1 2 0.1 0.2\n")
    with pytest.raises(ValueError, match="expected 26"):
        load_cmapss(bad)


def test_load_cmapss_ignores_trailing_space(tmp_path: Path) -> None:
    line = " ".join(["1", "1"] + ["0.5"] * 24) + " \n"
    path = tmp_path / "trailing.txt"
    path.write_text(line)
    assert load_cmapss(path).shape == (1, 26)


def test_rul_counts_down_to_zero_at_last_cycle(synthetic_raw_dir: Path) -> None:
    df = add_rul_target(load_cmapss(synthetic_raw_dir / "train_FD001.txt"))
    last = df.groupby("engine_id").tail(1)
    assert (last["rul"] == 0).all()


def test_rul_is_clipped_at_125(synthetic_raw_dir: Path) -> None:
    df = add_rul_target(load_cmapss(synthetic_raw_dir / "train_FD001.txt"))
    assert df["rul"].max() == RUL_CLIP
    engine_1 = df[df["engine_id"] == 1]  # 160 cycles, so cycle 1 has 159 left before clipping
    assert engine_1.iloc[0]["rul"] == RUL_CLIP
    assert engine_1.iloc[-30]["rul"] == 29  # below the clip, so unchanged


def test_rul_target_does_not_modify_input(synthetic_raw_dir: Path) -> None:
    df = load_cmapss(synthetic_raw_dir / "train_FD001.txt")
    add_rul_target(df)
    assert "rul" not in df.columns


def test_load_test_rul_indexes_by_engine_id(synthetic_raw_dir: Path) -> None:
    rul = load_test_rul(synthetic_raw_dir / "RUL_FD001.txt")
    assert rul.to_dict() == {1: 60, 2: 50, 3: 60}
    assert isinstance(rul, pd.Series)


def test_informative_sensors_drop_the_constant_ones() -> None:
    assert len(INFORMATIVE_SENSORS) == 14
    assert "s1" not in INFORMATIVE_SENSORS
    assert "s2" in INFORMATIVE_SENSORS
