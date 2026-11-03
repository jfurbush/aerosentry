"""Load NASA CMAPSS FD001 files and build the clipped RUL target (FR-1, FR-2)."""

from pathlib import Path

import pandas as pd

RUL_CLIP = 125

INDEX_COLUMNS = ["engine_id", "cycle"]
SETTING_COLUMNS = [f"setting_{i}" for i in (1, 2, 3)]
SENSOR_COLUMNS = [f"s{i}" for i in range(1, 22)]
ALL_COLUMNS = INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS

# In FD001 these sensors never change (s6 only takes two values), so they carry no signal.
CONSTANT_SENSORS = ["s1", "s5", "s6", "s10", "s16", "s18", "s19"]
INFORMATIVE_SENSORS = [s for s in SENSOR_COLUMNS if s not in CONSTANT_SENSORS]


def load_cmapss(path: Path) -> pd.DataFrame:
    """Read a CMAPSS train or test file into a frame sorted by engine and cycle.

    Raises ValueError if the file does not have the 26 expected columns.
    """
    raw = pd.read_csv(path, sep=r"\s+", header=None)
    raw = raw.dropna(axis=1, how="all")  # some copies of the files end lines with a space
    if raw.shape[1] != len(ALL_COLUMNS):
        raise ValueError(f"{path} has {raw.shape[1]} columns, expected {len(ALL_COLUMNS)}")
    raw.columns = ALL_COLUMNS
    raw = raw.astype({"engine_id": "int64", "cycle": "int64"})
    return raw.sort_values(INDEX_COLUMNS).reset_index(drop=True)


def add_rul_target(df: pd.DataFrame, clip: int = RUL_CLIP) -> pd.DataFrame:
    """Add a `rul` column: cycles remaining until each engine's last cycle, capped at `clip`.

    Only valid for run-to-failure data (the FD001 training file).
    """
    out = df.copy()
    last_cycle = out.groupby("engine_id")["cycle"].transform("max")
    out["rul"] = (last_cycle - out["cycle"]).clip(upper=clip)
    return out


def load_test_rul(path: Path) -> pd.Series:
    """Read RUL_FD001.txt: true RUL at the last recorded cycle of each test engine.

    The file has one value per line, in engine order, so the index is engine IDs 1..N.
    """
    values = pd.read_csv(path, header=None).iloc[:, 0].astype("int64")
    values.index = pd.RangeIndex(1, len(values) + 1, name="engine_id")
    values.name = "actual_rul"
    return values
