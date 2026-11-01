"""Rolling window features per sensor (FR-3).

Windows only look backward, within one engine, so a row never sees a later cycle.
"""

from collections.abc import Sequence

import pandas as pd

from pipeline.data import INFORMATIVE_SENSORS

DEFAULT_WINDOWS: tuple[int, ...] = (5, 10)


def feature_columns(
    sensors: Sequence[str] = INFORMATIVE_SENSORS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> list[str]:
    """Model input column names: raw sensors, then a mean and std column per window."""
    names = list(sensors)
    for window in windows:
        names += [f"{s}_mean{window}" for s in sensors]
        names += [f"{s}_std{window}" for s in sensors]
    return names


def add_rolling_features(
    df: pd.DataFrame,
    sensors: Sequence[str] = INFORMATIVE_SENSORS,
    windows: Sequence[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """Return a copy of `df` with rolling mean and std columns added for each sensor and window.

    Early cycles use the cycles available so far (min_periods=1). The std of a single
    reading is undefined, so it is set to 0.
    """
    if any(w < 2 for w in windows):
        raise ValueError("every window must be at least 2 cycles")
    missing = [s for s in sensors if s not in df.columns]
    if missing:
        raise ValueError(f"missing sensor columns: {missing}")

    out = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    grouped = out.groupby("engine_id")[list(sensors)]
    parts = [out]
    for window in windows:
        rolling = grouped.rolling(window, min_periods=1)
        mean = rolling.mean().reset_index(level=0, drop=True)
        std = rolling.std().reset_index(level=0, drop=True).fillna(0.0)
        parts.append(mean.add_suffix(f"_mean{window}"))
        parts.append(std.add_suffix(f"_std{window}"))
    return pd.concat(parts, axis=1)
