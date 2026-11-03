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

    Early cycles use the cycles available so far (min_periods=1). An engine's first cycle
    has only one reading, so its std is defined as 0. A std that is undefined for any other
    reason, such as a missing sensor reading, stays NaN so the gap is visible.
    """
    if any(w < 2 for w in windows):
        raise ValueError("every window must be at least 2 cycles")
    if len(set(windows)) != len(windows):
        raise ValueError("window sizes must not repeat")
    missing = [s for s in sensors if s not in df.columns]
    if missing:
        raise ValueError(f"missing sensor columns: {missing}")
    new_names = feature_columns(sensors, windows)[len(sensors) :]
    already_there = [name for name in new_names if name in df.columns]
    if already_there:
        raise ValueError(f"feature columns already exist: {already_there}")

    out = df.sort_values(["engine_id", "cycle"]).reset_index(drop=True)
    first_cycle = out.groupby("engine_id").cumcount() == 0
    grouped = out.groupby("engine_id")[list(sensors)]
    parts = [out]
    for window in windows:
        rolling = grouped.rolling(window, min_periods=1)
        mean = rolling.mean().reset_index(level=0, drop=True)
        std = rolling.std().reset_index(level=0, drop=True)
        std[first_cycle] = 0.0
        parts.append(mean.add_suffix(f"_mean{window}"))
        parts.append(std.add_suffix(f"_std{window}"))
    return pd.concat(parts, axis=1)
