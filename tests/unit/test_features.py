import math

import pandas as pd
import pytest

from pipeline.features import DEFAULT_WINDOWS, add_rolling_features, feature_columns


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 2, 2],
            "cycle": [1, 2, 3, 4, 1, 2],
            "s2": [10.0, 20.0, 30.0, 40.0, 1000.0, 2000.0],
        }
    )


def test_rolling_mean_and_std_match_hand_calculation() -> None:
    out = add_rolling_features(_frame(), sensors=["s2"], windows=[3])
    engine_1 = out[out["engine_id"] == 1]
    assert engine_1["s2_mean3"].tolist() == [10.0, 15.0, 20.0, 30.0]
    # std of [10, 20, 30] with ddof=1 is 10
    assert engine_1["s2_std3"].iloc[2] == pytest.approx(10.0)
    assert engine_1["s2_std3"].iloc[3] == pytest.approx(10.0)


def test_first_cycle_std_is_zero_not_nan() -> None:
    out = add_rolling_features(_frame(), sensors=["s2"], windows=[3])
    first_rows = out[out["cycle"] == 1]
    assert (first_rows["s2_std3"] == 0.0).all()
    assert not out["s2_std3"].isna().any()


def test_windows_do_not_cross_engines() -> None:
    out = add_rolling_features(_frame(), sensors=["s2"], windows=[3])
    engine_2 = out[out["engine_id"] == 2]
    assert engine_2["s2_mean3"].tolist() == [1000.0, 1500.0]


def test_features_do_not_look_ahead() -> None:
    base = add_rolling_features(_frame(), sensors=["s2"], windows=[3])
    changed_frame = _frame()
    changed_frame.loc[3, "s2"] = 9999.0  # change engine 1, cycle 4 only
    changed = add_rolling_features(changed_frame, sensors=["s2"], windows=[3])
    earlier = base["cycle"] < 4
    assert base.loc[earlier, "s2_mean3"].tolist() == changed.loc[earlier, "s2_mean3"].tolist()


def test_multiple_windows_add_columns_for_each() -> None:
    out = add_rolling_features(_frame(), sensors=["s2"], windows=[2, 3])
    for name in ["s2_mean2", "s2_std2", "s2_mean3", "s2_std3"]:
        assert name in out.columns


def test_input_is_not_modified() -> None:
    frame = _frame()
    add_rolling_features(frame, sensors=["s2"], windows=[3])
    assert list(frame.columns) == ["engine_id", "cycle", "s2"]


def test_unsorted_input_is_sorted_first() -> None:
    shuffled = _frame().sample(frac=1.0, random_state=1)
    out = add_rolling_features(shuffled, sensors=["s2"], windows=[3])
    assert out["s2_mean3"].tolist()[:4] == [10.0, 15.0, 20.0, 30.0]


def test_window_below_two_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        add_rolling_features(_frame(), sensors=["s2"], windows=[1])


def test_unknown_sensor_is_rejected() -> None:
    with pytest.raises(ValueError, match="missing sensor"):
        add_rolling_features(_frame(), sensors=["s99"], windows=[3])


def test_feature_columns_order_and_count() -> None:
    names = feature_columns(sensors=["s2", "s3"], windows=[5, 10])
    assert names == [
        "s2", "s3",
        "s2_mean5", "s3_mean5", "s2_std5", "s3_std5",
        "s2_mean10", "s3_mean10", "s2_std10", "s3_std10",
    ]  # fmt: skip
    assert len(feature_columns()) == 14 * (1 + 2 * len(DEFAULT_WINDOWS))


def test_output_contains_every_feature_column() -> None:
    frame = _frame()
    frame["s3"] = frame["s2"] * 2
    out = add_rolling_features(frame, sensors=["s2", "s3"], windows=[5, 10])
    assert set(feature_columns(["s2", "s3"], [5, 10])) <= set(out.columns)
    assert not any(math.isnan(v) for v in out["s3_std10"])
