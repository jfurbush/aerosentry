"""Sensor names, units and the moving average used on the dashboard chart."""

import statistics

import pytest

from app.db import SENSOR_COLUMNS
from app.sensors import DEFAULT_SENSOR, MOVING_AVERAGE_WINDOW, NO_UNIT, SENSORS, moving_average

# Table 2 of Saxena et al. (PHM 2008), "C-MAPSS outputs to measure system response".
PAPER_TABLE = {
    1: ("T2", "°R"),
    2: ("T24", "°R"),
    3: ("T30", "°R"),
    4: ("T50", "°R"),
    5: ("P2", "psia"),
    6: ("P15", "psia"),
    7: ("P30", "psia"),
    8: ("Nf", "rpm"),
    9: ("Nc", "rpm"),
    10: ("epr", NO_UNIT),
    11: ("Ps30", "psia"),
    12: ("phi", "pps/psi"),
    13: ("NRf", "rpm"),
    14: ("NRc", "rpm"),
    15: ("BPR", NO_UNIT),
    16: ("farB", NO_UNIT),
    17: ("htBleed", NO_UNIT),
    18: ("Nf_dmd", "rpm"),
    19: ("PCNfR_dmd", "rpm"),
    20: ("W31", "lbm/s"),
    21: ("W32", "lbm/s"),
}


def test_sensors_cover_every_database_column_in_order() -> None:
    assert [s.column for s in SENSORS] == list(SENSOR_COLUMNS)
    assert [s.number for s in SENSORS] == list(range(1, 22))


def test_symbols_and_units_match_the_published_table() -> None:
    assert {s.number: (s.symbol, s.unit) for s in SENSORS} == PAPER_TABLE


def test_every_sensor_has_a_description() -> None:
    assert all(s.description.strip() for s in SENSORS)


def test_default_sensor_exists() -> None:
    assert DEFAULT_SENSOR in SENSOR_COLUMNS


def test_labels_name_the_sensor_and_its_unit() -> None:
    sensor = next(s for s in SENSORS if s.column == "s11")
    assert sensor.label == "Sensor 11: Ps30 (Static pressure at HPC outlet, psia)"
    assert sensor.axis_title == "Ps30: Static pressure at HPC outlet (psia)"
    assert sensor.hover_unit == " psia"


def test_a_quantity_with_no_unit_adds_nothing_to_a_tooltip() -> None:
    sensor = next(s for s in SENSORS if s.column == "s15")
    assert sensor.unit == NO_UNIT
    assert sensor.hover_unit == ""
    assert sensor.axis_title.endswith("(no unit)")


def test_moving_average_starts_once_a_full_window_exists() -> None:
    values = [float(v) for v in range(1, 13)]
    averaged = moving_average(values, window=10)
    assert averaged[:9] == [None] * 9
    assert averaged[9] == pytest.approx(statistics.fmean(values[0:10]))
    assert averaged[10] == pytest.approx(statistics.fmean(values[1:11]))
    assert averaged[11] == pytest.approx(statistics.fmean(values[2:12]))
    assert len(averaged) == len(values)


def test_moving_average_of_a_constant_is_that_constant() -> None:
    assert moving_average([518.67] * 15)[-1] == pytest.approx(518.67)


def test_moving_average_with_fewer_readings_than_the_window_is_all_none() -> None:
    assert moving_average([1.0, 2.0, 3.0], window=10) == [None, None, None]
    assert moving_average([], window=10) == []


def test_moving_average_with_window_one_returns_the_values() -> None:
    assert moving_average([1.0, 2.5, 4.0], window=1) == [1.0, 2.5, 4.0]


def test_moving_average_rejects_a_window_below_one() -> None:
    with pytest.raises(ValueError, match="window"):
        moving_average([1.0], window=0)


def test_default_window_is_ten_cycles() -> None:
    assert MOVING_AVERAGE_WINDOW == 10
    assert moving_average([1.0] * 10)[8] is None
    assert moving_average([1.0] * 10)[9] == 1.0
