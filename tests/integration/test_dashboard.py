"""Dashboard routes (FR-9 to FR-12): engine list page, per-engine page and error pages."""

import html
import sqlite3
import statistics
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from app.model import PredictionError
from app.sensors import DEFAULT_SENSOR, SENSORS
from tests.conftest import Artifacts


def _client_for(settings: Settings) -> Iterator[TestClient]:
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def client(artifacts: Artifacts) -> Iterator[TestClient]:
    yield from _client_for(Settings(artifacts.db_path, artifacts.model_path))


@pytest.fixture
def client_without_model(artifacts: Artifacts, tmp_path: Path) -> Iterator[TestClient]:
    yield from _client_for(Settings(artifacts.db_path, tmp_path / "missing.joblib"))


@pytest.fixture
def client_without_db(artifacts: Artifacts, tmp_path: Path) -> Iterator[TestClient]:
    yield from _client_for(Settings(tmp_path / "missing.db", artifacts.model_path))


def _assert_error_page(response: Any, status_code: int, *expected_text: str) -> None:
    """An error is a readable HTML page with a link back to the engine list (FR-12)."""
    assert response.status_code == status_code
    assert "text/html" in response.headers["content-type"]
    assert f"Error {status_code}" in response.text
    assert 'href="/"' in response.text
    visible_text = html.unescape(response.text)  # what the reader sees, quotes included
    for text in expected_text:
        assert text in visible_text


# --- GET / (FR-9) ----------------------------------------------------------------------


def test_home_lists_every_engine_with_cycles_and_a_link(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    for engine_id, cycles in [(1, 120), (2, 150), (3, 100)]:
        assert f"Engine {engine_id}" in response.text
        assert f"{cycles} cycles" in response.text
        assert f'href="/engines/{engine_id}"' in response.text


def test_home_says_so_when_no_engines_are_loaded(artifacts: Artifacts, client: TestClient) -> None:
    with closing(sqlite3.connect(artifacts.db_path)) as conn:
        conn.execute("DELETE FROM engines")
        conn.commit()
    response = client.get("/")
    assert response.status_code == 200
    assert "No engines are loaded" in response.text
    assert "pipeline.load_db" in response.text
    assert "View Telemetry" not in response.text


def test_home_reports_unavailable_database(client_without_db: TestClient) -> None:
    response = client_without_db.get("/")
    _assert_error_page(response, 503, "engine database is unavailable")


# --- GET /engines/{id} (FR-10, FR-11) --------------------------------------------------


def test_detail_shows_the_same_rul_the_api_predicts(client: TestClient) -> None:
    predicted = client.post("/predict", json={"engine_id": 2}).json()["predicted_rul"]
    response = client.get("/engines/2")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Engine 2" in response.text
    assert f"{predicted:.1f}" in response.text
    assert "cycles" in response.text


def test_detail_explains_that_the_rul_is_capped(client: TestClient) -> None:
    response = client.get("/engines/1")
    cap = response.context["rul_cap"]  # type: ignore[attr-defined]
    assert f"capped at {cap}" in response.text
    assert f"about {cap} cycles or more" in response.text


def test_detail_has_no_made_up_rul_trajectory(client: TestClient) -> None:
    response = client.get("/engines/1")
    assert "Trajectory" not in response.text
    assert "rul_trajectory" not in response.context  # type: ignore[attr-defined]


def test_detail_offers_all_21_sensors_with_names_and_units(client: TestClient) -> None:
    response = client.get("/engines/1")
    for sensor in SENSORS:
        assert f'value="{sensor.column}"' in response.text
        assert sensor.label in response.text
    assert f'value="{DEFAULT_SENSOR}" selected' in response.text


def test_detail_chart_data_has_raw_readings_and_a_moving_average(
    artifacts: Artifacts, client: TestClient
) -> None:
    response = client.get("/engines/3")
    chart = response.context["chart"]  # type: ignore[attr-defined]
    assert chart["cycles"] == list(range(1, 101))
    assert chart["window"] == 10
    assert set(chart["sensors"]) == {s.column for s in SENSORS}

    with closing(sqlite3.connect(artifacts.db_path)) as conn:
        stored = [
            row[0]
            for row in conn.execute(
                "SELECT s11 FROM readings WHERE engine_id = 3 ORDER BY cycle"
            ).fetchall()
        ]
    sensor = chart["sensors"]["s11"]
    assert sensor["raw"] == stored  # raw values, not scaled
    assert len(sensor["average"]) == 100
    assert sensor["average"][:9] == [None] * 9
    assert sensor["average"][9] == pytest.approx(statistics.fmean(stored[:10]), abs=1e-4)
    assert sensor["average"][99] == pytest.approx(statistics.fmean(stored[90:100]), abs=1e-4)
    assert sensor["axis_title"] == "Ps30: Static pressure at HPC outlet (psia)"


def test_detail_marks_a_sensor_that_never_changes(artifacts: Artifacts, client: TestClient) -> None:
    # Several FD001 sensors never change for an engine. The page must say so.
    with closing(sqlite3.connect(artifacts.db_path)) as conn:
        conn.execute("UPDATE readings SET s1 = 518.67 WHERE engine_id = 1")
        conn.commit()
    response = client.get("/engines/1")
    assert response.status_code == 200
    sensors = response.context["chart"]["sensors"]  # type: ignore[attr-defined]
    assert sensors["s1"]["constant"] is True
    assert sensors["s1"]["constant_value"] == 518.67
    assert sensors["s11"]["constant"] is False
    assert sensors["s11"]["constant_value"] is None
    assert "constant for this engine" in response.text


def test_detail_without_readings_is_a_404_page(artifacts: Artifacts, client: TestClient) -> None:
    with closing(sqlite3.connect(artifacts.db_path)) as conn:
        conn.execute("DELETE FROM readings WHERE engine_id = 1")
        conn.commit()
    response = client.get("/engines/1")
    _assert_error_page(response, 404, "No telemetry found for Engine 1")


def test_detail_for_unknown_engine_is_a_404_page(client: TestClient) -> None:
    response = client.get("/engines/999")
    _assert_error_page(response, 404, "Engine 999 was not found")


def test_detail_rejects_a_non_integer_engine_id_with_a_page(client: TestClient) -> None:
    response = client.get("/engines/abc")
    _assert_error_page(response, 422, '"abc" is not a valid engine ID')


@pytest.mark.parametrize("engine_id", ["0", "-1", "2147483648", "99999999999999999999"])
def test_detail_rejects_engine_ids_outside_the_valid_range(
    client: TestClient, engine_id: str
) -> None:
    response = client.get(f"/engines/{engine_id}")
    _assert_error_page(response, 422, "is not a valid engine ID")


def test_a_very_long_engine_id_is_shortened_on_the_error_page(client: TestClient) -> None:
    response = client.get("/engines/" + "a" * 100)
    _assert_error_page(response, 422, "a" * 40 + "...")
    assert "a" * 41 not in response.text


def test_the_entered_engine_id_cannot_inject_markup(client: TestClient) -> None:
    response = client.get("/engines/%3Cb%3Ex")
    _assert_error_page(response, 422)
    assert "<b>x" not in response.text
    assert "&lt;b&gt;x" in response.text


def test_detail_reports_unavailable_database_with_a_page(client_without_db: TestClient) -> None:
    response = client_without_db.get("/engines/1")
    _assert_error_page(response, 503, "engine database is unavailable")


def test_detail_reports_unavailable_model_with_a_page(client_without_model: TestClient) -> None:
    response = client_without_model.get("/engines/1")
    _assert_error_page(response, 503, "prediction model is unavailable")


def test_detail_reports_a_model_database_mismatch_with_a_page(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object, **__: object) -> float:
        raise PredictionError("columns do not match")

    monkeypatch.setattr("app.main.predict_rul", fail)
    response = client.get("/engines/1")
    _assert_error_page(response, 500, "model and the database do not match")


# --- The JSON API keeps answering errors in JSON ----------------------------------------


def test_the_engine_list_api_still_answers_errors_in_json(client_without_db: TestClient) -> None:
    response = client_without_db.get("/engines")
    assert response.status_code == 503
    assert response.headers["content-type"].startswith("application/json")
    assert set(response.json()) == {"detail"}


def test_predict_still_answers_errors_in_json(client: TestClient) -> None:
    unknown = client.post("/predict", json={"engine_id": 999})
    assert unknown.status_code == 404
    assert unknown.headers["content-type"].startswith("application/json")
    assert set(unknown.json()) == {"detail"}

    malformed = client.post("/predict", json={"engine_id": "x"})
    assert malformed.status_code == 422
    assert malformed.headers["content-type"].startswith("application/json")
    assert "Send JSON like" in malformed.json()["detail"]
