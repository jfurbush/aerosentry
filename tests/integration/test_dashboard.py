"""Dashboard routes (FR-9 to FR-12): engine list page and per-engine detail page."""

import sqlite3
from collections.abc import Iterator
from contextlib import closing
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.db import SENSOR_COLUMNS
from app.main import app
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


# --- GET / (FR-9) ----------------------------------------------------------------------


def test_home_lists_every_engine_with_cycles_and_a_link(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    for engine_id, cycles in [(1, 120), (2, 150), (3, 100)]:
        assert f"Engine {engine_id}" in response.text
        assert f"{cycles} cycles" in response.text
        assert f'href="/engines/{engine_id}"' in response.text


def test_home_reports_unavailable_database(client_without_db: TestClient) -> None:
    response = client_without_db.get("/")
    assert response.status_code == 503
    assert set(response.json()) == {"detail"}


# --- GET /engines/{id} (FR-10, FR-11) --------------------------------------------------


def test_detail_shows_the_same_rul_the_api_predicts(client: TestClient) -> None:
    predicted = client.post("/predict", json={"engine_id": 2}).json()["predicted_rul"]
    response = client.get("/engines/2")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Engine 2" in response.text
    assert f"{predicted:.1f}" in response.text
    assert "cycles" in response.text


def test_detail_offers_all_21_sensors_to_plot(client: TestClient) -> None:
    response = client.get("/engines/1")
    for sensor in SENSOR_COLUMNS:
        assert f'value="{sensor}"' in response.text


def test_detail_chart_data_covers_every_cycle_and_is_scaled_0_to_1(client: TestClient) -> None:
    response = client.get("/engines/3")
    context = response.context  # type: ignore[attr-defined]
    assert context["cycles"] == list(range(1, 101))
    assert set(context["sensors_data"]) == set(SENSOR_COLUMNS)
    for values in context["sensors_data"].values():
        assert len(values) == 100
        assert min(values) >= 0.0
        assert max(values) <= 1.0


def test_detail_trajectory_ends_at_the_current_prediction(client: TestClient) -> None:
    response = client.get("/engines/1")
    context = response.context  # type: ignore[attr-defined]
    assert context["rul_trajectory"][-1] == pytest.approx(context["predicted_rul"], abs=0.01)
    assert all(value >= 0.0 for value in context["rul_trajectory"])


def test_detail_scales_a_constant_sensor_to_zero(artifacts: Artifacts, client: TestClient) -> None:
    # Several FD001 sensors never change. Min-Max scaling must not divide by zero.
    with closing(sqlite3.connect(artifacts.db_path)) as conn:
        conn.execute("UPDATE readings SET s1 = 518.67 WHERE engine_id = 1")
        conn.commit()
    response = client.get("/engines/1")
    assert response.status_code == 200
    assert set(response.context["sensors_data"]["s1"]) == {0.0}  # type: ignore[attr-defined]


def test_detail_without_readings_is_404(artifacts: Artifacts, client: TestClient) -> None:
    with closing(sqlite3.connect(artifacts.db_path)) as conn:
        conn.execute("DELETE FROM readings WHERE engine_id = 1")
        conn.commit()
    response = client.get("/engines/1")
    assert response.status_code == 404
    assert "telemetry" in response.json()["detail"]


def test_detail_for_unknown_engine_is_404(client: TestClient) -> None:
    response = client.get("/engines/999")
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_detail_rejects_a_non_integer_engine_id(client: TestClient) -> None:
    response = client.get("/engines/abc")
    assert response.status_code == 422
    assert set(response.json()) == {"detail"}


def test_detail_reports_unavailable_database(client_without_db: TestClient) -> None:
    response = client_without_db.get("/engines/1")
    assert response.status_code == 503


def test_detail_reports_unavailable_model(client_without_model: TestClient) -> None:
    response = client_without_model.get("/engines/1")
    assert response.status_code == 503
