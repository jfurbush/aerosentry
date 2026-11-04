import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from pipeline.load_db import build_database
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


def _assert_error_shape(body: Any) -> str:
    assert set(body) == {"detail"}
    assert isinstance(body["detail"], str)
    return body["detail"]


# --- GET /engines (FR-6) ---------------------------------------------------------------


def test_engines_lists_ids_and_cycle_counts(client: TestClient) -> None:
    response = client.get("/engines")
    assert response.status_code == 200
    assert response.json() == [
        {"engine_id": 1, "n_cycles": 120},
        {"engine_id": 2, "n_cycles": 150},
        {"engine_id": 3, "n_cycles": 100},
    ]


# --- POST /predict (FR-7) --------------------------------------------------------------


@pytest.mark.parametrize(("engine_id", "cycles"), [(1, 120), (2, 150), (3, 100)])
def test_predict_returns_id_latest_cycle_and_rul(
    client: TestClient, engine_id: int, cycles: int
) -> None:
    response = client.post("/predict", json={"engine_id": engine_id})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"engine_id", "cycle", "predicted_rul"}
    assert body["engine_id"] == engine_id
    assert body["cycle"] == cycles
    assert 0.0 <= body["predicted_rul"] <= 125.0


def test_predict_is_repeatable(client: TestClient) -> None:
    first = client.post("/predict", json={"engine_id": 1}).json()
    second = client.post("/predict", json={"engine_id": 1}).json()
    assert first == second


# --- errors (FR-8) ---------------------------------------------------------------------


def test_unknown_engine_is_404_with_a_clear_message(client: TestClient) -> None:
    response = client.post("/predict", json={"engine_id": 999})
    assert response.status_code == 404
    assert "999" in _assert_error_shape(response.json())


@pytest.mark.parametrize(
    "body",
    [
        {},
        {"engine_id": "1"},
        {"engine_id": 1.5},
        {"engine_id": 1.0},
        {"engine_id": 0},
        {"engine_id": -3},
        {"engine_id": None},
        {"engine_id": True},
        {"engine_id": 10**30},
        {"engine_id": 1, "extra": 2},
        [],
        "engine 1",
    ],
)
def test_malformed_bodies_are_422_with_a_clear_message(client: TestClient, body: Any) -> None:
    response = client.post("/predict", json=body)
    assert response.status_code == 422
    assert _assert_error_shape(response.json()).startswith("Malformed request")


def test_body_that_is_not_json_is_422(client: TestClient) -> None:
    response = client.post(
        "/predict", content="engine_id=1", headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert "not valid JSON" in _assert_error_shape(response.json())


def test_missing_body_is_422(client: TestClient) -> None:
    response = client.post("/predict")
    assert response.status_code == 422
    _assert_error_shape(response.json())


def test_predict_only_accepts_post(client: TestClient) -> None:
    assert client.get("/predict").status_code == 405


# --- unavailable pieces (UC-2 exception flows) -----------------------------------------


def test_missing_model_gives_503_logs_it_and_hides_the_path(
    client_without_model: TestClient, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.ERROR, logger="aerosentry"):
        response = client_without_model.post("/predict", json={"engine_id": 1})
    assert response.status_code == 503
    detail = _assert_error_shape(response.json())
    assert "model" in detail
    assert "missing.joblib" not in detail
    assert any("missing.joblib" in record.getMessage() for record in caplog.records)


def test_engine_list_still_works_without_the_model(client_without_model: TestClient) -> None:
    assert client_without_model.get("/engines").status_code == 200


def test_unknown_engine_is_still_404_without_the_model(client_without_model: TestClient) -> None:
    assert client_without_model.post("/predict", json={"engine_id": 999}).status_code == 404


@pytest.mark.parametrize(("method", "path"), [("get", "/engines"), ("post", "/predict")])
def test_missing_database_gives_503(client_without_db: TestClient, method: str, path: str) -> None:
    kwargs = {"json": {"engine_id": 1}} if method == "post" else {}
    response = getattr(client_without_db, method)(path, **kwargs)
    assert response.status_code == 503
    assert "database" in _assert_error_shape(response.json())


def test_model_and_database_that_do_not_match_give_500(
    client: TestClient, artifacts: Artifacts, synthetic_raw_dir: Path
) -> None:
    # Rebuild the database with different window sizes than the model was trained with.
    build_database(synthetic_raw_dir, artifacts.db_path, windows=(3,))
    response = client.post("/predict", json={"engine_id": 1})
    assert response.status_code == 500
    assert "do not match" in _assert_error_shape(response.json())


# --- the service never changes data (UC-1, UC-2 postconditions) ------------------------


def test_requests_do_not_change_the_database(client: TestClient, artifacts: Artifacts) -> None:
    before = artifacts.db_path.read_bytes()
    client.get("/engines")
    client.post("/predict", json={"engine_id": 1})
    client.post("/predict", json={"engine_id": 999})
    assert artifacts.db_path.read_bytes() == before


def test_openapi_documents_the_error_responses(client: TestClient) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert {"404", "422", "503"} <= set(paths["/predict"]["post"]["responses"])
