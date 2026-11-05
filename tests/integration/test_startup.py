"""The model is loaded when the app starts, so the first prediction is not slow (NFR-2)."""

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import _cached_model, app
from tests.conftest import Artifacts


@pytest.fixture(autouse=True)
def _fresh_model_cache() -> Iterator[None]:
    _cached_model.cache_clear()
    yield
    _cached_model.cache_clear()
    app.dependency_overrides.clear()


def _use(monkeypatch: pytest.MonkeyPatch, settings: Settings) -> None:
    # The startup hook reads settings directly, so point it at the test files by environment.
    monkeypatch.setenv("AEROSENTRY_DB", str(settings.db_path))
    monkeypatch.setenv("AEROSENTRY_MODEL", str(settings.model_path))
    app.dependency_overrides[get_settings] = lambda: settings


def test_model_is_loaded_before_the_first_request(
    artifacts: Artifacts, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(monkeypatch, Settings(artifacts.db_path, artifacts.model_path))
    assert _cached_model.cache_info().currsize == 0
    with TestClient(app) as client:
        assert _cached_model.cache_info().currsize == 1  # loaded by startup, no request yet
        assert _cached_model.cache_info().misses == 1
        assert client.post("/predict", json={"engine_id": 1}).status_code == 200
        assert _cached_model.cache_info().misses == 1  # the request reused the loaded model
        assert _cached_model.cache_info().hits >= 1


def test_missing_model_does_not_stop_startup(
    artifacts: Artifacts,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _use(monkeypatch, Settings(artifacts.db_path, tmp_path / "missing.joblib"))
    with caplog.at_level(logging.WARNING, logger="aerosentry"):
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/engines").status_code == 200
            response = client.post("/predict", json={"engine_id": 1})
    assert response.status_code == 503
    assert "model" in response.json()["detail"].lower()
    assert "Model not loaded at startup" in caplog.text


def test_unreadable_model_does_not_stop_startup(
    artifacts: Artifacts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not a model")
    _use(monkeypatch, Settings(artifacts.db_path, bad))
    with TestClient(app) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/predict", json={"engine_id": 1}).status_code == 503


def test_a_model_fixed_after_startup_is_picked_up(
    artifacts: Artifacts, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "model.joblib"
    _use(monkeypatch, Settings(artifacts.db_path, target))
    with TestClient(app) as client:
        assert client.post("/predict", json={"engine_id": 1}).status_code == 503
        target.write_bytes(artifacts.model_path.read_bytes())
        assert client.post("/predict", json={"engine_id": 1}).status_code == 200
