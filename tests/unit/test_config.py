from pathlib import Path

import pytest

from app.config import get_settings


def test_defaults_follow_the_repository_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AEROSENTRY_DB", raising=False)
    monkeypatch.delenv("AEROSENTRY_MODEL", raising=False)
    settings = get_settings()
    assert settings.db_path.parts[-2:] == ("data", "aerosentry.db")
    assert settings.model_path.parts[-2:] == ("models", "rul_model.joblib")


def test_environment_variables_override_the_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("AEROSENTRY_DB", str(tmp_path / "x.db"))
    monkeypatch.setenv("AEROSENTRY_MODEL", str(tmp_path / "x.joblib"))
    settings = get_settings()
    assert settings.db_path == tmp_path / "x.db"
    assert settings.model_path == tmp_path / "x.joblib"
