"""The dashboard works without internet access: no page refers to an external URL (NFR-8)."""

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app
from tests.conftest import Artifacts

STATIC = Path(__file__).resolve().parents[2] / "app" / "static"
PLOTLY = STATIC / "vendor" / "plotly-basic-2.27.0.min.js"


@pytest.fixture
def client(artifacts: Artifacts) -> Iterator[TestClient]:
    settings = Settings(artifacts.db_path, artifacts.model_path)
    app.dependency_overrides[get_settings] = lambda: settings
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("path", ["/", "/engines/1"])
def test_page_has_no_external_urls(client: TestClient, path: str) -> None:
    html = client.get(path).text
    assert re.findall(r"https?://", html) == []


@pytest.mark.parametrize("path", ["/", "/engines/1"])
def test_every_local_asset_a_page_loads_is_served(client: TestClient, path: str) -> None:
    html = client.get(path).text
    urls = re.findall(r'(?:src|href)="(/static/[^"]+)"', html)
    assert "/static/app.css" in urls
    assert "/static/vendor/plotly-basic-2.27.0.min.js" in urls
    for url in urls:
        response = client.get(url)
        assert response.status_code == 200, url
        assert len(response.content) > 0, url


def test_vendored_plotly_matches_the_recorded_checksum() -> None:
    readme = (STATIC / "vendor" / "README.md").read_text()
    match = re.search(r"File sha256: `([0-9a-f]{64})`", readme)
    assert match is not None
    assert hashlib.sha256(PLOTLY.read_bytes()).hexdigest() == match.group(1)


def test_vendored_plotly_keeps_its_license_notice() -> None:
    assert "Licensed under the MIT license" in PLOTLY.read_text(encoding="utf-8")[:400]
    assert "MIT License" in (STATIC / "vendor" / "PLOTLY-LICENSE.txt").read_text()
