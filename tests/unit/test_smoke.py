from app.main import health_status


def test_health_status_payload() -> None:
    assert health_status() == {"status": "ok"}
