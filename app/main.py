"""AeroSentry FastAPI application.

Scaffold only. /engines (FR-6), /predict (FR-7, FR-8) and the dashboard (FR-9 to FR-12)
are not implemented yet. See docs/changes-log.md for status.
"""

from fastapi import FastAPI

app = FastAPI(title="AeroSentry", version="0.1.0")


def health_status() -> dict[str, str]:
    """Return a simple liveness payload."""
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by smoke tests and container health probes."""
    return health_status()
