"""AeroSentry FastAPI application: engine list, RUL prediction, and dashboard (FR-6 to FR-12)."""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db
from app.config import Settings, get_settings
from app.model import ModelUnavailable, PredictionError, RulModel, load_model, predict_rul
from app.schemas import EngineSummary, ErrorResponse, PredictRequest, PredictResponse

logger = logging.getLogger("aerosentry")

app = FastAPI(
    title="AeroSentry",
    version="0.1.0",
    description=(
        "Estimates Remaining Useful Life (RUL), in operating cycles, for turbofan engines "
        "in the simulated NASA CMAPSS FD001 dataset. Every error response has the shape "
        '{"detail": "<message>"}.'
    ),
)

# ------------------------------------------------------------------------------
# Dashboard Setup (FR-9 to FR-12) with Absolute Path Resolution
# ------------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

SettingsDep = Annotated[Settings, Depends(get_settings)]

_UNAVAILABLE: dict[int | str, dict[str, Any]] = {
    503: {"model": ErrorResponse, "description": "Database or model unavailable"}
}


@lru_cache(maxsize=2)
def _cached_model(path: str) -> RulModel:
    """Load the model once per path. Failures are not cached, so a fixed file is picked up."""
    return load_model(Path(path))


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": message})


def _describe(problem: Any) -> str:
    if problem["type"] == "json_invalid":
        return "the request body is not valid JSON"
    where = ".".join(str(part) for part in problem["loc"] if part != "body")
    return f"{where or 'request body'}: {problem['msg']}"


@app.exception_handler(RequestValidationError)
async def malformed_request(request: Request, exc: RequestValidationError) -> JSONResponse:
    problems = "; ".join(_describe(p) for p in exc.errors())
    return _error(422, f'Malformed request: {problems}. Send JSON like {{"engine_id": 1}}.')


@app.exception_handler(db.DatabaseUnavailable)
async def database_unavailable(request: Request, exc: db.DatabaseUnavailable) -> JSONResponse:
    logger.error("Database unavailable: %s", exc)
    return _error(503, "The engine database is unavailable. Try again later.")


@app.exception_handler(ModelUnavailable)
async def model_unavailable(request: Request, exc: ModelUnavailable) -> JSONResponse:
    logger.error("Model unavailable: %s", exc)
    return _error(503, "The prediction model is unavailable. Try again later.")


@app.exception_handler(PredictionError)
async def prediction_failed(request: Request, exc: PredictionError) -> JSONResponse:
    logger.error("Prediction failed: %s", exc)
    return _error(
        500,
        "The prediction failed because the model and the database do not match. "
        "Rebuild both with the pipeline commands in the README.",
    )


def health_status() -> dict[str, str]:
    """Return a simple liveness payload."""
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness check used by smoke tests and container health probes."""
    return health_status()


@app.get("/engines", response_model=list[EngineSummary], responses=_UNAVAILABLE)
def engines(settings: SettingsDep) -> list[EngineSummary]:
    """List the available engines and how many cycles are recorded for each."""
    rows = db.list_engines(settings.db_path)
    return [EngineSummary(engine_id=r.engine_id, n_cycles=r.n_cycles) for r in rows]


@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Unknown engine ID"},
        422: {"model": ErrorResponse, "description": "Malformed request"},
        **_UNAVAILABLE,
    },
)
def predict(body: PredictRequest, settings: SettingsDep) -> PredictResponse:
    """Predict remaining useful life for an engine's latest recorded cycle."""
    latest = db.latest_features(settings.db_path, body.engine_id)
    if latest is None:
        raise HTTPException(status_code=404, detail=f"Engine {body.engine_id} not found")
    model = _cached_model(str(settings.model_path))
    rul = predict_rul(model, latest.values)
    return PredictResponse(engine_id=latest.engine_id, cycle=latest.cycle, predicted_rul=rul)


@app.get("/")
def dashboard_home(request: Request, settings: SettingsDep) -> Response:
    """Render the dashboard overview listing all engines."""
    rows = db.list_engines(settings.db_path)
    engine_list = [EngineSummary(engine_id=r.engine_id, n_cycles=r.n_cycles) for r in rows]
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "engines": engine_list},
    )


@app.get("/engines/{engine_id}")
def engine_detail(engine_id: int, request: Request, settings: SettingsDep) -> Response:
    """Render detailed view with RUL model trajectory and normalized sensor telemetry."""
    latest = db.latest_features(settings.db_path, engine_id)
    if latest is None:
        raise HTTPException(status_code=404, detail=f"Engine {engine_id} not found")

    model = _cached_model(str(settings.model_path))
    current_rul = predict_rul(model, latest.values)

    history = db.get_engine_history(Path(settings.db_path), engine_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No telemetry found for Engine {engine_id}")

    cycles = [h["cycle"] for h in history]
    sensor_cols = [f"s{i}" for i in range(1, 22)]

    # Compute data-driven trajectory anchored to the current model prediction
    rul_trajectory = [
        round(max(0.0, float(current_rul + (latest.cycle - h["cycle"]))), 2)
        for h in history
    ]

    # Apply per-sensor Min-Max normalization [0, 1] for multi-sensor display scaling
    normalized_sensors = {}
    raw_sensors = {}
    for col in sensor_cols:
        vals = [h[col] for h in history]
        raw_sensors[col] = vals
        v_min, v_max = min(vals), max(vals)
        if v_max > v_min:
            normalized_sensors[col] = [round((v - v_min) / (v_max - v_min), 4) for v in vals]
        else:
            normalized_sensors[col] = [0.0 for _ in vals]

    return templates.TemplateResponse(
        "engine.html",
        {
            "request": request,
            "engine_id": engine_id,
            "cycle": latest.cycle,
            "predicted_rul": current_rul,
            "cycles": cycles,
            "sensors_data": normalized_sensors,
            "raw_sensors": raw_sensors,
            "rul_trajectory": rul_trajectory,
        },
    )
