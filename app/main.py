"""AeroSentry FastAPI application: engine list, RUL prediction, and dashboard (FR-6 to FR-12)."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi import Path as PathParam
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app import db
from app.config import Settings, get_settings
from app.model import (
    ModelUnavailable,
    PredictionError,
    RulModel,
    load_model,
    predict_rul,
    predict_rul_series,
)
from app.schemas import (
    MAX_ENGINE_ID,
    EngineSummary,
    ErrorResponse,
    PredictRequest,
    PredictResponse,
)
from app.sensors import DEFAULT_SENSOR, MOVING_AVERAGE_WINDOW, SENSORS, moving_average

logger = logging.getLogger("aerosentry")


@lru_cache(maxsize=2)
def _cached_model(path: str) -> RulModel:
    """Load the model once per path. Failures are not cached, so a fixed file is picked up."""
    return load_model(Path(path))


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Load the model before serving, so the first request does not pay for it (NFR-2).

    A missing or unreadable model does not stop the app. The engine list and the error
    responses still work, and prediction requests keep returning 503 and retry the load.
    """
    path = get_settings().model_path
    try:
        _cached_model(str(path))
    except ModelUnavailable as exc:
        logger.warning("Model not loaded at startup; predictions will return 503: %s", exc)
    yield


app = FastAPI(
    title="AeroSentry",
    version="0.1.0",
    lifespan=lifespan,
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


DATABASE_UNAVAILABLE = "The engine database is unavailable. Try again later."
MODEL_UNAVAILABLE = "The prediction model is unavailable. Try again later."
PREDICTION_MISMATCH = (
    "The prediction failed because the model and the database do not match. "
    "Rebuild both with the pipeline commands in the README."
)


def _is_dashboard_page(request: Request) -> bool:
    """The HTML pages are "/" and "/engines/{id}". Every other path answers in JSON."""
    path = request.url.path
    return path == "/" or path.startswith("/engines/")


def _error_page(request: Request, status_code: int, heading: str, message: str) -> Response:
    """A readable error page with a link back to the engine list (FR-12)."""
    return templates.TemplateResponse(
        request,
        "error.html",
        {"status_code": status_code, "heading": heading, "message": message},
        status_code=status_code,
    )


def _error(status_code: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status_code, content={"detail": message})


def _describe(problem: Any) -> str:
    if problem["type"] == "json_invalid":
        return "the request body is not valid JSON"
    where = ".".join(str(part) for part in problem["loc"] if part != "body")
    return f"{where or 'request body'}: {problem['msg']}"


@app.exception_handler(RequestValidationError)
async def malformed_request(request: Request, exc: RequestValidationError) -> Response:
    if _is_dashboard_page(request):
        entered = request.url.path.rsplit("/", 1)[-1]
        shown = entered if len(entered) <= 40 else entered[:40] + "..."
        return _error_page(
            request,
            422,
            "Invalid engine ID",
            f'"{shown}" is not a valid engine ID. Choose an engine from the list.',
        )
    problems = "; ".join(_describe(p) for p in exc.errors())
    return _error(422, f'Malformed request: {problems}. Send JSON like {{"engine_id": 1}}.')


@app.exception_handler(db.DatabaseUnavailable)
async def database_unavailable(request: Request, exc: db.DatabaseUnavailable) -> Response:
    logger.error("Database unavailable: %s", exc)
    if _is_dashboard_page(request):
        return _error_page(request, 503, "Service unavailable", DATABASE_UNAVAILABLE)
    return _error(503, DATABASE_UNAVAILABLE)


@app.exception_handler(ModelUnavailable)
async def model_unavailable(request: Request, exc: ModelUnavailable) -> Response:
    logger.error("Model unavailable: %s", exc)
    if _is_dashboard_page(request):
        return _error_page(request, 503, "Service unavailable", MODEL_UNAVAILABLE)
    return _error(503, MODEL_UNAVAILABLE)


@app.exception_handler(PredictionError)
async def prediction_failed(request: Request, exc: PredictionError) -> Response:
    logger.error("Prediction failed: %s", exc)
    if _is_dashboard_page(request):
        return _error_page(request, 500, "Prediction failed", PREDICTION_MISMATCH)
    return _error(500, PREDICTION_MISMATCH)


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
    """Render the dashboard overview listing all engines (FR-9)."""
    rows = db.list_engines(settings.db_path)
    engine_list = [EngineSummary(engine_id=r.engine_id, n_cycles=r.n_cycles) for r in rows]
    return templates.TemplateResponse(request, "index.html", {"engines": engine_list})


def _sensor_chart_data(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Raw readings, their moving average and unit labels for every sensor of one engine."""
    sensors: dict[str, Any] = {}
    for sensor in SENSORS:
        raw = [float(row[sensor.column]) for row in history]
        constant = min(raw) == max(raw)
        sensors[sensor.column] = {
            "axis_title": sensor.axis_title,
            "hover_unit": sensor.hover_unit,
            "raw": raw,
            "average": moving_average(raw),
            "constant": constant,
            "constant_value": raw[0] if constant else None,
        }
    return {
        "cycles": [row["cycle"] for row in history],
        "window": MOVING_AVERAGE_WINDOW,
        "sensors": sensors,
    }


@app.get("/engines/{engine_id}")
def engine_detail(
    engine_id: Annotated[int, PathParam(ge=1, le=MAX_ENGINE_ID)],
    request: Request,
    settings: SettingsDep,
) -> Response:
    """Render one engine: its predicted RUL and one chosen sensor over its cycles (FR-10, FR-11)."""
    features = db.feature_history(settings.db_path, engine_id)
    if not features:
        return _error_page(
            request,
            404,
            "Engine not found",
            f"Engine {engine_id} was not found. Choose an engine from the list.",
        )

    model = _cached_model(str(settings.model_path))
    # The model's prediction at every recorded cycle. The last one is the engine's current RUL,
    # the same number /predict returns.
    rul_by_cycle = predict_rul_series(model, [row.values for row in features])
    current_rul = rul_by_cycle[-1]

    history = db.get_engine_history(Path(settings.db_path), engine_id)
    if not history:
        return _error_page(
            request,
            404,
            "No telemetry",
            f"No telemetry found for Engine {engine_id}. It has no recorded sensor readings.",
        )

    chart = _sensor_chart_data(history)
    options = [
        {
            "column": sensor.column,
            "label": sensor.label,
            "constant": chart["sensors"][sensor.column]["constant"],
        }
        for sensor in SENSORS
    ]
    return templates.TemplateResponse(
        request,
        "engine.html",
        {
            "engine_id": engine_id,
            "cycle": features[-1].cycle,
            "predicted_rul": current_rul,
            "rul_cap": model.rul_clip,
            "rul_chart": {
                "cycles": [row.cycle for row in features],
                "rul": rul_by_cycle,
                "cap": model.rul_clip,
            },
            "chart": chart,
            "default_sensor": DEFAULT_SENSOR,
            "sensor_options": options,
        },
    )
