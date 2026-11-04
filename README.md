# AeroSentry

Predictive maintenance dashboard that estimates Remaining Useful Life (RUL), in operating cycles, for turbofan engines. It uses the NASA CMAPSS FD001 dataset, which is simulated data. It is not a failure analysis tool and does not produce fixed maintenance intervals. It gives a per-engine RUL estimate and a sensor trend so a planner can decide when to schedule maintenance.

CIS 553 Software Engineering, University of Michigan-Dearborn, Fall 2026.
Group members: Jeffery Furbush (individual project).

## Status

Working: FD001 loading and features (`pipeline/data.py`, `pipeline/features.py`), model training (`pipeline/train.py`), SQLite storage (`pipeline/load_db.py`), and the API (`/health`, `/engines`, `/predict`). Not done: the dashboard. The Docker build has not been tested yet, and the CD workflow is disabled until it has been. See `docs/changes-log.md` for the status table.

## Stack

Python 3.11+, FastAPI, scikit-learn (one model), pandas, SQLite, Jinja2 with Plotly, Docker, GitHub Actions.

## Repository layout

```
app/                FastAPI app, database access, model loading, schemas, templates
pipeline/           data.py, features.py, train.py, load_db.py
models/             trained .joblib (generated)
data/raw/           CMAPSS FD001 files (committed)
tests/unit/         unit tests
tests/integration/  API tests
docs/               changes log, report template, ai-review-log/
.github/workflows/  ci.yml, cd.yml
```

## Setup

Windows PowerShell:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

Linux or macOS:

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```
If PowerShell refuses to run the activation script with an execution policy error, allow local scripts for your user once (answer Y if asked), then activate again:

Set-ExecutionPolicy -Scope CurrentUser RemoteSigned


## Rebuild the database and model

The database and model are generated files and are not committed. The raw FD001 files are in `data/raw/`. To build them:

```
python -m pipeline.load_db     # writes data/aerosentry.db
python -m pipeline.train       # writes models/rul_model.joblib and models/metrics.json
```

Training uses a fixed random seed.

## Run the service

Build the database and model first (see above), then:

```
uvicorn app.main:app --reload
```

API docs are at http://localhost:8000/docs. There is no dashboard yet.

API:

- `GET /health` returns `{"status": "ok"}` while the service is running. It is a liveness check only and does not touch the model or database.
- `GET /engines` returns a list of `{"engine_id", "n_cycles"}` for each engine in the database.
- `POST /predict` with `{"engine_id": 1}` returns `engine_id`, `cycle` (the engine's latest recorded cycle) and `predicted_rul` in cycles. The model was trained on RUL capped at 125, so a value near 125 means "at least about 125".

Errors are always `{"detail": "..."}`:

- 404: unknown engine ID.
- 422: malformed request, including a non-integer, zero, negative, or missing `engine_id`.
- 503: the database or model file is unavailable.
- 500: the model and database do not match.

## Run the tests

```
pytest --cov=app --cov=pipeline --cov-report=term-missing
```

Coverage must be at least 80%.

## Static analysis

```
ruff check .
ruff format --check .
mypy
bandit -r app pipeline -lll
```

## Docker

Not yet tested: this section describes the intended build. It has not been run successfully yet.

Build (trains the model during the build, from `data/raw/`):

```
docker build -t aerosentry .
```

Run:

```
docker run --rm -p 8000:8000 aerosentry
```

The container runs as a non-root user.

## CI/CD and deployment

- `ci.yml` runs on every push and pull request: ruff, mypy, bandit, pytest with coverage. Reports are uploaded as a workflow artifact named `ci-reports`.
- `cd.yml` builds the Docker image and pushes it to GitHub Container Registry as `ghcr.io/<owner>/<repo>`, tagged with the commit SHA and `latest`. It is configured to run on merge to `main`, but it is currently disabled until the Docker build has been tested.

No image has been published yet. Once one is, it can be pulled with:

```
docker pull ghcr.io/<owner>/<repo>:latest
docker run --rm -p 8000:8000 ghcr.io/<owner>/<repo>:latest
```

A live hosted instance is not part of the core scope.

## Pull requests and AI-assisted review

Significant changes go through a pull request. Before merging, an AI tool reviews the diff, and the feedback and my response are recorded in `docs/ai-review-log/PR-<number>.md` using `TEMPLATE.md`. This substitutes for second-person review on an individual project and was approved by the instructor.

## AI use

Parts of this project were written with an AI assistant (Claude), including the initial scaffold (folder layout, CI/CD workflows, Dockerfile, templates) and later pipeline and API code. I read, ran and tested what it produced before committing. Every pull request is also reviewed by an AI tool before merging, and the feedback and my response are in `docs/ai-review-log/`.

## Data and attribution

NASA CMAPSS Turbofan Engine Degradation Simulation dataset: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," NASA Prognostics Data Repository.
