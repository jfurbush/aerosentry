# AeroSentry

Predictive maintenance dashboard that estimates Remaining Useful Life (RUL), in operating cycles, for turbofan engines. It uses the NASA CMAPSS FD001 dataset, which is simulated data. It is not a failure analysis tool and does not produce fixed maintenance intervals. It gives a per-engine RUL estimate and a sensor trend so a planner can decide when to schedule maintenance.

CIS 553 Software Engineering, University of Michigan-Dearborn, Fall 2026.
Group members: Jeffery Furbush (individual project).

## Status

This repository is at scaffold stage. CI, CD, the Dockerfile, and the folder layout are in place. The data pipeline, model, `/engines`, `/predict`, and the dashboard are not implemented yet. See `docs/changes-log.md` for the current status table. This README describes the target behavior; update it as each piece lands.

## Stack

Python 3.11+, FastAPI, scikit-learn (one model), pandas, SQLite, Jinja2 with Plotly, Docker, GitHub Actions.

## Repository layout

```
app/                FastAPI app, database access, model loading, schemas, templates
pipeline/           features.py, train.py, load_db.py
models/             trained .joblib (generated)
data/raw/           CMAPSS FD001 files (committed)
tests/unit/         unit tests
tests/integration/  API tests
docs/               changes log, report template, ai-review-log/
.github/workflows/  ci.yml, cd.yml
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

## Rebuild the database and model

Place the FD001 files in `data/raw/`, then:

```
python -m pipeline.load_db     # writes data/aerosentry.db
python -m pipeline.train       # writes models/rul_model.joblib, prints test RMSE and baseline
```

Training uses a fixed random seed.

## Run the service

```
uvicorn app.main:app --reload
```

Then open http://localhost:8000 for the dashboard and http://localhost:8000/docs for the API docs.

API (target behavior):

- `GET /engines` returns engine IDs and cycle counts.
- `POST /predict` with `{"engine_id": 1}` returns `engine_id`, `cycle`, `predicted_rul`.
- Unknown engine ID returns 404. Malformed request returns 422.

Currently only `GET /health` exists.

## Run the tests

```
pytest --cov=app --cov=pipeline --cov-report=term-missing
```

Coverage must be at least 80%.

## Static analysis

```
ruff check .
mypy
bandit -r app pipeline -lll
```

## Docker

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
- `cd.yml` runs on merge to `main`: builds the Docker image and pushes it to GitHub Container Registry as `ghcr.io/<owner>/<repo>`, tagged with the commit SHA and `latest`.

Pull the published image:

```
docker pull ghcr.io/<owner>/<repo>:latest
docker run --rm -p 8000:8000 ghcr.io/<owner>/<repo>:latest
```

A live hosted instance is not part of the core scope.

## Pull requests and AI-assisted review

Significant changes go through a pull request. Before merging, an AI tool reviews the diff, and the feedback and my response are recorded in `docs/ai-review-log/PR-<number>.md` using `TEMPLATE.md`. This substitutes for second-person review on an individual project and was approved by the instructor. scaffold was generated with an AI assistant and reviewed by myself.

## Data and attribution

NASA CMAPSS Turbofan Engine Degradation Simulation dataset: A. Saxena, K. Goebel, D. Simon, N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation," NASA Prognostics Data Repository.
