# Clean-clone check

Done on 2026-11-16 on the same laptop as the measurements. The repository was cloned from GitHub into a new folder (`git clone https://github.com/jfurbush/aerosentry`) at main after PR #31 (0f4ab89), and the README was followed from there.

## What I did and what happened

| Step | Result |
|------|--------|
| `python --version` outside any virtual environment | Python 3.15.2 |
| Virtual environment | Made with `py -3.11 -m venv .venv`, not with the README's plain `python -m venv .venv` of that time, because the default Python here is 3.15.2. The README now says to use Python 3.11 |
| `pip install -r requirements-dev.txt` | Completed. The last lines I saw were only pip's "new release available" notice. That the install worked is shown by the next steps |
| `python -m pipeline.load_db` | Wrote the database: 100 engines, 13096 readings, 13096 feature rows |
| `python -m pipeline.train` | Finished. `git status --short` afterwards printed nothing, so the retrained `models/metrics.json` is identical to the committed one |
| `pytest --cov=app --cov=pipeline --cov-report=term-missing -q` | 173 passed, coverage 98.98%, 29.22 s |
| `uvicorn app.main:app`, then `GET /health` | `status: ok` |
| `GET /engines` | 100 engines |
| `POST /predict` with engine 1 | `engine_id` 1, `cycle` 31, `predicted_rul` 121.7 |

## What this does not cover

- The static analysis commands and the Docker build were not repeated in the clean clone. CI runs both on every push from a fresh checkout.
- Only Windows was tried here. The Linux container is built and smoke-tested by CI.
- I saw the terminal output, not a log file, so this page is my record of it.
