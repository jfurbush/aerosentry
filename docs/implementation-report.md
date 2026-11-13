# AeroSentry: Phase III Implementation and Testing Report

Group members: Jeffery Furbush (individual project)
Repository: https://github.com/jfurbush/aerosentry
Date: 2026-11-13
Code state: `main` at 7f71b82. The tests and measurements were run on de1b174, which has the same application code. Copies of the tool output are in `docs/evidence/`.

AeroSentry estimates the remaining useful life (RUL), in operating cycles, of turbofan engines from the simulated NASA CMAPSS FD001 data. It is a FastAPI service with a SQLite database, a random forest model and a server-rendered dashboard, packaged as a Docker image. All 14 functional requirements are implemented (FR-5 with a scope change, see section 2). The model test RMSE is 18.6 cycles against a target of 25. Gaps are listed in section 8.

## 1. Completed functionality

| Requirement | Status | Notes |
|-------------|--------|-------|
| FR-1 Load FD001 training and test files | Done | `pipeline/data.py`. Readings are organized per engine and cycle. |
| FR-2 RUL target clipped at 125 | Done | Cycles to failure, clipped at 125. |
| FR-3 Rolling mean and standard deviation | Done | Window sizes are a parameter (5 and 10 by default). The 7 constant sensors are not used as features. The windows only look backward. |
| FR-4 Train, save .joblib, report RMSE | Done | Random forest, 200 trees, saved to `models/rul_model.joblib`. The RMSE is in `models/metrics.json`. |
| FR-5 SQLite storage | Done, scope change | The database holds history and features for the 100 test engines (the in-service fleet), not the training engines. |
| FR-6 `GET /engines` | Done | Returns each engine ID and its number of cycles. |
| FR-7 `POST /predict` | Done | Returns engine ID, latest cycle and predicted RUL, kept between 0 and 125. |
| FR-8 404 and 422 | Done | 404 for an unknown engine. 422 for strings, decimals, booleans, unknown fields and out-of-range IDs. Errors share one shape, `{"detail": ...}`. A missing database or model gives 503. |
| FR-9 Engine list and selection | Done | |
| FR-10 Predicted RUL and chart over cycles | Done | The page shows the predicted RUL, a sensor chart with a moving average, and a chart of the model's predicted RUL at every cycle. |
| FR-11 Choose the sensor | Done | One sensor at a time, from all 21 stored sensors. |
| FR-12 Readable errors | Done | Error pages for an unknown engine and for an unavailable model or database. |
| FR-13 Docker image | Done | Two-stage build. The image holds the application, the model and the database, runs as a non-root user (`id` inside the container printed `uid=10001(appuser)`), and starts with `docker run`. I pulled the published image and ran it on Windows on 2026-11-12. The image is 1.18 GB as `docker image ls` shows it. Output: `docs/evidence/docker-image.txt`. |
| FR-14 CI and CD | Done | CI runs on every push and pull request. CD builds and pushes the image after CI passes on a push to `main`. |

### Non-functional requirements

| Requirement | Result | Evidence |
|-------------|--------|----------|
| NFR-1 RMSE reported with a baseline, target 25 or less | Met | 18.557 against a baseline of 41.942 (section 5). |
| NFR-2 `/predict` under 500 ms | Met | Median 8.2 ms running directly, 51.9 ms in the container, maximum 54.2 ms (section 3). |
| NFR-2 Detail page under 3 s | Met in Chrome | Page load took 374 to 453 ms in 9 loads with the cache off. Not measured in Firefox. |
| NFR-3 Coverage of at least 80%. All tests pass before a merge | Coverage met. Merge rule partly shown | Coverage is 98.98%. For PRs #26 to #30 I watched the checks go green before merging. I did not check earlier PRs, and I did not check whether branch protection enforces it. |
| NFR-4 ruff and mypy clean on `main`. Type hints on public functions | Met | Both clean in CI and locally. mypy runs in strict mode on `app/` and `pipeline/`. |
| NFR-5 No high-severity bandit findings. No secrets. Non-root container. No uploads | Met, with a limit | bandit reports no findings at either level. The container runs as a non-root user (checked with `id`). No secret scanner was run. The API takes only an engine ID. |
| NFR-6 Fixed seed. Pinned dependencies. Builds from a clean checkout | Met, with a limit | Seed is 42. The direct dependencies are pinned, the transitive ones are not. CI builds the image from a fresh checkout on every push and pull request. |
| NFR-7 Python 3.11+ and a Linux container | Met | Python 3.11 on Windows and on the Linux CI runner. Other Python versions were not tested. |
| NFR-8 Chrome and Firefox. Values labelled in cycles | Chrome only | Chrome 157 on Windows, and Chromium on Linux earlier. Firefox was not tested. The axis labels were checked on engines 1, 97 and 49. |
| NFR-9 Every merged PR has an AI-review log | Partly met | 20 PRs were merged (#11 to #30). Each has a log file. Several logs were written after the merge, and PR #20 had no AI review. See section 7. |
| NFR-10 README covers setup, running, tests, rebuilding the model | Met | README sections: Setup, Run, Test, Rebuild the model, Docker, CI/CD and deployment. |

## 2. Changes to requirements or scope

The full list is in `docs/changes-log.md`. The main changes:

- The engine page used to draw a trajectory that counted back from the final prediction, which was not model output. PR #25 removed it. PR #26 added a chart of the model's own prediction at every cycle (FR-10).
- CD used to run on every push. It now starts only after CI passes on a push to `main`, and CI also builds and smoke-tests the Docker image (FR-13, FR-14).
- The database holds the 100 test engines, not the training engines (FR-5, FR-6).
- Seven sensors that never change in FD001 (s1, s5, s6, s10, s16, s18, s19) are not model features. All 21 stay in the database for charting (FR-3, FR-11).
- The model is scored on the last recorded cycle of each test engine, with the true RUL clipped at 125. The unclipped result is also reported (NFR-1).
- `/predict` takes an engine ID and reads precomputed features. API errors share one shape. Engine IDs must be whole numbers from 1 to 2147483647. The service opens the database read-only and loads the model once at startup.

## 3. Test results

- Command: `pytest --cov=app --cov=pipeline --cov-fail-under=80`. Result: 173 passed, 101 unit tests and 72 integration tests (the integration tests cover the API, the dashboard, startup, offline use and the real FD001 data).
- Line coverage on `app/` and `pipeline/`: 98.98% (491 statements, 5 not covered). The target is 80%.
- Passing CI run on `main`: https://github.com/jfurbush/aerosentry/actions/runs/43312277462
- Output copies: `docs/evidence/pytest.txt`.

Response times, measured on one laptop (Intel Core Ultra 7 265H, 63 GB RAM, Windows build 10.0.26300, Python 3.11.9, Docker 29.9.2) on 2026-11-12 with `scripts/measure.py`. These are server times over localhost, one reused connection:

| Request | Running directly: median, 95th percentile, max (ms) | In the container: median, 95th percentile, max (ms) |
|---------|------|------|
| `POST /predict`, 50 requests | 8.2, 9.4, 11.0 | 51.9, 53.6, 54.2 |
| `GET /` engine list, 10 requests | 3.5, 62.1, 62.1 | 2.7, 63.3, 63.3 |
| `GET /engines/1`, 10 requests | 11.4, 59.3, 59.3 | 11.0, 13.1, 13.1 |
| `GET /engines/97`, 10 requests | 21.9, 34.3, 34.3 | 20.9, 22.2, 22.2 |
| `GET /engines/49`, 10 requests | 40.3, 52.9, 52.9 | 37.7, 41.2, 41.2 |

The flat 52 ms for `/predict` in the container is probably a fixed delay in the network path through Docker Desktop and not the model, but I did not confirm that. The first `/predict` after startup took 19.8 ms directly and 55.0 ms in the container, a single sample each, so there is no sign of a slow first request. In a browser, Chrome 157 loaded the engine pages in 374 to 453 ms (engine 1: 420, 382, 374; engine 97: 453, 401, 377; engine 49: 428, 402, 440), measured in the developer tools with the cache disabled. The charts are drawn before the page load event, so this includes drawing them. Files: `docs/evidence/measurements-windows-uvicorn.md` and `docs/evidence/measurements-windows-docker.md`.

## 4. Static analysis

- ruff (rules E, F, W, I, B, UP, S): no problems. `ruff format --check`: 29 files already formatted.
- mypy (strict mode, 12 source files in `app/` and `pipeline/`): no issues.
- bandit: no issues at `-ll` (Medium and up, run locally) and at `-lll` (High only, run in CI). 834 lines scanned. The target was no high-severity findings.
- Output copies: `docs/evidence/ruff-check.txt`, `ruff-format.txt`, `mypy.txt`, `bandit-ll.txt`, `bandit-lll.txt`.

## 5. Model result

- Test-set RMSE on FD001: 18.557 cycles against the clipped true RUL (95% interval 15.0 to 21.9). The target was 25 or less, so it is met.
- Baseline (predicting the training mean): 41.942 cycles.
- Against the clipped true RUL: mean absolute error 12.9 cycles, mean error +4.0 cycles (so the model overestimates remaining life slightly on average), largest single error 60.6 cycles. Against the unclipped true RUL: RMSE 19.574, baseline 43.067, mean absolute error 14.0 cycles, mean error +3.0 cycles, largest single error 60.6 cycles.
- Evaluated only at the last recorded cycle of each of the 100 test engines, because `RUL_FD001.txt` gives the true RUL only there. Accuracy at earlier cycles is not measured.
- Random forest, 200 trees, seed 42, scikit-learn 1.5.1. The 21-row sensor table in `app/sensors.py` was compared by hand with Table 2 of Saxena et al. (PHM 2008) on 2026-11-10.

## 6. CI/CD status

- CI (`.github/workflows/ci.yml`): runs on every push and pull request. Job `test-and-analyze` runs ruff, mypy, bandit and pytest with an 80% coverage minimum. Job `docker-smoke-test` builds the image, starts the container and runs `scripts/smoke_test.sh` against it. Both pass on `main`.
- CD (`.github/workflows/cd.yml`): builds the image and pushes it to GitHub Container Registry as `ghcr.io/jfurbush/aerosentry`, tagged `latest` and `sha-<first 7 characters of the commit>`. It starts after CI passes on a push to `main`, or by hand. Runs: manual on 2026-11-09 (42708003586), automatic on 2026-11-10 (42949254063) and automatic on 2026-11-12 with the updated actions (43312819800). All succeeded. The package is public.
- Live deployment: not attempted. The deliverable is the published image, which I pulled and ran on Windows on 2026-11-12. The image I ran has the registry digest `sha256:42f2fa6e70c19910bb0cd5c469064bfccbffcd1fb970eaf4e7ecc680cc41658a` (`docs/evidence/docker-image.txt`).

## 7. AI use

The code was written with Claude (Claude Code), and I reviewed it and ran it. Each pull request was also reviewed by a second AI tool, either DeepSeek (free web chat) or Gemini (web chat). The feedback and my decision on each item are in `docs/ai-review-log/`, one file per PR. The README records that the instructor approved this in place of second-person review on an individual project.

How far this can be trusted:

- The triage of each review was done with Claude, which also wrote most of the code, so it is not independent. I checked claims by running things where I could.
- Reviewer model versions were mostly not recorded. One DeepSeek reply named itself as Claude 3.5 Sonnet, which I could not confirm.
- Some reviews were thin: a few returned no findings and quoted no lines, and one missed a real gap.
- The logs for PRs #13, #14, #17, #18, #19 and #28 were written after the merge. The logs for #17 and #18 contained wrong statements, and PR #28 added visible correction notes to them. PR #20 added the logs for #17 to #19 and had no AI review of its own. Its log records that.

## 8. Known limitations

- The model's error at earlier cycles is not measured, only at each test engine's last cycle.
- Firefox was not tested. Chrome 157 on Windows was tested on three engines.
- Measurements come from one laptop, with one run per row. The cause of the 52 ms `/predict` time in the container is unconfirmed.
- The image is 1.18 GB, which is large for this service.
- `scripts/measure.py` has no automated tests, and `mypy` does not cover `scripts/`.
- A NULL sensor value in the database would give a 500 on the dashboard. The API docs page (`/docs`) needs a CDN, so it does not work offline. The site has no favicon, so browsers get a 404 for it.
- Only the direct dependencies are pinned. No secret scanner was run.
- Windows application control blocked scikit-learn intermittently on one machine, and `core.autocrlf=true` changed the vendored Plotly file until `.gitattributes` fixed it.
- This is an individual project, so there was no second contributor and no second-person code review.
