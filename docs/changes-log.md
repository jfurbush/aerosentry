# Changes Log

Record any change to the Phase I or Phase II requirements and scope here. Phase III asks for this.

| Date | Change | Reason | Requirements affected |
|------|--------|--------|-----------------------|
| 2026-11 | The engine page plots the model's predicted RUL at every recorded cycle. The earlier trajectory chart, which counted back from the final prediction and so was not model output, was removed | The old chart showed made-up values and started above the 125 cap. The new chart is the model's own output. Its error at earlier cycles is not measured, because `RUL_FD001.txt` gives the true RUL only at each test engine's last cycle | FR-10 (PR #25, PR #26) |
| 2026-11 | CD starts after CI has passed on a push to `main`, instead of on every push, and it can be started by hand. CI now also builds the Docker image and smoke-tests the container | CD used to run without waiting for CI, so an untested commit could be published, and nothing had built the image in CI | FR-13, FR-14 (PR #27) |
| 2026-11 | The database holds the 100 FD001 test engines (the in-service fleet), not the training engines. The training engines run to failure and are used only to fit the model | Planner needs engines whose RUL is still unknown | FR-5, FR-6 |
| 2026-11 | Seven constant sensors (s1, s5, s6, s10, s16, s18, s19) are excluded from model features. All 21 sensors are still stored in `readings` for charting | They never change in FD001, so they carry no signal | FR-3, FR-11 |
| 2026-11 | Model is scored on the last recorded cycle of each test engine against RUL_FD001.txt, with the true value clipped at 125 to match the training target. The unclipped RMSE is also reported | Standard FD001 protocol | NFR-1 |
| 2026-11 | All API errors share one shape, `{"detail": "<message>"}`. A missing or unreadable database or model returns 503 and is logged, without showing server paths to the caller. A model and database that do not match return 500 | UC-2 exception flows need readable messages and logging | FR-8, FR-12 |
| 2026-11 | `engine_id` must be a whole number from 1 to 2147483647. Strings such as "1", decimals, true or false, and unknown fields are rejected with 422 | Keeps the input unambiguous and avoids a database overflow on huge numbers | FR-8 |
| 2026-11 | The service opens the database read-only and loads the model once. Predicted RUL is kept between 0 and 125 and rounded to one decimal | UC-1 and UC-2 state that no data is changed. The model was trained on a target capped at 125 | FR-5, FR-7 |
| 2026-10 | `/predict` takes an engine ID and reads precomputed features from SQLite instead of accepting raw sensor data | Keeps requests small, clients do not build feature windows | FR-7 (Phase II, Section 7) |

## Implementation status

Status as of 2026-11-13.

| Item | Status |
|------|--------|
| FR-1 to FR-5 data pipeline, training, SQLite | Implemented. Test RMSE 18.6 cycles vs 41.9 mean baseline (target 25). See `models/metrics.json` |
| FR-6 to FR-8 `/engines`, `/predict`, error handling | Implemented. 404, 422 and 503 cases are tested |
| FR-9 to FR-12 dashboard | Implemented: engine list, engine page with the predicted RUL, a chart of the model's predicted RUL at every cycle, a one-sensor chart with a moving average, and error pages. It works without internet access. Checked in Chromium on Linux and in Chrome 157 on Windows. Firefox not tested |
| FR-13 Docker image | Built and started by the CI job `docker-smoke-test` on every push and pull request, and the smoke test passes. Pushed to `ghcr.io/jfurbush/aerosentry` by CD. The published image was pulled and run on Windows with Docker 29.9.2 on 2026-11-12 (image 1.18 GB as `docker image ls` shows it) |
| FR-14 CI and CD workflows | CI is active: ruff, mypy, bandit, pytest with coverage, and the Docker smoke test. CD succeeded in a manual run (run 42708003586) and has started by itself after CI passed on `main` (runs 42949254063 and 43312819800) |

Update this table as work lands. It feeds the Phase III implementation and testing report.
