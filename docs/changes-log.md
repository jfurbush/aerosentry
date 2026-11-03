# Changes Log

Record any change to the Phase I or Phase II requirements and scope here. Phase III asks for this.

| Date | Change | Reason | Requirements affected |
|------|--------|--------|-----------------------|
| 2026-11 | The database holds the 100 FD001 test engines (the in-service fleet), not the training engines. The training engines run to failure and are used only to fit the model | Planner needs engines whose RUL is still unknown | FR-5, FR-6 |
| 2026-11 | Seven constant sensors (s1, s5, s6, s10, s16, s18, s19) are excluded from model features. All 21 sensors are still stored in `readings` for charting | They never change in FD001, so they carry no signal | FR-3, FR-11 |
| 2026-11 | Model is scored on the last recorded cycle of each test engine against RUL_FD001.txt, with the true value clipped at 125 to match the training target. The unclipped RMSE is also reported | Standard FD001 protocol | NFR-1 |
| 2026-10 | `/predict` takes an engine ID and reads precomputed features from SQLite instead of accepting raw sensor data | Keeps requests small, clients do not build feature windows | FR-7 (Phase II, Section 7) |

## Implementation status

| Item | Status |
|------|--------|
| FR-1 to FR-5 data pipeline, training, SQLite | Implemented. Test RMSE 18.6 cycles vs 41.9 mean baseline (target 25). See `models/metrics.json` |
| FR-6 to FR-8 `/engines`, `/predict`, error handling | Not started |
| FR-9 to FR-12 dashboard | Not started |
| FR-13 Docker image | Dockerfile drafted, fails until pipeline exists |
| FR-14 CI and CD workflows | Written, CI runs on scaffold, CD untested |

Update this table as work lands. It feeds the Phase III implementation and testing report.
