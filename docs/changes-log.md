# Changes Log

Record any change to the Phase I or Phase II requirements and scope here. Phase III asks for this.

| Date | Change | Reason | Requirements affected |
|------|--------|--------|-----------------------|
| 2026-10 | `/predict` takes an engine ID and reads precomputed features from SQLite instead of accepting raw sensor data | Keeps requests small, clients do not build feature windows | FR-7 (Phase II, Section 7) |

## Implementation status

| Item | Status |
|------|--------|
| FR-1 to FR-5 data pipeline, training, SQLite | Not started |
| FR-6 to FR-8 `/engines`, `/predict`, error handling | Not started |
| FR-9 to FR-12 dashboard | Not started |
| FR-13 Docker image | Dockerfile drafted, fails until pipeline exists |
| FR-14 CI and CD workflows | Written, CI runs on scaffold, CD untested |

Update this table as work lands. It feeds the Phase III implementation and testing report.
