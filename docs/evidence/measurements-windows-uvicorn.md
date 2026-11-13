# Measurements

- Measured: 2026-11-12 21:46 UTC
- Machine: work laptop, Intel(R) Core(TM) Ultra 7 265H, 63 GB RAM
- Operating system: Windows-10-10.0.26300-SP0
- Processor: Intel64 Family 6 Model 197 Stepping 2, GenuineIntel, 16 logical CPUs
- Python running this script: 3.11.9
- Server: http://localhost:8000 (uvicorn, direct on Windows)
- /health answered 10.7 s after the script started waiting
- Plotly file served: 998,628 bytes, sha256 matches the recorded checksum

| Request | Samples | Median ms | 95th pct ms | Max ms | Body KB |
|---------|---------|-----------|-------------|--------|---------|
| first POST /predict, engine 1 (right after /health) | 1 | 19.8 | 19.8 | 19.8 | 0.0 |
| POST /predict, engine 1 | 50 | 8.2 | 9.4 | 11.0 | 0.0 |
| GET / (engine list) | 10 | 3.5 | 62.1 | 62.1 | 29.5 |
| GET /engines/1 | 10 | 11.4 | 59.3 | 59.3 | 22.6 |
| GET /engines/97 | 10 | 21.9 | 34.3 | 34.3 | 58.0 |
| GET /engines/49 | 10 | 40.3 | 52.9 | 52.9 | 116.3 |

Times are server time plus the local network stack over localhost, with one reused
connection. They do not include a browser drawing the charts.
