# Measurements

- Measured: 2026-11-12 22:01 UTC
- Machine: work laptop, Intel(R) Core(TM) Ultra 7 265H, 63 GB RAM
- Operating system: Windows-10-10.0.26300-SP0
- Processor: Intel64 Family 6 Model 197 Stepping 2, GenuineIntel, 16 logical CPUs
- Python running this script: 3.11.9
- Server: http://localhost:8000 (docker, image pulled from ghcr.io/jfurbush/aerosentry:latest)
- /health answered 0.8 s after the script started waiting
- Plotly file served: 998,628 bytes, sha256 matches the recorded checksum

| Request | Samples | Median ms | 95th pct ms | Max ms | Body KB |
|---------|---------|-----------|-------------|--------|---------|
| first POST /predict, engine 1 (right after /health) | 1 | 55.0 | 55.0 | 55.0 | 0.0 |
| POST /predict, engine 1 | 50 | 51.9 | 53.6 | 54.2 | 0.0 |
| GET / (engine list) | 10 | 2.7 | 63.3 | 63.3 | 29.5 |
| GET /engines/1 | 10 | 11.0 | 13.1 | 13.1 | 22.6 |
| GET /engines/97 | 10 | 20.9 | 22.2 | 22.2 | 58.0 |
| GET /engines/49 | 10 | 37.7 | 41.2 | 41.2 | 116.3 |

Times are server time plus the local network stack over localhost, with one reused
connection. They do not include a browser drawing the charts.
