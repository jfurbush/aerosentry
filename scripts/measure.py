"""Measure response times of a running AeroSentry server or container.

Start the server first, then run this script right away. It does not start anything itself.

    uvicorn app.main:app                      (or: docker run --rm -p 8000:8000 aerosentry:latest)
    python scripts/measure.py --machine "Laptop name as printed in Settings"

The first /predict request is timed once, as soon as /health answers. That is the number to compare
with the cold start before the model was loaded at startup, so run the script straight after the
server starts. Every other row repeats the request and reports the median, the 95th percentile
(nearest rank) and the maximum.

The times are server time plus the local network stack, measured over localhost with one reused
connection. They do not include a browser drawing the charts. Use the browser's developer tools
for that.
"""

import argparse
import hashlib
import math
import os
import platform
import statistics
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

PLOTLY_PATH = "/static/vendor/plotly-basic-2.27.0.min.js"
VENDOR_README = Path(__file__).resolve().parent.parent / "app" / "static" / "vendor" / "README.md"
# Test engines 1, 97 and 49 are the ones used for the dashboard checks in PR #26.
PAGE_ENGINES = (1, 97, 49)


@dataclass
class Row:
    label: str
    times_ms: list[float]
    body_bytes: int

    def cells(self) -> str:
        ordered = sorted(self.times_ms)
        p95 = ordered[max(math.ceil(0.95 * len(ordered)) - 1, 0)]
        median = statistics.median(ordered)
        return (
            f"| {self.label} | {len(ordered)} | {median:.1f} | {p95:.1f} | {ordered[-1]:.1f} "
            f"| {self.body_bytes / 1024:.1f} |"
        )


def timed(client: httpx.Client, method: str, path: str, **kwargs: object) -> tuple[float, int]:
    """Return (milliseconds, response body size in bytes). Stop with a message on an error."""
    start = time.perf_counter()
    try:
        response = client.request(method, path, **kwargs)  # type: ignore[arg-type]
    except httpx.HTTPError as exc:
        raise SystemExit(f"{method} {path} failed: {exc}") from exc
    elapsed = (time.perf_counter() - start) * 1000
    if response.status_code != 200:
        raise SystemExit(f"{method} {path} returned HTTP {response.status_code}, expected 200")
    return elapsed, len(response.content)


def wait_for_health(client: httpx.Client, seconds: int) -> float:
    """Wait until /health answers and return how long that took, in seconds."""
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        try:
            if client.get("/health").status_code == 200:
                return time.perf_counter() - start
        except httpx.HTTPError:
            pass
        time.sleep(0.25)
    raise SystemExit(f"/health did not answer within {seconds} seconds. Is the server running?")


def repeat(
    client: httpx.Client, label: str, method: str, path: str, count: int, **kw: object
) -> Row:
    times: list[float] = []
    size = 0
    for _ in range(count):
        elapsed, size = timed(client, method, path, **kw)
        times.append(elapsed)
    return Row(label, times, size)


def plotly_check(client: httpx.Client) -> str:
    """Compare the Plotly file the server sends with the checksum recorded in the vendor README."""
    response = client.get(PLOTLY_PATH)
    if response.status_code != 200:
        return f"HTTP {response.status_code} for {PLOTLY_PATH}"
    actual = hashlib.sha256(response.content).hexdigest()
    size = f"{len(response.content):,} bytes"
    if not VENDOR_README.exists():
        return f"{size}, sha256 {actual} (vendor README not found, not compared)"
    marker = "File sha256: `"
    text = VENDOR_README.read_text(encoding="utf-8")
    if marker not in text:
        return f"{size}, sha256 {actual} (no recorded checksum found in the vendor README)"
    recorded = text.split(marker, 1)[1][:64]
    verdict = (
        "matches the recorded checksum"
        if actual == recorded
        else "DOES NOT MATCH the recorded checksum"
    )
    return f"{size}, sha256 {verdict}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument(
        "--machine", default="not given", help="name of the machine, as printed by the OS"
    )
    parser.add_argument("--note", default="", help="free text, for example 'uvicorn' or 'docker'")
    parser.add_argument("--predict-runs", type=int, default=50)
    parser.add_argument("--page-runs", type=int, default=10)
    parser.add_argument("--out", help="also write the result to this file (UTF-8)")
    args = parser.parse_args()

    started = datetime.now(UTC)
    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        ready_after = wait_for_health(client, 60)
        payload = {"json": {"engine_id": 1}}
        first_ms, first_size = timed(client, "POST", "/predict", **payload)
        rows = [Row("first POST /predict, engine 1 (right after /health)", [first_ms], first_size)]
        rows.append(
            repeat(
                client, "POST /predict, engine 1", "POST", "/predict", args.predict_runs, **payload
            )
        )
        rows.append(repeat(client, "GET / (engine list)", "GET", "/", args.page_runs))
        for engine in PAGE_ENGINES:
            rows.append(
                repeat(
                    client, f"GET /engines/{engine}", "GET", f"/engines/{engine}", args.page_runs
                )
            )
        plotly = plotly_check(client)

    lines = [
        "# Measurements",
        "",
        f"- Measured: {started:%Y-%m-%d %H:%M} UTC",
        f"- Machine: {args.machine}",
        f"- Operating system: {platform.platform()}",
        f"- Processor: {platform.processor() or 'not reported'}, {os.cpu_count()} logical CPUs",
        f"- Python running this script: {platform.python_version()}",
        f"- Server: {args.base_url}" + (f" ({args.note})" if args.note else ""),
        f"- /health answered {ready_after:.1f} s after the script started waiting",
        f"- Plotly file served: {plotly}",
        "",
        "| Request | Samples | Median ms | 95th pct ms | Max ms | Body KB |",
        "|---------|---------|-----------|-------------|--------|---------|",
        *[row.cells() for row in rows],
        "",
        "Times are server time plus the local network stack over localhost, with one reused",
        "connection. They do not include a browser drawing the charts.",
        "",
    ]
    result = "\n".join(lines)
    print(result)
    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
