#!/usr/bin/env bash
# Smoke test for a running AeroSentry server or container.
# Usage: scripts/smoke_test.sh [base_url]     (default: http://localhost:8000)
# Run it from anywhere; it moves to the repository root first.
# The expected Plotly hash is read from this checkout, so run it from the same commit that the
# server was built from. A different commit can give a false failure.
# It checks that the service answers, that the data and model are inside it, and that
# the vendored Plotly file served by the app is byte for byte the one recorded in the README.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-http://localhost:8000}"
fail() { echo "SMOKE TEST FAILED: $*" >&2; exit 1; }
ok() { echo "ok: $*"; }

echo "Waiting for $BASE/health (up to 60 seconds)"
for _ in $(seq 1 60); do
  curl -fsS "$BASE/health" >/dev/null 2>&1 && break
  sleep 1
done
curl -fsS "$BASE/health" >/dev/null 2>&1 || fail "/health did not answer"

curl -fsS "$BASE/health" | python3 -c '
import json, sys
assert json.load(sys.stdin) == {"status": "ok"}
' || fail "/health body is not {\"status\": \"ok\"}"
ok "/health"

curl -fsS "$BASE/engines" | python3 -c '
import json, sys
engines = json.load(sys.stdin)
assert len(engines) == 100, f"expected 100 engines, got {len(engines)}"
assert all(e["n_cycles"] > 0 for e in engines)
' || fail "/engines does not list 100 engines with recorded cycles"
ok "/engines lists 100 engines"

curl -fsS -X POST "$BASE/predict" -H "Content-Type: application/json" -d '{"engine_id": 1}' | python3 -c '
import json, sys
body = json.load(sys.stdin)
assert set(body) == {"engine_id", "cycle", "predicted_rul"}, body
assert body["engine_id"] == 1
assert 0.0 <= body["predicted_rul"] <= 125.0, body
' || fail "/predict for engine 1 did not return a prediction between 0 and 125"
ok "/predict for engine 1"

code=$(curl -s -o /dev/null -w '%{http_code}' -X POST "$BASE/predict" \
  -H "Content-Type: application/json" -d '{"engine_id": 999999}')
[ "$code" = "404" ] || fail "/predict for an unknown engine returned $code, expected 404"
ok "/predict for an unknown engine returns 404"

for path in / /engines/1; do
  code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE$path")
  [ "$code" = "200" ] || fail "dashboard page $path returned $code, expected 200"
done
ok "dashboard pages / and /engines/1"

expected=$(grep -oE 'File sha256: `[0-9a-f]{64}`' app/static/vendor/README.md | grep -oE '[0-9a-f]{64}')
[ -n "$expected" ] || fail "could not read the Plotly sha256 from app/static/vendor/README.md"
actual=$(curl -fsS "$BASE/static/vendor/plotly-basic-2.27.0.min.js" | sha256sum | cut -d' ' -f1)
[ "$actual" = "$expected" ] || fail "/static Plotly sha256 is $actual, expected $expected"
ok "/static serves the vendored Plotly file unchanged"

echo "All smoke checks passed against $BASE"
