# Browser check

The dashboard was opened on the same laptop as the measurements, with the server running directly (`uvicorn app.main:app`, no container).

## Chrome 157.0.8103.78 (64-bit), Windows, checked 2026-11-13

Load time of the engine page, read from the Network tab of the developer tools with the cache disabled. Three reloads per engine.

| Engine | Load 1 (ms) | Load 2 (ms) | Load 3 (ms) |
|--------|-------------|-------------|-------------|
| 1 | 420 | 382 | 374 |
| 97 | 453 | 401 | 377 |
| 49 | 428 | 402 | 440 |

The highest of the nine loads is 453 ms. The requirement is under 3 seconds. The charts are drawn by scripts that run before the page load event, so this time includes drawing them (checked in headless Chromium on Linux, where both charts were present when the load event fired).

On engines 1, 97 and 49 both charts drew, the axis labels showed cycles, and changing the sensor in the list changed the chart. The server log from this session shows only requests to `localhost:8000`, plus two harmless 404s: a missing favicon and a probe from the Chrome developer tools.

## Firefox 160.0 (64-bit), Windows, reported 2026-11-17

I opened the dashboard in Firefox and saw no problems. I did not record which engines I opened or which things I checked, and I did not record any load times. This is a check by eye, not a measurement.
