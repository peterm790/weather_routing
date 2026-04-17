# Code Backlog

## Operational-Risk Cleanup

### P1: Antimeridian route script retries permanent weather/cache errors and silently falls back to unrelated cached windows

- File/line: `test/antimeridian_equator_route.py:37`, `test/antimeridian_equator_route.py:91`, `test/antimeridian_equator_route.py:119`
- Risk: `_load_weather` retries every exception three times, then `main` searches for any matching cached Zarr and may use a shorter weather window. A bad contract, corrupt cache, or changed upstream layout can be masked as a successful real-data route, weakening the regression and potentially producing a plot from stale or mismatched weather.
- Contract/fix: Treat local cache identity as exact. Use one canonical cache path keyed by provider, dataset, init time, lead hours, and bounds. If that cache is corrupt or the requested upstream load fails, fail the script with the original exception. Do not use shorter or wildcard fallback caches.

### P2: Route normalization drops malformed optimized route data instead of failing

- File/line: `src/weather_router/isochronal_weather_router.py:1109`
- Risk: `_normalize_route_points` catches broad exceptions and returns an empty route. Optimization callers can then continue with less information than intended, making malformed route state look like a valid no-points condition.
- Contract/fix: Accept only `DataFrame` with `lat`/`lon` or an iterable of numeric `(lat, lon)` pairs. Raise `ValueError` with the offending route shape/type when the contract is not met. Tests should assert malformed route inputs fail explicitly.

### P2: Optimization progress callback errors are swallowed

- File/line: `src/weather_router/isochronal_weather_router.py:1340`
- Risk: Callback failures during optimization are ignored after a broad catch. In the Modal service this can hide stream/reporting defects while the route continues, causing clients to miss progress/result snapshots without a visible server-side failure.
- Contract/fix: Define a single callback signature and call it directly. If the callback fails, let the exception propagate or convert it to one explicit routing error event at the API boundary. Do not support multiple callback signatures in the optimizer.

### P3: Benchmark polar download silently falls back to local polar data

- File/line: `test/demo_timing.py:315`
- Risk: The benchmark can report a successful run using local polar data when the remote polar fetch fails. That is useful for development, but it makes the benchmark environment ambiguous and can mask an operational dependency outage.
- Contract/fix: Add an explicit `--offline-polar` flag or environment variable for local-polar mode. By default, benchmark runs should fail if the configured remote polar source is unavailable.

### P3: Benchmark map render failure is non-fatal

- File/line: `test/demo_timing.py:466`
- Risk: The benchmark can succeed without producing the requested route visualization. That is acceptable for timing-only runs but conflicts with the repo rule that generated plots come from real routing outputs and weather data.
- Contract/fix: Add an explicit `--skip-map` flag for timing-only runs. By default, map rendering should fail the benchmark if it cannot write the output image.
