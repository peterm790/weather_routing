<INSTRUCTIONS>
# AGENTS

## Benchmarks
- Latest change (replace geopy great-circle calls with `_haversine_nm_scalar`) did not speed up the benchmark.
- The most recent run showed a slight increase in `algo_runtime_seconds`.
- Always run the benchmark after any code change that could affect performance or routing.
- Attempted to avoid per-heading route list copies in `get_possible`/`get_possible_optimized`; no meaningful improvement, reverted.
- Attempted a fast Numba-based `get_wind` in the benchmark script; no meaningful improvement, reverted.
- Benchmark now runs with `avoid_land_crossings="strict"` for land checks.

## Data Integrity
- Never use fake data or visualizations. All plots must be generated from real routing outputs and real weather data.

## Antimeridian Plotting
- Use `ccrs.PlateCarree(central_longitude=180)` to center plots on the dateline.
- When setting extents across the dateline, keep west < east by using a 0..360 window
  (e.g. `set_extent([170, 190, -10, 10])` because `190 == -170`).
- For plotting routes/isochrones, convert longitudes to a continuous 0..360 view so
  values like `-179` appear next to `+179` (e.g. `(lon + 360) % 360`).
</INSTRUCTIONS>
