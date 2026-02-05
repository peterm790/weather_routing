<INSTRUCTIONS>
# AGENTS

## Benchmarks
- Latest change (replace geopy great-circle calls with `_haversine_nm_scalar`) did not speed up the benchmark.
- The most recent run showed a slight increase in `algo_runtime_seconds`.
- Always run the benchmark after any code change that could affect performance or routing.
- Attempted to avoid per-heading route list copies in `get_possible`/`get_possible_optimized`; no meaningful improvement, reverted.
- Attempted a fast Numba-based `get_wind` in the benchmark script; no meaningful improvement, reverted.
</INSTRUCTIONS>
