## QtVLM-Style Router vs Isochronal Weather Router

This note compares the new QtVLM-style router (`src/weather_router/qtvlm_weather_router.py`) with the existing isochronal router (`src/weather_router/isochronal_weather_router.py`), and explains how results and performance may differ in practice.

### High-level algorithmic differences

- **Routing pass**
  - QtVLM router: single-pass isochrone expansion with simple pruning (bucket + wake). No secondary optimization pass.
  - Isochronal router: full isochrone expansion plus an optional optimization pass that constrains a second search around the first route to refine it.

- **Search sector and sampling**
  - QtVLM router: samples headings in a sector around the goal bearing using `headings_spread` and `heading_step_deg`.
  - Isochronal router: similar sector approach, but prunes with additional strategies (`prune_equidistant`, `prune_close_together`, step-behind pruning during the optimization pass).

- **Finish handling**
  - Both: compute great-circle intersection with a finish circle and clip the leg precisely when it hits the finish.

- **Land/sea crossing checks**
  - Both: support `avoid_land_crossings` in modes `point`, `step`, `strict`, all delegated to the land–sea mask utilities.
  - Isochronal router: caps checks to `step` during the main routing and allows `strict` in the optimization pass (for accuracy near land).
  - QtVLM router: uses the chosen mode for the single routing pass (choose `strict` when you need maximum safety near coasts; it will be slower).

- **Progress and streaming**
  - `deploy/modal/weather_routing_server.py` (isochronal): streams progress for the main route, then for the optimization pass, and emits a final `"result"`. May also emit a preliminary optimized route snapshot.
  - `deploy/modal/weather_routing_server_qtvlm.py` (qtvlm): streams progress for the single routing pass and emits a single `"initial"` route payload (no separate optimization stage in v1).

### Key parameters you’ll see

- QtVLM router server (`deploy/modal/weather_routing_server_qtvlm.py`):
  - Adds `use_refine`, `heading_step_deg`, `headings_spread` to tune sector sampling and (future) refinement behavior.
  - Shares existing controls like `avoid_land_crossings`, `leg_check_spacing_nm`, `finish_size`, `tack_penalty`, `twa_change_penalty`, `twa_change_threshold`.

- Isochronal router server (`deploy/modal/weather_routing_server.py`):
  - Includes optimization-specific controls, e.g. `optimise_n_points`, `optimise_window`, and a richer set of pruning operations.

### Expected differences in results

- **Route smoothness and stability**
  - Isochronal router’s optimization pass tends to produce smoother, more stable routes, especially in variable winds, because the second pass prunes and focuses around a prior feasible route.
  - QtVLM router can be more “greedy” because it does not run a second constrained pass; route choice depends more directly on sector sampling density and the bucket/wake pruning behavior.

- **Narrow passages and coastal detail**
  - With `strict` checks in the optimization pass, the isochronal router is generally more reliable at avoiding subtle land crossings near shorelines.
  - The QtVLM router can run in `strict` mode too, but doing so for the entire single pass may be slower. In `step` mode it samples a path sparsely and can miss very thin crossings.

- **Arrival timing fidelity**
  - Isochronal router computes precise finish intersection and also accounts for tack and TWA-change penalties along the way; the optimization step typically improves arrival fidelity.
  - The QtVLM router v1 applies the same penalties and precise finish intersection, but it does not include a separate refinement/optimization stage. In complex wind fields this can yield small differences in arrival location/time compared to the optimized route.

### Performance characteristics

- **Which is faster?**
  - QtVLM router is typically faster to a complete answer because it runs a single pass and uses simpler pruning.
  - Isochronal router is often slower overall because it performs an additional optimization pass and more advanced pruning steps; however, those prunings can also reduce frontier sizes during the pass.

- **Which is likely most accurate?**
  - Isochronal router, especially when the optimization pass is enabled and `avoid_land_crossings='strict'` near coasts, is more likely to find robust, higher-quality routes in tricky conditions.
  - QtVLM router is usually “good enough” for quick previews and broad-scale offshore routing. For final routes and close-quarters coastal work, prefer the isochronal router with optimization.

### Practical guidance

- **Use QtVLM router when**:
  - You want a quick route preview or an interactive, lower-latency experience.
  - You are offshore (far from land) where sparse leg checks and single-pass behavior are adequate.

- **Use Isochronal router when**:
  - You want the most accurate/robust route and can afford extra compute time.
  - You need better behavior near coastlines or in highly variable wind fields.

### File references

- QtVLM router: `src/weather_router/qtvlm_weather_router.py`
- QtVLM server: `deploy/modal/weather_routing_server_qtvlm.py`
- Isochronal router: `src/weather_router/isochronal_weather_router.py`
- Isochronal server: `deploy/modal/weather_routing_server.py`


