# Routing Weather Source Contract

This handoff describes the routing API weather-source contract shared by the frontend, weather API, routing API, worker, cache identity, and routing tools.

## Canonical Request Shape

New clients and tools should send `provider` first, then `dataset_id`:

```json
{ "provider": "dynamical", "dataset_id": "gfs" }
```

```json
{ "provider": "dynamical", "dataset_id": "aifs" }
```

For query strings:

```text
provider=dynamical&dataset_id=gfs
provider=dynamical&dataset_id=aifs
```

User-facing dataset labels are simply `GFS` and `AIFS`.

## Defaults And Validation

- Missing or blank `provider` defaults to `dynamical`.
- Missing or blank `dataset_id` defaults to `gfs`.
- Any provider other than `dynamical` returns a structured `400` validation error.
- Any dataset id other than `gfs`, `aifs`, or the temporary rollout aliases returns a structured `400` validation error.
- The temporary rollout aliases are accepted inbound only and normalize immediately to `gfs` or `aifs`.

Responses, logs, metadata, request fingerprints, generated examples, and tool output should use only canonical values:

```json
{
  "provider": "dynamical",
  "dataset_id": "gfs",
  "dataset_name": "GFS"
}
```

```json
{
  "provider": "dynamical",
  "dataset_id": "aifs",
  "dataset_name": "AIFS"
}
```

## Dataset Registry

| Provider | Dataset ID | Label | Bucket | Prefix | Region | Branch | Frequencies |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `dynamical` | `gfs` | `GFS` | `dynamical-noaa-gfs` | `noaa-gfs-forecast/v0.2.7.icechunk/` | `us-west-2` | `main` | `1hr`, `3hr` |
| `dynamical` | `aifs` | `AIFS` | `dynamical-ecmwf-aifs-single` | `ecmwf-aifs-single-forecast/v0.1.0.icechunk/` | `us-west-2` | `main` | `6hr` |

## Error Shapes

Unsupported provider:

```json
{
  "error": "invalid_provider",
  "message": "Unknown provider: static",
  "field": "provider",
  "provider": "static",
  "supported_providers": ["dynamical"]
}
```

Unsupported dataset:

```json
{
  "error": "invalid_dataset_id",
  "message": "Unknown dataset_id: not-real",
  "field": "dataset_id",
  "dataset_id": "not-real",
  "supported_dataset_ids": ["gfs", "aifs"]
}
```

## Example Requests

GFS:

```bash
curl -N "$ROUTING_URL?provider=dynamical&dataset_id=gfs&start_lat=37&start_lon=-122&end_lat=34&end_lon=-118&min_lat=30&min_lon=-130&max_lat=45&max_lon=-110&init_time=-1&lead_time_start=0&freq=1hr&optimise_max_passes=0"
```

AIFS:

```bash
curl -N "$ROUTING_URL?provider=dynamical&dataset_id=aifs&start_lat=37&start_lon=-122&end_lat=34&end_lon=-118&min_lat=30&min_lon=-130&max_lat=45&max_lon=-110&init_time=-1&lead_time_start=0&freq=6hr&optimise_max_passes=0"
```

The routing API includes canonical weather source metadata in stream messages:

```json
{
  "type": "result",
  "metadata": {
    "provider": "dynamical",
    "dataset_id": "gfs",
    "dataset_name": "GFS",
    "request_fingerprint": "..."
  },
  "preliminary": false,
  "data": []
}
```

## Implementation Notes

- Normalize `provider` and `dataset_id` once at the API boundary.
- Use `(provider, dataset_id)` in cache keys, request fingerprints, job payloads, result metadata, headers, and debug output anywhere weather source identity matters.
- Keep dataset-specific time-step mapping separate: GFS supports `1hr` and `3hr`; AIFS supports `6hr`.
- Do not generate new client or tool examples using the longer rollout alias values.
