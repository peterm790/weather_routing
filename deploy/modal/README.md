# Weather Routing API

Deploys a weather routing service on Modal.

## Deploy

```bash
modal deploy weather_routing_server.py
```

## Usage

Endpoint: `GET /get_route`

### Parameters

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `start_lat` | float | - | Start latitude |
| `start_lon` | float | - | Start longitude |
| `end_lat` | float | - | End latitude |
| `end_lon` | float | - | End longitude |
| `min_lat` | float | - | Bounding box min latitude |
| `min_lon` | float | - | Bounding box min longitude |
| `max_lat` | float | - | Bounding box max latitude |
| `max_lon` | float | - | Bounding box max longitude |
| `init_time` | int | -1 | Forecast initialization index (-1 for latest) |
| `lead_time_start` | int | 0 | Start index for forecast lead time slice |
| `provider` | str | "dynamical" | Weather provider namespace. Currently only "dynamical" is supported. |
| `dataset_id` | str | "gfs" | Weather dataset id. Use "gfs" for GFS or "aifs" for AIFS. |
| `freq` | str | "1hr" | Time step frequency. GFS supports "1hr" and "3hr"; AIFS supports "6hr". |
| `polar_file` | str | "volvo70" | Name of polar file (e.g. "volvo70") |
| `optimise_max_passes` | int | 0 | Max optimization passes (`0` skips optimization stage) |
| `use_equidistant_pruning` | bool or null | null | Enable/disable equidistant pruning. If null: defaults to `False` when `optimise_max_passes=0`, else `True`. |

### Weather Sources

New clients should send one of these canonical source shapes:

```json
{ "provider": "dynamical", "dataset_id": "gfs" }
```

```json
{ "provider": "dynamical", "dataset_id": "aifs" }
```

For query strings, use `provider=dynamical&dataset_id=gfs` or `provider=dynamical&dataset_id=aifs`.

Older clients may omit `provider` and `dataset_id`; the API defaults them to `dynamical` and `gfs`. Temporary legacy dataset aliases are accepted inbound for rollout compatibility, but responses, logs, stream metadata, and generated request examples use the canonical values above.

### Example Request

You can test directly in your browser or via curl:

```bash
curl -X GET "https://peterm790--weather-routing-get-route.modal.run?provider=dynamical&dataset_id=gfs&start_lat=43.0&start_lon=5.0&end_lat=40.0&end_lon=8.0&min_lat=39.0&min_lon=4.0&max_lat=44.0&max_lon=9.0&init_time=-1&lead_time_start=0&freq=1hr&polar_file=volvo70"
```

AIFS example:

```bash
curl -X GET "https://peterm790--weather-routing-get-route.modal.run?provider=dynamical&dataset_id=aifs&start_lat=43.0&start_lon=5.0&end_lat=40.0&end_lon=8.0&min_lat=39.0&min_lon=4.0&max_lat=44.0&max_lon=9.0&init_time=-1&lead_time_start=0&freq=6hr&polar_file=volvo70"
```

### Response Format

The API returns Server-Sent Events. Each `data:` frame contains a single JSON object. The sequence of messages is:

1. `progress`: Real-time updates on the initial routing algorithm's progress.
2. `initial`: The initial route calculated before optimization.
3. `progress`: Real-time updates on the optimization pass (only when `optimise_max_passes > 0`).
4. `result`: Final route. If optimization is disabled, this mirrors the initial route.

Each message has the following structure:

**Progress Message:**
```json
{
  "type": "progress",
  "metadata": {"provider": "dynamical", "dataset_id": "gfs", "dataset_name": "GFS", "request_fingerprint": "..."},
  "step": 5,
  "dist": 120.5,
  "isochrones": [[-34.0, 18.0], [-34.1, 18.1], ...]
}
```

**Initial Route Message:**
```json
{"type": "initial", "metadata": {"provider": "dynamical", "dataset_id": "gfs", "dataset_name": "GFS", "request_fingerprint": "..."}, "data": [...route points...]}
```

**Result Message:**
```json
{"type": "result", "metadata": {"provider": "dynamical", "dataset_id": "gfs", "dataset_name": "GFS", "request_fingerprint": "..."}, "data": [...route points...]}
```

Validation errors for unsupported providers or dataset ids return HTTP `400` with a structured JSON body such as:

```json
{
  "error": "invalid_dataset_id",
  "message": "Unknown dataset_id: not-real",
  "field": "dataset_id",
  "dataset_id": "not-real",
  "supported_dataset_ids": ["gfs", "aifs"]
}
```

Example usage with Python:

```python
import requests
import json

url = "..."
with requests.get(url, stream=True) as r:
    for line in r.iter_lines():
        if line:
            msg = json.loads(line)
            if msg['type'] == 'progress':
                print(f"Step: {msg['step']}, Distance: {msg['dist']}")
            elif msg['type'] == 'initial':
                print("Received initial route")
            elif msg['type'] == 'result':
                print("Received final route")
```

Each route point object contains:

```json
{
  "lat": -34.0,
  "lon": 18.0,
  ...
}
```

### Field Descriptions

- `lat`, `lon`: Position of the boat at this time step.
- `time`: Timestamp of the position (ISO 8601 string).
- `twd`: True Wind Direction (degrees).
- `tws`: True Wind Speed (knots).
- `pos`: Current position tuple `(lat, lon)`.
- `next_pos`: Target position tuple for the next time step `(lat, lon)`.
- `heading`: Boat heading (degrees).
- `twa`: True Wind Angle (degrees relative to boat).
- `base_boat_speed`: Theoretical speed from polar diagram (knots).
- `is_tacking`: Boolean indicating if a tack maneuver is required.
- `boat_speed`: Actual speed over ground (knots), includes penalties if tacking.
- `hours_elapsed`: Hours since start.
- `days_elapsed`: Days since start.
