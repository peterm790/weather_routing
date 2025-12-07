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
| `freq` | str | "1hr" | Time step frequency ("1hr" or "3hr") |
| `polar_file` | str | "volvo70" | Name of polar file (e.g. "volvo70") |

### Example Request

You can test directly in your browser or via curl:

```bash
curl -X GET "https://peterm790--weather-routing-get-route.modal.run?start_lat=43.0&start_lon=5.0&end_lat=40.0&end_lon=8.0&min_lat=39.0&min_lon=4.0&max_lat=44.0&max_lon=9.0&init_time=-1&lead_time_start=0&freq=1hr&polar_file=volvo70"
```

### Response Format

The API returns a stream of Newline Delimited JSON (NDJSON) objects. The sequence of messages is:

1. `progress`: Real-time updates on the initial routing algorithm's progress.
2. `initial`: The initial route calculated before optimization.
3. `progress`: Real-time updates on the optimization pass.
4. `result`: The final optimized route.

Each message has the following structure:

**Progress Message:**
```json
{
  "type": "progress", 
  "step": 5, 
  "dist": 120.5,
  "isochrones": [[-34.0, 18.0], [-34.1, 18.1], ...]
}
```

**Initial Route Message:**
```json
{"type": "initial", "data": [...route points...]}
```

**Result Message:**
```json
{"type": "result", "data": [...route points...]}
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
