# Weather Routing API

Deploys a weather routing service on Modal.

## Deploy

```bash
modal deploy weather_routing_server.py
```

## Usage

Endpoint: `POST /get_route`

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

### Example Request

```python
import requests
import numpy as np
import io


url = "https://peterm790--weather-routing-get-route.modal.run"

params = {
    "start_lat": 43.0,
    "start_lon": 5.0,
    "end_lat": 40.0,
    "end_lon": 8.0,
    "min_lat": 39.0,
    "min_lon": 4.0,
    "max_lat": 44.0,
    "max_lon": 9.0,
    "init_time": -1,
    "lead_time_start": 0
}

response = requests.post(url, params=params)
arr = np.load(io.BytesIO(response.content))
print(arr)
```

### Quick Test URL

You can test directly in your browser or via curl with this URL (adjust parameters as needed):

```
https://peterm790--weather-routing-get-route.modal.run?start_lat=43.0&start_lon=5.0&end_lat=40.0&end_lon=8.0&min_lat=39.0&min_lon=4.0&max_lat=44.0&max_lon=9.0&init_time=-1&lead_time_start=0
```

