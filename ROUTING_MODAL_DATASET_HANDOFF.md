# Routing Modal Dataset Handoff

This handoff gives the routing Modal team the boilerplate and edit instructions needed to make the routing SSE service dataset-aware. The frontend now sends `dataset_id` with routing requests, matching the weather map dataset selector.

The goal is one routing deployment that can serve both:

- `gfs-dynamical`
- `ecmwf-aifs-single`

GFS must remain the default for legacy clients that omit `dataset_id`.

## Required Behavior

- Add an optional `dataset_id` query parameter to the routing endpoint.
- Treat missing or blank `dataset_id` as `gfs-dynamical`.
- Reject unknown dataset ids with HTTP `400` JSON.
- Do not silently fall back from an unknown dataset to GFS.
- Use the same Icechunk physical stores as the weather API.
- Read `wind_u_10m` and `wind_v_10m` from the selected dataset.
- Interpret `lead_time_start` as lead hours, not as a raw array index.
- Include `X-Weather-Dataset-Id` on successful streaming responses and structured dataset errors.

Supported dataset stores:

| Dataset id | Bucket | Prefix | Region | Branch | Routing frequencies |
|---|---|---|---|---|---|
| `gfs-dynamical` | `dynamical-noaa-gfs` | `noaa-gfs-forecast/v0.2.7.icechunk/` | `us-west-2` | `main` | `1hr`, `3hr` |
| `ecmwf-aifs-single` | `dynamical-ecmwf-aifs-single` | `ecmwf-aifs-single-forecast/v0.1.0.icechunk/` | `us-west-2` | `main` | `6hr` |

## Dependency Change

Add `icechunk` to the Modal image:

```python
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .env({"FORCE_BUILD": "20260330"})
    .uv_pip_install(
        "xarray[complete]>=2025.1.2",
        "zarr>=3.0.8",
        "icechunk",
        "numpy",
        "pandas",
        "numba",
        "geopy",
        "fsspec",
        "s3fs",
        "fastapi",
        "git+https://github.com/peterm790/weather_routing",
    )
)
```

## Dataset Registry Boilerplate

Place this near the top of the routing service module, outside `get_route`, so Icechunk handles are reused across requests in a warm Modal container.

```python
from dataclasses import dataclass
from threading import Lock
from typing import Optional


DEFAULT_DATASET_ID = "gfs-dynamical"


@dataclass(frozen=True)
class RoutingDatasetConfig:
    dataset_id: str
    bucket: str
    prefix: str
    region: str
    branch: str
    allowed_freqs: tuple[str, ...]


DATASET_REGISTRY = {
    "gfs-dynamical": RoutingDatasetConfig(
        dataset_id="gfs-dynamical",
        bucket="dynamical-noaa-gfs",
        prefix="noaa-gfs-forecast/v0.2.7.icechunk/",
        region="us-west-2",
        branch="main",
        allowed_freqs=("1hr", "3hr"),
    ),
    "ecmwf-aifs-single": RoutingDatasetConfig(
        dataset_id="ecmwf-aifs-single",
        bucket="dynamical-ecmwf-aifs-single",
        prefix="ecmwf-aifs-single-forecast/v0.1.0.icechunk/",
        region="us-west-2",
        branch="main",
        allowed_freqs=("6hr",),
    ),
}


_DATASET_HANDLES = {}
_DATASET_HANDLES_LOCK = Lock()


def normalize_dataset_id(dataset_id: Optional[str]) -> str:
    if isinstance(dataset_id, str) and dataset_id.strip():
        return dataset_id.strip()
    return DEFAULT_DATASET_ID


def invalid_dataset_response(dataset_id: str):
    from fastapi.responses import JSONResponse

    return JSONResponse(
        status_code=400,
        content={
            "error": "invalid_dataset",
            "message": f"Unknown dataset_id: {dataset_id}",
            "dataset_id": dataset_id,
            "supported_dataset_ids": list(DATASET_REGISTRY.keys()),
        },
        headers={"X-Weather-Dataset-Id": dataset_id or DEFAULT_DATASET_ID},
    )


def open_weather_dataset(dataset_id: str):
    import icechunk
    import xarray as xr

    config = DATASET_REGISTRY[dataset_id]

    with _DATASET_HANDLES_LOCK:
        cached = _DATASET_HANDLES.get(dataset_id)
        if cached is not None:
            return cached["ds"]

        storage = icechunk.s3_storage(
            bucket=config.bucket,
            prefix=config.prefix,
            region=config.region,
            anonymous=True,
        )
        repo = icechunk.Repository.open(storage)
        session = repo.readonly_session(config.branch)
        ds = xr.open_zarr(session.store, chunks=None, decode_timedelta=True)

        _DATASET_HANDLES[dataset_id] = {
            "repo": repo,
            "session": session,
            "ds": ds,
        }
        return ds
```

## Lead-Time Mapping Boilerplate

The frontend sends `lead_time_start` as lead hours. The routing service must convert those hours into the selected dataset's `lead_time` indices.

GFS mapping:

- `0..120h` are hourly and map to indices `0..120`.
- `123..384h` are 3-hourly and map to indices `121..208`.
- `121h` and `122h` are not valid GFS lead hours.

AIFS mapping:

- `0..360h` are 6-hourly and map to indices `0..60`.

```python
def gfs_hour_to_timestep_index(hour: int) -> int:
    if hour < 0:
        raise ValueError(f"Invalid GFS lead hour: {hour}")
    if hour > 384:
        raise ValueError("GFS lead hour exceeds dataset range: max 384h")
    if hour <= 120:
        return hour
    if hour < 123:
        raise ValueError("GFS lead hour 121h/122h is unavailable")
    if hour % 3 != 0:
        raise ValueError("GFS extended lead hours must be 3-hour multiples")
    return 120 + ((hour - 120) // 3)


def aifs_hour_to_timestep_index(hour: int) -> int:
    if hour < 0:
        raise ValueError(f"Invalid AIFS lead hour: {hour}")
    if hour > 360:
        raise ValueError("AIFS lead hour exceeds dataset range: max 360h")
    if hour % 6 != 0:
        raise ValueError("AIFS lead hours must be 6-hour multiples")
    return hour // 6


def routing_lead_indices(
    dataset_id: str,
    freq: str,
    lead_time_start: int,
    lead_time_size: int,
) -> list[int]:
    if dataset_id == "gfs-dynamical":
        if freq == "1hr":
            start_index = gfs_hour_to_timestep_index(lead_time_start)
            if start_index > 120:
                raise ValueError("freq='1hr' only supports GFS lead hours 0..120")
            return list(range(start_index, min(121, lead_time_size)))

        if freq == "3hr":
            lead_hours = list(range(0, 121, 3)) + list(range(123, 385, 3))
            selected_hours = [hour for hour in lead_hours if hour >= lead_time_start]
            return [
                gfs_hour_to_timestep_index(hour)
                for hour in selected_hours
                if gfs_hour_to_timestep_index(hour) < lead_time_size
            ]

        raise ValueError("GFS freq must be '1hr' or '3hr'")

    if dataset_id == "ecmwf-aifs-single":
        if freq != "6hr":
            raise ValueError("AIFS freq must be '6hr'")
        start_index = aifs_hour_to_timestep_index(lead_time_start)
        return list(range(start_index, min(61, lead_time_size)))

    raise ValueError(f"Unknown dataset_id: {dataset_id}")
```

## Endpoint Signature Change

Add `dataset_id` near the time parameters:

```python
def get_route(
    start_lat: float,
    start_lon: float,
    end_lat: float,
    end_lon: float,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    init_time: int = -1,
    lead_time_start: int = 0,
    freq: str = "1hr",
    dataset_id: str = DEFAULT_DATASET_ID,
    ...
):
```

Add it to the request logging:

```python
print(f"  dataset_id='{dataset_id}'")
```

Then normalize and validate it before validating `freq`:

```python
dataset_id = normalize_dataset_id(dataset_id)
if dataset_id not in DATASET_REGISTRY:
    return invalid_dataset_response(dataset_id)

dataset_config = DATASET_REGISTRY[dataset_id]

if freq not in dataset_config.allowed_freqs:
    from fastapi import Response

    return Response(
        content=(
            f"freq must be one of {dataset_config.allowed_freqs} "
            f"for dataset_id='{dataset_id}'"
        ),
        status_code=400,
        headers={"X-Weather-Dataset-Id": dataset_id},
    )
```

## Replace Weather Loading

Replace the hard-coded GFS URL:

```python
ds = xr.open_zarr(
    "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
    decode_timedelta=True,
)
```

with the selected Icechunk dataset:

```python
print(f"Opening weather dataset {dataset_id}")
ds = open_weather_dataset(dataset_id)
```

Keep the shared wind field selection:

```python
ds = ds.rename({"latitude": "lat", "longitude": "lon"})
ds = ds[["wind_u_10m", "wind_v_10m"]]
```

Keep the spatial and init slicing:

```python
ds = ds.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
ds = ds.isel(init_time=init_time)
```

Then replace the existing `if freq == "1hr"` / `elif freq == "3hr"` lead slicing block with:

```python
try:
    lead_indices = routing_lead_indices(
        dataset_id=dataset_id,
        freq=freq,
        lead_time_start=int(lead_time_start),
        lead_time_size=ds.sizes["lead_time"],
    )
except ValueError as exc:
    from fastapi import Response

    return Response(
        content=str(exc),
        status_code=400,
        headers={"X-Weather-Dataset-Id": dataset_id},
    )

if not lead_indices:
    from fastapi import Response

    return Response(
        content="No lead_time values available for the requested dataset/frequency/start lead",
        status_code=400,
        headers={"X-Weather-Dataset-Id": dataset_id},
    )

ds = ds.isel(lead_time=lead_indices)
```

Everything after `ds.load()` can stay structurally the same:

```python
ds = ds.load()
ds = ds.assign_coords(time=ds.init_time + ds.lead_time)
ds = ds.swap_dims({"lead_time": "time"})
```

## Streaming Response Header

Add the selected dataset id to the stream response:

```python
return StreamingResponse(
    generate(),
    media_type="text/event-stream; charset=utf-8",
    headers={
        "Cache-Control": "no-cache, no-transform",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
        "X-Content-Type-Options": "nosniff",
        "X-Weather-Dataset-Id": dataset_id,
    },
)
```

## Important Gotchas

- Do not keep using `https://data.dynamical.org/noaa/gfs/forecast/latest.zarr` once `dataset_id` is supported. That URL can only represent GFS.
- Do not treat `lead_time_start` as a raw `lead_time` index. The frontend sends lead hours.
- Do not let AIFS use `freq=1hr` or `freq=3hr`; AIFS routing requests should use `freq=6hr`.
- Do not let GFS use `freq=6hr`; GFS frontend views currently send `1hr` or `3hr`.
- Both datasets use `wind_u_10m` and `wind_v_10m`, so the downstream wind-speed and wind-direction code should not need dataset-specific variable names.
- The land-mask path and routing engine setup can remain unchanged.

## Smoke Checks

Use the deployed Modal URL in place of `$ROUTING_URL`.

GFS regression request:

```bash
curl -N "$ROUTING_URL?dataset_id=gfs-dynamical&start_lat=37&start_lon=-122&end_lat=34&end_lon=-118&min_lat=30&min_lon=-130&max_lat=45&max_lon=-110&init_time=-1&lead_time_start=0&freq=1hr&optimise_max_passes=0"
```

AIFS request:

```bash
curl -N "$ROUTING_URL?dataset_id=ecmwf-aifs-single&start_lat=37&start_lon=-122&end_lat=34&end_lon=-118&min_lat=30&min_lon=-130&max_lat=45&max_lon=-110&init_time=-1&lead_time_start=0&freq=6hr&optimise_max_passes=0"
```

Invalid dataset:

```bash
curl -i "$ROUTING_URL?dataset_id=not-real&start_lat=37&start_lon=-122&end_lat=34&end_lon=-118&min_lat=30&min_lon=-130&max_lat=45&max_lon=-110"
```

Expected: HTTP `400` with:

```json
{
  "error": "invalid_dataset",
  "message": "Unknown dataset_id: not-real",
  "dataset_id": "not-real",
  "supported_dataset_ids": ["gfs-dynamical", "ecmwf-aifs-single"]
}
```

AIFS invalid frequency:

```bash
curl -i "$ROUTING_URL?dataset_id=ecmwf-aifs-single&start_lat=37&start_lon=-122&end_lat=34&end_lon=-118&min_lat=30&min_lon=-130&max_lat=45&max_lon=-110&freq=1hr"
```

Expected: HTTP `400`, because AIFS routing should use `freq=6hr`.

## Frontend Contract Summary

The frontend now forwards:

- `dataset_id`: `gfs-dynamical` or `ecmwf-aifs-single`
- `init_time`: global init index, with `-1` meaning latest
- `lead_time_start`: lead hour from the active time slider
- `freq`: active view cadence, one of `1hr`, `3hr`, `6hr`

The routing Modal service should resolve those values against the selected dataset, not against a hard-coded GFS store.
