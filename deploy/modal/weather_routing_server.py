import modal
import io
import urllib.request
from typing import Union
from functools import lru_cache

# Define the image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    # Add a timestamp or version to force cache invalidation when git repo changes
    .env({"FORCE_BUILD": "20260106_8"}) 
    .uv_pip_install(
        "xarray[complete]>=2025.1.2",
        "zarr>=3.0.8",
        "numpy",
        "pandas",
        "numba",
        "geopy",
        "fsspec",
        "s3fs",
        "fastapi",
        "git+https://github.com/peterm790/weather_routing"
    )
)

app = modal.App("weather-routing", image=image)

@app.function(timeout=1200)
@modal.fastapi_endpoint()
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
    crank_step: int = 30,
    avoid_land_crossings: Union[bool, str] = True,
    leg_check_spacing_nm: float = 2.0,
    polar_file: str = "volvo70",
    spread: int = 270,
    wake_lim: int = 15,
    rounding: int = 3,
    n_points: int = 30,
    tack_penalty: float = 0.5,
    finish_size: float = 5.0,
    optimise_n_points: int = 60,
    optimise_window: int = 24,
    leg_check_max_samples: int = 10,
    point_validity_method: str = "nearest",
    twa_change_penalty: float = 0.02,
    twa_change_threshold: float = 5.0
):
    print("query params:")
    print(f"  start_lat={start_lat}")
    print(f"  start_lon={start_lon}")
    print(f"  end_lat={end_lat}")
    print(f"  end_lon={end_lon}")
    print(f"  min_lat={min_lat}")
    print(f"  min_lon={min_lon}")
    print(f"  max_lat={max_lat}")
    print(f"  max_lon={max_lon}")
    print(f"  init_time={init_time}")
    print(f"  lead_time_start={lead_time_start}")
    print(f"  freq='{freq}'")
    print(f"  crank_step={crank_step}")
    print(f"  avoid_land_crossings={avoid_land_crossings}")
    print(f"  leg_check_spacing_nm={leg_check_spacing_nm}")
    print(f"  polar_file='{polar_file}'")
    print(f"  spread={spread}")
    print(f"  wake_lim={wake_lim}")
    print(f"  rounding={rounding}")
    print(f"  n_points={n_points}")
    print(f"  tack_penalty={tack_penalty}")
    print(f"  finish_size={finish_size}")
    print(f"  optimise_n_points={optimise_n_points}")
    print(f"  optimise_window={optimise_window}")
    print(f"  leg_check_max_samples={leg_check_max_samples}")
    print(f"  point_validity_method='{point_validity_method}'")
    print(f"  twa_change_penalty={twa_change_penalty}")
    print(f"  twa_change_threshold={twa_change_threshold}")

    from fastapi import Response
    from fastapi.responses import StreamingResponse
    import xarray as xr
    import numpy as np
    import json
    import queue
    import threading
    import time
    # Import here to ensure they are available in the container
    from weather_router import isochronal_weather_router, polar, point_validity
    from numba import njit

    if freq not in ["1hr", "3hr"]:
        return Response(content="freq must be '1hr' or '3hr'", status_code=400)

    if crank_step <= 0:
        return Response(content="crank_step must be a positive integer (minutes)", status_code=400)

    if leg_check_spacing_nm < 0.25:
        return Response(content="leg_check_spacing_nm must be >= 0.25 (nautical miles)", status_code=400)

    # Normalize/validate land-crossing mode.
    # Accepted: 'point', 'step', 'strict' (or bool for backward compatibility: False->point, True->step).
    if isinstance(avoid_land_crossings, bool):
        avoid_land_crossings_mode = 'step' if avoid_land_crossings else 'point'
    elif isinstance(avoid_land_crossings, str):
        avoid_land_crossings_mode = avoid_land_crossings.lower()
        if avoid_land_crossings_mode not in ("point", "step", "strict"):
            return Response(
                content="avoid_land_crossings must be one of: 'point', 'step', 'strict' (or bool)",
                status_code=400,
            )
    else:
        return Response(
            content="avoid_land_crossings must be one of: 'point', 'step', 'strict' (or bool)",
            status_code=400,
        )

    start_point = (start_lat, start_lon)
    end_point = (end_lat, end_lon)

    # Load weather data
    # Note: Using the URL from the original script
    print('Opening weather zarr')
    ds = xr.open_zarr(
        "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
        decode_timedelta=True,
    )

    ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    ds = ds[['wind_u_10m', 'wind_v_10m']]

    # Slice for the region - re-enabled with dynamic bounds
    # GFS latitude is typically 90 to -90. 
    # slice(max_lat, min_lat) covers this range if max > min.
    ds = ds.sel(lat=slice(max_lat, min_lat), lon=slice(min_lon, max_lon))
    ds = ds.isel(init_time=init_time)
    
    if freq == "1hr":
        crank_step = 60
        leg_check_spacing_nm = 3
        ds = ds.isel(lead_time=slice(lead_time_start, 120))
    elif freq == "3hr":
        leg_check_spacing_nm = 9
        crank_step = 180
        # Construct indices for 3-hourly sequence
        # Hourly part: 0 to 120 (inclusive) every 3 hours
        hourly_indices = list(range(0, 121, 3))
        # 3-hourly part: 121 to end
        three_hourly_indices = list(range(121, ds.sizes['lead_time']))
        indices = hourly_indices + three_hourly_indices
        
        ds = ds.isel(lead_time=indices)
        
        # Calculate new start index
        start_index = int(round(lead_time_start / 3))
        ds = ds.isel(lead_time=slice(start_index, None))

    ds = ds.load()

    ds = ds.assign_coords(time=ds.init_time + ds.lead_time)
    ds = ds.swap_dims({'lead_time': 'time'})

    ds = ds.drop_vars([
        'expected_forecast_length',
        'ingested_forecast_length',
        'init_time',
        'lead_time',
        'spatial_ref',
        'valid_time'
    ], errors='ignore')

    # Interpolate wind components to the finer crank cadence, then recompute TWS/TWD.
    # This avoids angular wrap issues that can occur when interpolating direction directly.
    if 'time' not in ds.dims or ds.sizes.get('time', 0) < 2:
        raise ValueError("Weather dataset has insufficient time steps to interpolate")

    time_vals = ds.time.values
    dt = np.timedelta64(int(crank_step), 'm')

    # Build a uniform time grid at crank_step minutes, ensuring the final original time is included.
    new_times = np.arange(time_vals[0], time_vals[-1], dt)
    if new_times.size == 0 or new_times[-1] != time_vals[-1]:
        new_times = np.concatenate([new_times, np.array([time_vals[-1]], dtype=time_vals.dtype)])

    ds_interp = ds[['wind_u_10m', 'wind_v_10m']].interp(time=new_times)

    u10 = ds_interp['wind_u_10m']
    v10 = ds_interp['wind_v_10m']
    tws = np.hypot(v10, u10)
    tws = tws * 1.94384  # convert m/s to knots
    twd = (180 + np.rad2deg(np.arctan2(u10, v10))) % 360
    
    ds_processed = tws.to_dataset(name='tws')
    ds_processed['twd'] = twd

    # Fast nearest-index helpers over 1D sorted coords (ascending or descending)
    lat_vals = ds_processed['lat'].values
    lon_vals = ds_processed['lon'].values
    time_vals = ds_processed['time'].values
    tws_arr = ds_processed['tws'].values  # shape: (time, lat, lon)
    twd_arr = ds_processed['twd'].values  # shape: (time, lat, lon)

    # Prepare numeric views for Numba-friendly comparisons
    lat_vals_f64 = lat_vals.astype('float64')
    lon_vals_f64 = lon_vals.astype('float64')
    time_vals_ns = time_vals.astype('datetime64[ns]').astype('int64')
    lat_asc = bool(lat_vals_f64[0] <= lat_vals_f64[-1])
    lon_asc = bool(lon_vals_f64[0] <= lon_vals_f64[-1])
    lon_min = float(lon_vals_f64.min())
    lon_max = float(lon_vals_f64.max())

    @njit(cache=True)
    def _nearest_index_numba(vals, x, asc):
        # Binary search insertion point then choose nearest neighbor
        left = 0
        right = vals.size
        if asc:
            while left < right:
                mid = (left + right) // 2
                if vals[mid] < x:
                    left = mid + 1
                else:
                    right = mid
        else:
            while left < right:
                mid = (left + right) // 2
                if vals[mid] > x:
                    left = mid + 1
                else:
                    right = mid
        i = left
        if i == 0:
            return 0
        n = vals.size
        if i >= n:
            return n - 1
        dr = abs(vals[i] - x)
        dl = abs(x - vals[i - 1])
        return i if dr <= dl else i - 1

    @njit(cache=True)
    def _wrap_lon_to_domain_numba(lon, lon_min_v, lon_max_v):
        width = lon_max_v - lon_min_v
        if width <= 0.0:
            return lon
        # normalize by 360 wrap
        lon_rel = lon - lon_min_v
        # Python-style modulo works in numba for floats
        wrapped_rel = lon_rel % 360.0
        wrapped = lon_min_v + wrapped_rel
        if wrapped < lon_min_v:
            wrapped = lon_min_v
        if wrapped > lon_max_v:
            wrapped = lon_max_v
        return wrapped

    @lru_cache(maxsize=16384)
    def _cached_wind_by_index(ti: int, yi: int, xi: int):
        return (np.float32(twd_arr[ti, yi, xi]), np.float32(tws_arr[ti, yi, xi]))

    # Define wind callback (nearest-neighbour via integer indexing)
    def get_wind(t, lat, lon):
        # Convert numpy.datetime64 to int64 ns for numba
        t_ns = np.int64(np.datetime64(t, 'ns').astype('int64'))
        ti = int(_nearest_index_numba(time_vals_ns, t_ns, True))
        yi = int(_nearest_index_numba(lat_vals_f64, float(lat), lat_asc))
        wl = float(_wrap_lon_to_domain_numba(float(lon), lon_min, lon_max))
        xi = int(_nearest_index_numba(lon_vals_f64, wl, lon_asc))
        return _cached_wind_by_index(ti, yi, xi)

    # Load Polar
    print('fetching polar')
    url = f"https://data.offshoreweatherrouting.com/polars/{polar_file}.pol"
    with urllib.request.urlopen(url) as response:
        polar_data = response.read().decode('utf-8')

    volvo70_polar = polar.Polar(f=io.StringIO(polar_data))

    # Download land-sea mask
    print('loading lsm zarr')
    ds_lsm = xr.open_zarr('https://data.offshoreweatherrouting.com/GEBCO_2025_land_mask_sharded.zarr',
                                consolidated=True)
                                #storage_options={"anon": True})
    ds_lsm = ds_lsm.rename({'lat':'latitude', 'lon':'longitude'})
    ds_lsm = ds_lsm.sortby('latitude', ascending = False)
    ds_lsm = ds_lsm.fillna(0)
    lat1,lon1,lat2,lon2 = [min_lat, min_lon, max_lat, max_lon]
    ds_lsm = ds_lsm.sel(latitude = slice(max([lat1, lat2]),min([lat1, lat2]))).sel(longitude = slice(min([lon1, lon2]),max([lon1, lon2])))
    ds_lsm = ds_lsm.load()

    #import os
    #mask_url = "https://peterm790.s3.af-south-1.amazonaws.com/era5_land-sea-mask.nc"
    #mask_file = "/tmp/era5_land-sea-mask.nc"
    
    #if not os.path.exists(mask_file):
    #    print(f"Downloading mask from {mask_url}...")
    #    with urllib.request.urlopen(mask_url) as response:
    #        with open(mask_file, 'wb') as out_file:
    #            out_file.write(response.read())

    # Initialize Router
    step_val = float(crank_step) / 60.0
    print('creating routing queue')
    progress_queue = queue.Queue()

    def progress_callback(step, dist_wp, isochrones, pass_idx=None, emit_prelim=False):
        # Simplify isochrones for transport - extract only lat/lon
        # Assuming isochrones is a list of [lat, lon, ...]
        simple_isochrones = []
        if isochrones is not None:
            # Extract just lat/lon from the possible points
            # Each point in 'possible' is [lat, lon, route, bearing_end, twa]
            simple_isochrones = [[float(p[0]), float(p[1])] for p in isochrones]

        progress_queue.put({
            "type": "progress",
            "step": step,
            "dist": float(dist_wp),
            "isochrones": simple_isochrones,
            "pass_idx": pass_idx
        })

        # If requested, emit a single preliminary route snapshot (end-of-pass)
        if emit_prelim:
            try:
                prelim_route = weatherrouter.get_fastest_route(stats=False, use_optimized=True)
                if isinstance(prelim_route, list):
                    prelim_route_data = prelim_route
                else:
                    prelim_route_copy = prelim_route.copy()
                    if 'time' in prelim_route_copy.columns:
                        prelim_route_copy['time'] = prelim_route_copy['time'].astype(str)
                    prelim_route_data = prelim_route_copy.to_dict(orient="records")
                progress_queue.put({
                    "type": "result",
                    "preliminary": True,
                    "pass_idx": pass_idx,
                    "step": step,
                    "data": prelim_route_data
                })
            except Exception as e:
                # Surface an explicit error object to the stream; no silent fallbacks
                progress_queue.put({
                    "type": "error",
                    "scope": "preliminary_result_emit",
                    "message": f"{type(e).__name__}: {str(e)}"
                })

    weatherrouter = isochronal_weather_router.weather_router(
        volvo70_polar,
        get_wind,
        time_steps=ds_processed.time.values,
        step=step_val,
        start_point=start_point,
        end_point=end_point,
        point_validity_extent=[min_lat, min_lon, max_lat, max_lon],
        point_validity_file=ds_lsm,
        avoid_land_crossings=avoid_land_crossings_mode,
        leg_check_spacing_nm=leg_check_spacing_nm,
        spread=spread,
        wake_lim=wake_lim,
        rounding=rounding,
        n_points=n_points,
        optimise_n_points=optimise_n_points,
        progress_callback=progress_callback,
        finish_size=finish_size,
        tack_penalty=tack_penalty,
        optimise_window=optimise_window,
        leg_check_max_samples=leg_check_max_samples,
        point_validity_method=point_validity_method,
        twa_change_penalty=twa_change_penalty,
        twa_change_threshold=twa_change_threshold
    )

    # Run Routing
    def generate():
        """
        Server-Sent Events (EventSource) stream compatible with `eventSource.onmessage`.

        We intentionally do NOT emit `event: ...` lines, because the client is using
        `onmessage` (default "message" event) and expects:
          data: <json>\n\n
        """

        def sse_data(data_obj) -> str:
            # Emit a single-line JSON payload as one SSE "message" event.
            # (Avoid multi-line JSON; EventSource concatenates multiple data lines with "\n".)
            payload = json.dumps(data_obj, ensure_ascii=False, separators=(",", ":"))
            return f"data: {payload}\n\n"

        # Start routing in a separate thread
        routing_thread = threading.Thread(target=weatherrouter.route)
        routing_thread.start()
        
        # Consume progress updates while routing is running
        last_keepalive = time.monotonic()
        KEEPALIVE_SECONDS = 15.0
        while routing_thread.is_alive():
            try:
                # Wait for progress or timeout
                progress = progress_queue.get(timeout=0.1)
                last_keepalive = time.monotonic()
                yield sse_data(progress)
            except queue.Empty:
                now = time.monotonic()
                if now - last_keepalive >= KEEPALIVE_SECONDS:
                    # SSE comment as keepalive (ignored by EventSource, but keeps proxies/LBs happy)
                    last_keepalive = now
                    yield ": keep-alive\n\n"
                continue
        
        routing_thread.join()

        # Yield any remaining progress messages from routing
        while not progress_queue.empty():
            try:
                progress = progress_queue.get_nowait()
                yield sse_data(progress)
            except queue.Empty:
                break

        initial_route = weatherrouter.get_fastest_route(stats=False)
        
        # Yield initial route
        if isinstance(initial_route, list):
            initial_route_data = initial_route
        else:
            initial_route_copy = initial_route.copy()
            if 'time' in initial_route_copy.columns:
                initial_route_copy['time'] = initial_route_copy['time'].astype(str)
            initial_route_data = initial_route_copy.to_dict(orient="records")
        
        initial_msg = {
            "type": "initial",
            "data": initial_route_data
        }
        yield sse_data(initial_msg)

        initial_isochrones = weatherrouter.get_isochrones()
        
        # Start optimization in a separate thread to stream progress
        optimize_thread = threading.Thread(target=weatherrouter.optimize, args=(initial_route, initial_isochrones))
        optimize_thread.start()

        # Consume progress updates while optimization is running
        last_keepalive = time.monotonic()
        while optimize_thread.is_alive():
            try:
                # Wait for progress or timeout
                progress = progress_queue.get(timeout=0.1)
                # Mark optimization progress differently if needed, or reuse 'progress' type
                # Here we reuse 'progress' type, but you could add a 'stage': 'optimization' field if desired
                # For now, we assume client handles step/dist same way
                last_keepalive = time.monotonic()
                yield sse_data(progress)
            except queue.Empty:
                now = time.monotonic()
                if now - last_keepalive >= KEEPALIVE_SECONDS:
                    last_keepalive = now
                    yield ": keep-alive\n\n"
                continue
        
        optimize_thread.join()

        # Yield any remaining progress messages from optimization
        while not progress_queue.empty():
            try:
                progress = progress_queue.get_nowait()
                yield sse_data(progress)
            except queue.Empty:
                break

        route_df = weatherrouter.get_fastest_route(use_optimized=True)
        
        # Process and yield final route
        if isinstance(route_df, list):
             route_data = route_df
        else:
            if 'time' in route_df.columns:
                route_df['time'] = route_df['time'].astype(str)
            route_data = route_df.to_dict(orient="records")
            
        result_msg = {
            "type": "result", 
            "preliminary": False,
            "data": route_data
        }
        yield sse_data(result_msg)

    return StreamingResponse(
        generate(), 
        media_type="text/event-stream; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        }
    )
