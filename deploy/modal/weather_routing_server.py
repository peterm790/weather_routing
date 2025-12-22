import modal
import io
import urllib.request

# Define the image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    # Add a timestamp or version to force cache invalidation when git repo changes
    .env({"FORCE_BUILD": "20251220_1"}) 
    .uv_pip_install(
        "xarray[complete]>=2025.1.2",
        "zarr>=3.0.8",
        "numpy",
        "pandas",
        "geopy",
        "fsspec",
        "s3fs",
        "fastapi",
        "git+https://github.com/peterm790/weather_routing"
    )
)

app = modal.App("weather-routing", image=image)

@app.function(timeout=600)
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
    avoid_land_crossings: bool = True,
    leg_check_spacing_nm: float = 2.0,
    polar_file: str = "volvo70"
):
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

    if freq not in ["1hr", "3hr"]:
        return Response(content="freq must be '1hr' or '3hr'", status_code=400)

    if crank_step <= 0:
        return Response(content="crank_step must be a positive integer (minutes)", status_code=400)

    if leg_check_spacing_nm < 0.25:
        return Response(content="leg_check_spacing_nm must be >= 0.25 (nautical miles)", status_code=400)

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

    # Define wind callback
    def get_wind(t, lat, lon):
        tws_sel = ds_processed.tws.sel(time=t, method='nearest')
        tws_sel = tws_sel.sel(lat=lat, lon=lon, method='nearest')
        twd_sel = ds_processed.twd.sel(time=t, method='nearest')
        twd_sel = twd_sel.sel(lat=lat, lon=lon, method='nearest')
        return (np.float32(twd_sel.values), np.float32(tws_sel.values))

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

    def progress_callback(step, dist_wp, isochrones):
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
            "isochrones": simple_isochrones
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
        avoid_land_crossings=avoid_land_crossings,
        leg_check_spacing_nm=leg_check_spacing_nm,
        spread=270,
        wake_lim=15,
        rounding=2,
        n_points=30,
        progress_callback=progress_callback,
        finish_size=5
    )

    # Run Routing
    def generate():
        # Chrome (and some intermediaries) can buffer small chunks.
        # We pad each NDJSON line to a minimum size to encourage progressive delivery.
        MIN_CHUNK_BYTES = 16 * 1024

        def ndjson_line_chunk(obj) -> str:
            payload = json.dumps(obj)  # keep existing JSON formatting
            payload_bytes = len(payload.encode("utf-8"))
            # Pad BEFORE newline so each yielded chunk is exactly one NDJSON line.
            if payload_bytes < (MIN_CHUNK_BYTES - 1):
                payload += " " * ((MIN_CHUNK_BYTES - 1) - payload_bytes)
            return payload + "\n"

        # Start routing in a separate thread
        routing_thread = threading.Thread(target=weatherrouter.route)
        routing_thread.start()
        
        # Consume progress updates while routing is running
        while routing_thread.is_alive():
            try:
                # Wait for progress or timeout
                progress = progress_queue.get(timeout=0.1)
                yield ndjson_line_chunk(progress)
            except queue.Empty:
                continue
        
        routing_thread.join()

        # Yield any remaining progress messages from routing
        while not progress_queue.empty():
            try:
                progress = progress_queue.get_nowait()
                yield ndjson_line_chunk(progress)
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
        yield ndjson_line_chunk(initial_msg)

        initial_isochrones = weatherrouter.get_isochrones()
        
        # Start optimization in a separate thread to stream progress
        optimize_thread = threading.Thread(target=weatherrouter.optimize, args=(initial_route, initial_isochrones))
        optimize_thread.start()

        # Consume progress updates while optimization is running
        while optimize_thread.is_alive():
            try:
                # Wait for progress or timeout
                progress = progress_queue.get(timeout=0.1)
                # Mark optimization progress differently if needed, or reuse 'progress' type
                # Here we reuse 'progress' type, but you could add a 'stage': 'optimization' field if desired
                # For now, we assume client handles step/dist same way
                yield ndjson_line_chunk(progress)
            except queue.Empty:
                continue
        
        optimize_thread.join()

        # Yield any remaining progress messages from optimization
        while not progress_queue.empty():
            try:
                progress = progress_queue.get_nowait()
                yield ndjson_line_chunk(progress)
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
            "data": route_data
        }
        yield ndjson_line_chunk(result_msg)

    return StreamingResponse(
        generate(), 
        media_type="application/x-ndjson; charset=utf-8",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Content-Type-Options": "nosniff",
        }
    )
