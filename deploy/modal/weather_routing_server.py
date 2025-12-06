import modal
import io
import urllib.request

# Define the image with dependencies
image = (
    modal.Image.debian_slim()
    .apt_install("git")
    # Add a timestamp or version to force cache invalidation when git repo changes
    .env({"FORCE_BUILD": "20241129_1"}) 
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
    polar_file: str = "volvo70"
):
    from fastapi import Response
    from fastapi.responses import StreamingResponse
    import xarray as xr
    import numpy as np
    import json
    import asyncio
    # Import here to ensure they are available in the container
    from weather_router import isochronal_weather_router, polar, point_validity

    if freq not in ["1hr", "3hr"]:
        return Response(content="freq must be '1hr' or '3hr'", status_code=400)

    start_point = (start_lat, start_lon)
    end_point = (end_lat, end_lon)

    # Load weather data
    # Note: Using the URL from the original script
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
        ds = ds.isel(lead_time=slice(lead_time_start, 120))
    elif freq == "3hr":
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

    u10 = ds['wind_u_10m']
    v10 = ds['wind_v_10m']
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
    url = f"https://peterm790.s3.af-south-1.amazonaws.com/polars/{polar_file}.pol"
    with urllib.request.urlopen(url) as response:
        polar_data = response.read().decode('utf-8')

    volvo70_polar = polar.Polar(f=io.StringIO(polar_data))

    # Download land-sea mask
    import os
    mask_url = "https://peterm790.s3.af-south-1.amazonaws.com/era5_land-sea-mask.nc"
    mask_file = "/tmp/era5_land-sea-mask.nc"
    
    if not os.path.exists(mask_file):
        print(f"Downloading mask from {mask_url}...")
        with urllib.request.urlopen(mask_url) as response:
            with open(mask_file, 'wb') as out_file:
                out_file.write(response.read())

    # Initialize Router
    step_val = 3 if freq == "3hr" else 1
    
    progress_queue = asyncio.Queue()

    def progress_callback(step, dist_wp):
        progress_queue.put_nowait({
            "type": "progress",
            "step": step,
            "dist": float(dist_wp)
        })

    weatherrouter = isochronal_weather_router.weather_router(
        volvo70_polar,
        get_wind,
        time_steps=ds_processed.time.values,
        step=step_val,
        start_point=start_point,
        end_point=end_point,
        point_validity_extent=[min_lat, min_lon, max_lat, max_lon],
        point_validity_file=mask_file,
        spread=130,
        wake_lim=30,
        rounding=2,
        n_points=30,
        progress_callback=progress_callback
    )

    # Run Routing
    async def generate():
        # Start routing in a separate task
        routing_task = asyncio.create_task(asyncio.to_thread(weatherrouter.route))
        
        # Consume progress updates while routing is running
        while not routing_task.done():
            try:
                # Wait for progress or task completion
                progress = await asyncio.wait_for(progress_queue.get(), timeout=0.1)
                yield json.dumps(progress) + "\n"
            except asyncio.TimeoutError:
                continue
        
        await routing_task

        # Yield any remaining progress messages
        while not progress_queue.empty():
            progress = await progress_queue.get()
            yield json.dumps(progress) + "\n"

        initial_route = weatherrouter.get_fastest_route(stats=False)
        
        initial_isochrones = weatherrouter.get_isochrones()
        # optimize is CPU bound, run in thread
        await asyncio.to_thread(weatherrouter.optimize, initial_route, initial_isochrones)
        route_df = weatherrouter.get_fastest_route(use_optimized=True)
        
        # Process and yield final route
        if isinstance(route_df, list):
             route_data = route_df
        else:
            if 'time' in route_df.columns:
                route_df['time'] = route_df['time'].astype(str)
            route_data = route_df.to_dict(orient="records")
            
        yield json.dumps({
            "type": "result", 
            "data": route_data
        }) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")

    return StreamingResponse(generate(), media_type="application/x-ndjson")
