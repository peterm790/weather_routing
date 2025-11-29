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
    lead_time_start: int = 0
):
    from fastapi import Response
    import xarray as xr
    import numpy as np
    # Import here to ensure they are available in the container
    from weather_router import isochronal_weather_router, polar, point_validity

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
    ds = ds.isel(lead_time=slice(lead_time_start, 119))

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
    url = "https://peterm790.s3.af-south-1.amazonaws.com/volvo70.pol"
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
    weatherrouter = isochronal_weather_router.weather_router(
        volvo70_polar,
        get_wind,
        time_steps=ds_processed.time.values,
        step=1,
        start_point=start_point,
        end_point=end_point,
        point_validity_extent=[min_lat, min_lon, max_lat, max_lon],
        point_validity_file=mask_file,
        spread=130,
        wake_lim=30,
        rounding=2,
        n_points=30,
    )

    # Run Routing
    weatherrouter.route()
    initial_route = weatherrouter.get_fastest_route(stats=False)
    initial_isochrones = weatherrouter.get_isochrones()
    _ = weatherrouter.optimize(initial_route, initial_isochrones)
    route_df = weatherrouter.get_fastest_route(use_optimized=True)
    arr = route_df.to_numpy()
    
    buffer = io.BytesIO()
    np.save(buffer, arr)
    buffer.seek(0)
    
    return Response(content=buffer.read(), media_type="application/octet-stream")
