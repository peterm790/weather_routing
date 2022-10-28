import xarray as xr
import zarr
import numpy as np
import sys
sys.path.insert(0, '../')
from isochronal_weather_router import weather_router
from polar import Polar
import pandas as pd


ds = xr.open_zarr('/home/peter/Documents/weather_routing/test/test_ds.zarr')

def getWindAt(t, lat, lon):
    tws_sel = ds.tws.sel(time = t, method = 'nearest')
    tws_sel = tws_sel.sel(lat = lat, lon = lon, method = 'nearest')
    twd_sel = ds.twd.sel(time = t, method = 'nearest')
    twd_sel = twd_sel.sel(lat = lat, lon = lon, method = 'nearest')
    return (np.float32(twd_sel.values), np.float32(tws_sel.values))


weatherrouter = weather_router(Polar('volvo70.pol'), getWindAt, ds.time.values, step = 12, (-34,17),(-24,-45))

weatherrouter.route()

print(weatherrouter.get_fastest_route())