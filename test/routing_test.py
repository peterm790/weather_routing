import xarray as xr
import zarr
import numpy as np
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


land = xr.open_dataset('weather_data/era5_land-sea-mask.nc')
land.coords['longitude'] = (land.coords['longitude'] + 180) % 360 - 180
land = land.sortby(land.longitude)
lsm = land.lsm[0]