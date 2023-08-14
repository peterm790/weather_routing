import xarray as xr
import zarr
import numpy as np
import sys
sys.path.insert(0, '../src/weather_router/.')
sys.path.insert(0, './src/weather_router/.')
from isochronal_weather_router import weather_router
from polar import Polar


ds = xr.open_zarr('test/test_ds.zarr')

def getWindAt(t, lat, lon):
    tws_sel = ds.tws.sel(time = t, method = 'nearest')
    tws_sel = tws_sel.sel(lat = lat, lon = lon, method = 'nearest')
    twd_sel = ds.twd.sel(time = t, method = 'nearest')
    twd_sel = twd_sel.sel(lat = lat, lon = lon, method = 'nearest')
    return (np.float32(twd_sel.values), np.float32(tws_sel.values))

#def route():
weatherrouter = weather_router(Polar('test/volvo70.pol'), getWindAt, ds.time.values[:4], 12, (-34,0),(-34,17))
weatherrouter.route()


def test_polar():
    assert weatherrouter.polar.getSpeed(20,45) == np.float64(12.5)

def test_isochrones():
    assert type(weatherrouter.get_isochrones()) == list
    assert len(weatherrouter.get_isochrones()) == 3
    assert len(weatherrouter.get_isochrones()[0][0]) == 5

def test_fastest():
    assert weatherrouter.get_fastest_route(stats=True).shape == (4, 12)
    assert weatherrouter.get_fastest_route(stats=True).iloc[0].boat_speed == 19.52
    assert weatherrouter.get_fastest_route(stats=True).iloc[0].pos == (-34.0, 0.0)
