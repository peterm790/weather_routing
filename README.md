# weather_routing

[![Tests](https://github.com/peterm790/weather_routing/actions/workflows/python-app.yml/badge.svg)](https://github.com/peterm790/weather_routing/actions/workflows/python-app.yml)

### A pure python [weather routing](https://en.wikipedia.org/wiki/Weather_routing) algorithm.

Usage:

To perform a historical routing using a reanalysis model such as ERA5:

- First import packages:

```python
import intake
import xarray as xr
import zarr
import numpy as np

from isochronal_weather_router import weather_router
from polar import Polar
from visualize import visualize_3d
```

- Then clean and load the necessary subset of data into memory:

```python
catalog = intake.open_catalog('s3://esip-qhub-public/ecmwf/intake_catalog.yml')
ds = catalog['ERA5-Kerchunk-2020-2022'].to_dask()

ds.coords['lon'] = ((ds.coords['lon'] + 180) % 360) - 180
ds = ds.sortby(ds.lon)
ds = ds.sel(lat = slice(40,35)).sel(lon = slice(-5,4))
ds = ds.sel(time0 = slice('2022-01-01T12:00:00', '2022-01-05T12:00:00'))
ds = ds.sel(time0 = ds.time0.values[::3]) #consider 3-hourly timesteps to speed up this demo

u10 = ds.eastward_wind_at_10_metres
v10 = ds.northward_wind_at_10_metres
tws = np.sqrt(v10**2 + u10**2)
tws = tws*1.94384 #convert m/s to knots
twd = np.mod(180+np.rad2deg(np.arctan2(u10, v10)),360)
ds = tws.to_dataset(name = 'tws')
ds['twd'] = twd
ds = ds.load()
ds = ds.interpolate_na(dim = 'time0', method = 'linear') #some nans in this dataset not sure why
ds = ds.rename({'time0':'time'})
```
- Rather than the routing program attempt to guess the layout of the weather data, to simplify things the users needs to declare a `get_wind()` function

```python
def get_wind(t, lat, lon):
    tws_sel = ds.tws.sel(time = t, method = 'nearest')
    tws_sel = tws_sel.sel(lat = lat, lon = lon, method = 'nearest')
    twd_sel = ds.twd.sel(time = t, method = 'nearest')
    twd_sel = twd_sel.sel(lat = lat, lon = lon, method = 'nearest')
    return (np.float32(twd_sel.values), np.float32(tws_sel.values))
```

- Next we initialize the routing, declaring the `polar class`, `get_wind` function, list of `time_steps`, number of hours between `steps`, `start_point` and `end_point`. It is also possible to explicitly declare the bounds of the routing area in `point_validity_extent` this helps speed up this part of the programme. While `spread` adjusts the range of possible headings to consider, 180 would consider all possibilities but would slow the programme significantly. This routing is relatively short so we will use 140 degrees either side of the bearing to finish. `wake_lim` controls the degree of 'pruning' where 35 degrees is the size of the wake, behind each point. Please see this [article](http://www.tecepe.com.br/nav/vrtool/routing.htm) for a detailed description of pruning techniques.

```python
Palma = (39.430, 2.596)
Gibraltar = (-36.073, -5.354)

weatherrouter = weather_router(Polar('polar/volvo70.pol'), 
                               get_wind, 
                               time_steps = ds.time.values,
                               step = 3,
                               start_point = Palma,
                               end_point = Gibraltar,
                               point_validity_extent = [35,-5,40,4],
                               spread = 140,
                               wake_lim = 35,
                              )
```

- To run the routing simply call:

```python
weatherrouter.route()
```

- To get a table of statistics from the fastest route:

```python
weatherrouter.get_fastest_route()
```

- And to visualise the routing call (this util is rather buggy):

```
visualize_3d(ds,Palma, Gibraltar, weatherrouter.get_isochrones(), weatherrouter.get_fastest_route(stats = False))
```



