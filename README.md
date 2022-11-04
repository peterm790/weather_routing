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
ds = ds.sel(lat = slice(40,35)).sel(lon = slice(-7,4))
ds = ds.sel(time0 = slice('2022-01-13T12:00:00', '2022-01-20T12:00:00'))

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
                               step = 1,
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

```
+----+---------+---------+---------------------+---------+----------+------------------------------------------+------------------------------------------+-----------+----------+--------------+-----------------+----------------+
|    |     lat |     lon | time                |     twd |      tws | pos                                      | next_pos                                 |   heading |      twa |   boat_speed |   hours_elapsed |   days_elapsed |
+====+=========+=========+=====================+=========+==========+==========================================+==========================================+===========+==========+==============+=================+================+
|  0 | 39.281  | 2.478   | 2022-01-13 12:00:00 | 26.131  |  7.17203 | (39.281, 2.478)                          | (39.26733007683735, 2.2786900146855307)  |       265 | -121.131 |         9.3  |               0 |           0    |
+----+---------+---------+---------------------+---------+----------+------------------------------------------+------------------------------------------+-----------+----------+--------------+-----------------+----------------+
|  1 | 39.2673 | 2.27869 | 2022-01-13 13:00:00 | 25.0675 |  8.31563 | (39.26733007683735, 2.2786900146855307)  | (39.24865599707216, 2.0519613060689865)  |       264 | -121.067 |        10.6  |               6 |           0.25 |
+----+---------+---------+---------------------+---------+----------+------------------------------------------+------------------------------------------+-----------+----------+--------------+-----------------+----------------+
|  2 | 39.2487 | 2.05196 | 2022-01-13 14:00:00 | 24.6236 |  9.62228 | (39.24865599707216, 2.0519613060689865)  | (39.18848697935533, 1.7991971114614662)  |       253 | -131.624 |        12.3  |              12 |           0.5  |
+----+---------+---------+---------------------+---------+----------+------------------------------------------+------------------------------------------+-----------+----------+--------------+-----------------+----------------+
|  3 | 39.1885 | 1.7992  | 2022-01-13 15:00:00 | 30.3027 | 10.8351  | (39.18848697935533, 1.7991971114614662)  | (39.159147147840955, 1.4951761951936726) |       263 | -127.303 |        14.26 |              18 |           0.75 |
+----+---------+---------+---------------------+---------+----------+------------------------------------------+------------------------------------------+-----------+----------+--------------+-----------------+----------------+
|  4 | 39.1591 | 1.49518 | 2022-01-13 16:00:00 | 34.992  | 10.3808  | (39.159147147840955, 1.4951761951936726) | (39.13128950950019, 1.2421527082136428)  |       262 | -132.992 |        11.9  |              24 |           1    |
+----+---------+---------+---------------------+---------+----------+------------------------------------------+------------------------------------------+-----------+----------+--------------+-----------------+----------------+
```

- And to visualise the routing call (this util is rather buggy):

```
visualize_3d(ds,Palma, Gibraltar, weatherrouter.get_isochrones(), weatherrouter.get_fastest_route(stats = False))
```

![plot](bokeh_plot.png)


An interactive example of the above plot is available [here](https://petemarsh.com/readme_example)
