import cartopy.crs as ccrs
import holoviews as hv
from holoviews import opts, dim
import geoviews as gv
import geoviews.feature as gf
from cartopy import crs

import xarray
import numpy as np
import pandas as pd

gv.extension('bokeh')



def visualize_3d(ds,start_point, end_point, isochrones, fastest):
    """
        visualize_3d: function
        :param ds: xarray-dataset
            an xarray dataset with tws (knots) and twd (deg compass)
        :param start_point: (float64, float64)
            (lat,lon) start position
        :param end_point: (float64, float64)
            (lat,lon) end position  
        :isochrones list
            list of np.arrays containing lat, lon, route, brg_end, dist_wp
        :fastest list[tuples]
            list of (lat, lon) of fastest route
    """
    ds_ = ds.coarsen({'lat':10, 'lon': 10}, boundary='pad').mean()
    twd_ = np.deg2rad((270 - (ds_.twd)) % 360)
    ds2 = twd_.to_dataset(name = 'wind_angle')
    ds2['tws'] = ds_.tws
    # Specify the dataset, its coordinates and requested variable 
    dataset = gv.Dataset(ds, ['lon', 'lat', 'time'], 'tws', crs=crs.PlateCarree())
    #wind speed background img
    images = dataset.to(gv.Image,dynamic=True)
    # Loading coastlines from Cartopy
    coastline = gf.coastline(line_width=2,line_color='k').opts(projection=ccrs.PlateCarree(),scale='10m')
    # Loading land mask from Cartopy
    land = gf.land.options(scale='10m', fill_color='lightgray')
    #start finish points
    df = pd.DataFrame([start_point,end_point])
    df.columns = ['lat','lon']
    df.index = ['start', 'finish']
    sample_points = dict(Longitude = df.lon.values, Latitude  = df.lat.values)
    points = gv.Points(sample_points).opts(size = 20, color = 'black', projection=ccrs.PlateCarree())
    #plot isochrone at time t
    iso_list = []
    for i in range(len(isochrones)):
        df = pd.DataFrame(isochrones[i])
        df.columns = ['lat', 'lon','route', 'brg', 'dist_wp']
        df = df[['lat','lon']]
        sample_points = dict(Longitude = df.lon.values,Latitude  = df.lat.values)
        iso_list.append(gv.Points(sample_points).opts(size = 5, color = 'white', projection=ccrs.PlateCarree()))
    dict_iso = {ds2.time.values[i]:iso_list[i] for i in range(len(isochrones))}
    iso_map = hv.HoloMap(dict_iso, kdims="time")
    #plot fastest route
    fast_list = []
    for i in range(len(isochrones)):
        df = pd.DataFrame(fastest)
        df.columns = ['lat', 'lon']
        sample_points = dict(Longitude = df.lon.values,Latitude  = df.lat.values)
        fast_list.append(gv.Points(sample_points).opts(size = 5, color = 'black', projection=ccrs.PlateCarree()))
    dict_fast = {ds2.time.values[i]:fast_list[i] for i in range(len(isochrones))}
    fast_map = hv.HoloMap(dict_fast, kdims="time")
    #plot current point
    fast_list = []
    for i in range(len(isochrones)):
        lat,lon = fastest[i]
        sample_points = dict(Longitude = lon, Latitude  = lat)
        fast_list.append(gv.Points(sample_points).opts(size = 10, color = 'green', projection=ccrs.PlateCarree()))
    dict_fast = {ds2.time.values[i]:fast_list[i] for i in range(len(isochrones))}
    fast_map_current = hv.HoloMap(dict_fast, kdims="time")
    # We will normalise the arrow to avoid changes in scale as the time evolves
    max_mag = ds2.tws.max()
    lat = ds2.lat
    lon = ds2.lon
    # Create a disctionary of VectorField values at each time interval
    vector_list = []
    for i in range(len(ds2.time.values)):
        vector_list.append(gv.VectorField((lon, lat, ds2.wind_angle[i],
                                    ds2.tws[i]/max_mag), 
                                    crs=crs.PlateCarree()))
    dict_vector = {ds2.time.values[i]:vector_list[i] for i in range(len(ds2.time.values))}
    # create HoloMap object 
    hmap = hv.HoloMap(dict_vector, kdims="time").opts(opts.VectorField(magnitude=dim('Magnitude'), 
                                                                color='k', 
                                                                width=600, height=500,
                                                                pivot='tip', line_width=1,  
                                                                rescale_lengths=False,
                                                                projection=crs.PlateCarree()))
    # Slider location
    hv.output(widget_location='bottom')
    #make plot
    return images.opts(active_tools=['pan'], cmap='viridis',colorbar=True, 
                width=900, height=700, clim=(0,30)) * coastline * land * hmap * points *iso_map * fast_map * fast_map_current




def visualize_2d(ds,step,start_point, end_point, isochrones, fastest):
    dataset = gv.Dataset(ds.sel(time = ds.time.values[step]), ['lon', 'lat'], 'tws', crs=crs.PlateCarree())
    #wind speed background img
    images = dataset.to(gv.Image)
    coastline = gf.coastline(line_width=2,line_color='k').opts(projection=ccrs.PlateCarree(),scale='10m')
    land = gf.land.options(scale='10m', fill_color='lightgray')
    #start finish points
    start,finish = (-34,17),(-24,-45)
    df = pd.DataFrame([start_point,end_point])
    df.columns = ['lat','lon']
    df.index = ['start', 'finish']
    sample_points = dict(Longitude = df.lon.values, Latitude  = df.lat.values)
    points = gv.Points(sample_points).opts(size = 20, color = 'black', projection=ccrs.PlateCarree())
    df = pd.DataFrame(isochrones)
    df.columns = ['lat', 'lon','route', 'dist_wp','brg_end']
    df = df[['lat','lon']]
    sample_points = dict(Longitude = df.lon.values,Latitude  = df.lat.values)
    iso = gv.Points(sample_points).opts(size = 10, color = 'red', projection=ccrs.PlateCarree())
    df = pd.DataFrame(fastest)
    df.columns = ['lat', 'lon']
    sample_points = dict(Longitude = df.lon.values,Latitude  = df.lat.values)
    fast = gv.Points(sample_points).opts(size = 5, color = 'black', projection=ccrs.PlateCarree())
    #make plot
    return images.opts(active_tools=['pan'], cmap='viridis',colorbar=True, 
                width=900, height=700, clim=(0,30)) * coastline * land * points * iso * fast