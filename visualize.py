"""
    WORK IN PROGRESS - doesn't work
"""


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

class vis_class:
    def __init__ (self,
        ds,
        start_point,
        end_point,
        isochrones
        ):

        """
        visualize: class
        :param ds: xarray-dataset
            an xarray dataset with tws (knots) and twd (deg compass)
        :param start_point: (float64, float64)
            (lat,lon) start position
        :param end_point: (float64, float64)
            (lat,lon) end position  
        :isochrones list
            isochrone list from weather router 
        """
        self.ds = ds
        self.start_point = start_point
        self.end_point = end_point
        self.isochrones = isochrones

        ds_ = self.ds.coarsen({'lat':10, 'lon': 10}, boundary='pad').mean()
        twd_ = np.deg2rad((270 - (ds_.twd)) % 360)
        ds2 = twd_.to_dataset(name = 'wind_angle')
        ds2['tws'] = ds_.tws

        dataset = gv.Dataset(self.ds, ['lon', 'lat', 'time'], 'tws', crs=crs.PlateCarree())
        #wind speed background img
        self.images = dataset.to(gv.Image,dynamic=True)
        # Loading coastlines from Cartopy
        self.coastline = gf.coastline(line_width=2,line_color='k').opts(projection=ccrs.PlateCarree(),scale='10m')
        # Loading land mask from Cartopy
        self.land = gf.land.options(scale='10m', fill_color='lightgray')
        #start finish points
        df = pd.DataFrame([self.start_point,self.end_point])
        df.columns = ['lat','lon']
        df.index = ['start', 'finish']
        sample_points = dict(Longitude = df.lon.values, Latitude  = df.lat.values)
        self.points = gv.Points(sample_points).opts(size = 20, color = 'black', projection=ccrs.PlateCarree())
        #wind vectors
        lat = ds2.lat
        lon = ds2.lon
        # We will normalise the arrow to avoid changes in scale as the time evolves
        max_mag = ds2.tws.max()
        # Create a disctionary of VectorField values at each time interval
        vector_list = []
        for i in range(len(ds2.time.values)):
            vector_list.append(gv.VectorField((lon, lat, ds2.wind_angle[i],
                                        ds2.tws[i]/max_mag), 
                                        crs=crs.PlateCarree()))
        dict_vector = {ds2.time.values[i]:vector_list[i] for i in range(len(ds2.time.values))}
        # create HoloMap object 
        self.hmap = hv.HoloMap(dict_vector, kdims="time").opts(opts.VectorField(magnitude=dim('Magnitude'), 
                                                                    color='k', 
                                                                    width=600, height=500,
                                                                    pivot='tip', line_width=1,  
                                                                    rescale_lengths=False,
                                                                    projection=crs.PlateCarree()))

        def make_plot(self):
            # Slider location
            hv.output(widget_location='bottom')
            #make plot
            im = self.images.opts(active_tools=['pan'], cmap='jet',colorbar=True, 
                        width=800, height=500, clim=(0,30)) * self.coastline * self.land * self.hmap * self.points
            return im