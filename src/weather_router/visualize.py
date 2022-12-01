import holoviews as hv
import geoviews as gv
import hvplot.xarray
import cartopy.crs as ccrs
from bokeh.resources import INLINE

import pandas
import xarray
import numpy



class visualize:
    def __init__(self,
                ds,
                start_point, 
                end_point, 
                route_df,
                filename = None):

        
        self.start_point = start_point
        self.end_point = end_point
        self.route_df = route_df
        self.filename = None

        self.ds = ds.sel(time = self.route_df.index.values)


    def get_current_lon_lat(self, time):
        now = self.route_df.loc[time]
        return gv.Points({'lon': [now.lon], 'lat':[now.lat], 'TWS':[round(now.tws)], 'TWD':[round(now.twd)], 'TWA':[round(now.twa)], 'Boat Speed':[round(now.boat_speed)]}, kdims = ['lon', 'lat'],vdims = ['TWS','TWD','TWA','Boat Speed']).opts(color = 'white', size = 12, tools = ['hover'])


    def make_plot(self):
        """
            visualize: function
            :param ds: xarray-dataset
                an xarray dataset with tws (knots) and twd (deg compass)
            :param start_point: (float64, float64)
                (lat,lon) start position
            :param end_point: (float64, float64)
                (lat,lon) end position  
            :route_df pandas.Dataframe
                pandas dataframe containing lat, lon, route, brg_end, dist_wp
        """
        wind = self.ds['tws'].hvplot(groupby = 'time', geo = True, tiles = 'OSM',alpha = 0.5, cmap = 'jet', clim=(0,40), hover=False)
        dsv = self.ds.coarsen({'lat':4, 'lon': 4}, boundary='pad').mean()
        vector = dsv.hvplot.vectorfield(x='lon', y='lat', angle='wind_angle', mag='tws', hover=False, groupby = 'time', geo = True).opts(magnitude='tws')


        sample_points = dict(Longitude = self.route_df.lon.values,Latitude  = self.route_df.lat.values)
        route = gv.Path(sample_points).opts(color = 'white',line_width=4)

        start = gv.Points({'lon': [self.start_point[1]], 
                        'lat':[self.start_point[0]]}, 
                        kdims = ['lon', 'lat']).opts(color = 'green', 
                                                    size = 8, 
                                                    tools = ['hover'])

        finish = gv.Points({'lon': [self.end_point[1]], 
                        'lat':[self.end_point[0]]}, 
                        kdims = ['lon', 'lat']).opts(color = 'red', 
                                                    size = 8, 
                                                    tools = ['hover'])

        current_point = hv.DynamicMap(self.get_current_lon_lat, kdims='time')

        plot = (wind*vector*start*finish*route*current_point).opts(fontscale=1, width=900, height=600)
        hv.output(widget_location='bottom')

        return plot

    def return_plot(self):
        return self.make_plot()
    
    def save_plot(self):
        plot = self.make_plot()
        hvplot.save(plot, 'holoviews_plot.html', resources=INLINE)
