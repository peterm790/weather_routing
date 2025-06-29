import holoviews as hv
import geoviews as gv
import hvplot.xarray
from bokeh.resources import INLINE
import numpy as np
import pandas as pd


class visualize:
    def __init__(self,
                 ds,
                 start_point,
                 end_point,
                 route_df,
                 isochrone=False,
                 filename=None):
        """
        visualize: class
        :param ds: xarray-dataset
            an xarray dataset with tws (knots) and twd (deg compass)
        :param start_point: (float64, float64)
            (lat,lon) start position
        :param end_point: (float64, float64)
            (lat,lon) end position
        :route_df pandas.Dataframe
            pandas dataframe containing lat, lon, route, brg_end, dist_wp
        """

        self.start_point = start_point
        self.end_point = end_point
        self.route_df = route_df
        self.filename = filename
        self.ds = ds.sel(time=self.route_df.time.values)
        twd_rad = np.deg2rad(self.ds['twd'])
        self.ds['twd_rad'] = twd_rad
        self.isochrone = isochrone

    def get_current_lon_lat(self, time):
        # Convert time to pandas Timestamp if it's not already
        if isinstance(time, np.datetime64):
            time = pd.Timestamp(time)
        now = self.route_df.loc[self.route_df.time == time].iloc[0]
        return gv.Points(
            {
                'lon': [now.lon],
                'lat': [now.lat],
                'TWS': [round(now.tws)],
                'TWD': [round(now.twd)],
                'TWA': [round(now.twa)],
                'Boat Speed': [round(now.boat_speed)]
            },
            kdims=['lon', 'lat'],
            vdims=['TWS', 'TWD', 'TWA', 'Boat Speed']).opts(color = 'white', size = 12, tools = ['hover'])  # noqa

    def get_current_isochrone(self, time):
        # self.isochrone is a list of lists. each being the list of isochrone points for a given time
        if isinstance(time, np.datetime64):
            time = pd.Timestamp(time)
        
        # Check if isochrone data exists and is not just a boolean
        if not self.isochrone or isinstance(self.isochrone, bool):
            # Return empty points if no isochrone data
            return gv.Points({}).opts(color='white', size=4)
            
        idx = self.route_df.index[self.route_df.time == time]
        
        # Convert pandas Index to integer - take the first match
        if len(idx) > 0:
            idx_int = idx[0]
            isochrone = self.isochrone[idx_int]
            sample_isos = dict(
                Longitude=isochrone[:, 1],
                Latitude=isochrone[:, 0]
                )
            return gv.Points(sample_isos).opts(color='white', size=4)
        else:
            # Return empty points if no matching time found
            return gv.Points({}).opts(color='white', size=4)

    def make_plot(self):
        wind = self.ds['tws'].hvplot(
            groupby='time',
            geo=True,
            tiles='OSM',
            alpha=0.5,
            cmap='jet',
            clim=(0, 40),
            hover=False,
            framewise=False,
            axiswise=False
            )
        dsv = self.ds.coarsen({'lat': 4, 'lon': 4}, boundary='pad').mean()
        vector = dsv.hvplot.vectorfield(
            x='lon',
            y='lat',
            angle='twd_rad',
            mag='tws',
            hover=False,
            groupby='time',
            geo=True,
            framewise=False,
            axiswise=False
            ).opts(magnitude='tws')
        sample_points = dict(
            Longitude=self.route_df.lon.values,
            Latitude=self.route_df.lat.values
            )
        route = gv.Path(sample_points).opts(color='white', line_width=4)
        start = gv.Points(
                    {
                        'lon': [self.start_point[1]],
                        'lat': [self.start_point[0]]
                    },
                    kdims=['lon', 'lat']).opts(
                                                color='green',
                                                size=8,
                                                tools=['hover']
                                                )
        finish = gv.Points(
                    {
                        'lon': [self.end_point[1]],
                        'lat': [self.end_point[0]]
                    },
                    kdims=['lon', 'lat']).opts(
                                                color='red',
                                                size=8,
                                                tools=['hover']
                                                )
        # Create time dimension with values from route_df.time column
        times = list(self.route_df.time.values)
        current_point = hv.DynamicMap(self.get_current_lon_lat, kdims=[hv.Dimension('time', values=times)])
        # show isochrone at time t
        if self.isochrone:
            current_isochrone = hv.DynamicMap(self.get_current_isochrone, kdims=[hv.Dimension('time', values=times)])
            plot = (wind*vector*start*finish*route*current_point*current_isochrone).opts(
                                                                    fontscale=1,
                                                                    width=900,
                                                                    height=600,
                                                                    framewise=False,
                                                                    axiswise=False,
                                                                    shared_axes=True
                                                                    )
        else:
            plot = (wind*vector*start*finish*route*current_point).opts(
                                                                    fontscale=1,
                                                                    width=900,
                                                                    height=600,
                                                                    framewise=False,
                                                                    axiswise=False,
                                                                    shared_axes=True
                                                                    )

        hv.output(widget_location='bottom')
        return plot

    def return_plot(self):
        return self.make_plot()

    def save_plot(self):
        plot = self.make_plot()
        hvplot.save(plot, f'{self.filename}.html', resources='inline')