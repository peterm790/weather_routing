import xarray as xr
import numpy as np
import pandas as pd
import math
import geopy
import geopy.distance


class weather_router:
    def __init__ (self, 
                polar,  
                get_wind, 
                time_steps,
                step,
                start_point,
                end_point, 
                point_validity = None,
                ):
        """
        weather_router: class
            :param polar: class
                a class to return boat speed given tws and twa
            :param  get_wind: function
                supplied function to return tuple of (twd, tws) given (t,lat,lon)
            :param time_steps: list[numpy.datetime64]
                list of time steps, time at which to start assumed to be at time_step[0]
            :param step: int,float64
                number of hours between time_steps
            :param start_point: (float64, float64)
                (lat,lon) start position
            :param end_point: (float64, float64)
                (lat,lon) end position                
            :param point_validity function
                supplied function to return boolean (land or no)
        """

        self.end = False
        self.polar = polar
        self.track = []
        self.get_wind = get_wind
        self.time_steps = time_steps
        self.step = step
        self.start_point = start_point
        self.end_point = end_point
        if point_validity == None:
            from point_validity import land_sea_mask
            lsm = land_sea_mask()
            self.point_validity = lsm.point_validity_arr
        else:
            self.point_validity = point_validity


    def getDist_wp(self, lat, lon):
        return geopy.distance.great_circle((lat,lon), self.end_point).nm


    def getBearing_from_start(self, lat2,lon2):
        pointA = self.start_point
        pointB = (lat2,lon2)
        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])
        diffLong = math.radians(pointB[1] - pointA[1])
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        return compass_bearing

    def getBearing_to_end(self, lat2,lon2):
        pointA = self.end_point
        pointB = (lat2,lon2)
        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])
        diffLong = math.radians(pointB[1] - pointA[1])
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)
        compass_bearing = (initial_bearing + 360) % 360
        return compass_bearing

    def getTWA_from_heading(self, bearing,TWD):
        TWA = bearing - TWD
        return (TWA + 180) % 360 - 180

    def myround(self, x, base=5):
        return base * round(x/base)

    def get_possible(self,lat_init, lon_init, route, t):
        possible = []
        twd, tws = self.get_wind(t, lat_init, lon_init)
        route.append('dummy')
        for heading in range(0,360,5):
            twa = self.getTWA_from_heading(heading, twd)
            speed = self.polar.getSpeed(tws,np.abs(twa))
            end_point = geopy.distance.geodesic(nautical=speed*self.step).destination((lat_init,lon_init), heading)
            lat,lon = end_point.latitude, end_point.longitude
            route = route[:-1]
            route.append((lat,lon))
            if self.point_validity(lat, lon):
                dist_wp = self.getDist_wp(lat, lon)
                if dist_wp <= self.dist_wp_init:
                    bearing_start = self.myround(int(self.getBearing_from_start(lat,lon)))            
                    possible.append([lat, lon, route, dist_wp, bearing_start])
        return possible
    
    def route(self):
        lat, lon = self.start_point
        self.dist_wp_init = self.getDist_wp(lat, lon)
        self.isochrones = []
        if not self.point_validity(lat,lon):
            print('start point error')
        else:
            step = 0
            not_done = True
            while not_done:
                for step,t in enumerate(self.time_steps):
                    print(step)
                    if step == 0:
                        possible = self.get_possible(lat, lon, [self.start_point], t)
                    else:
                        arr = np.array(possible, dtype=object)
                        arr = arr[arr[:, -1].argsort()]
                        split = np.split(arr, np.unique(arr[:, -1], return_index=True)[1][1:])
                        keep = []
                        for a in split:
                            keep.append(a[np.argmin(a[:,-2])])
                        self.isochrones.append(np.array(keep))
                        dist_wp = np.min(self.isochrones[-1][:,-2]) 
                        if dist_wp > 30:
                            if step == len(self.time_steps)-1:
                                print('out of time')
                                not_done = False
                                break
                            else:
                                possible_at_t = []
                                for lat, lon, route in list(self.isochrones[-1][:,:3]):
                                    possible_at_t.append(self.get_possible(lat, lon, route, t))
                        else:
                                print('reached dest')
                                not_done = False
                                break
                        possible = sum(possible_at_t,[]) 

    def get_isochrones(self):
        return self.isochrones

    def get_fastest_route(self):
        df = pd.DataFrame(np.concatenate(self.isochrones))
        df .columns =  ['lat', 'lon','route', 'dist_wp','brg_start']
        return df.iloc[pd.to_numeric(df['dist_wp']).idxmin()].route

