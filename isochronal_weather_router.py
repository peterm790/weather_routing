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
    
    def get_min_dist_wp(self, isochrones):
        dists = []
        for lat,lon in isochrones[:,:2]:
            dists.append(geopy.distance.great_circle((lat,lon), self.end_point).nm)
        return min(dists)

    def get_dist_wp(self, isochrones):
        dists = []
        for lat,lon in isochrones[:,:2]:
            dists.append(geopy.distance.great_circle((lat,lon), self.end_point).nm)
        return dists

    def getBearing(self, pointA, pointB):
        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])
        diffLong = math.radians(pointB[1] - pointA[1])
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diffLong))
        initial_bearing = math.atan2(x, y)
        return (math.degrees(initial_bearing) + 360) % 360

    def getTWA_from_heading(self, bearing,TWD):
        TWA = bearing - TWD
        return (TWA + 180) % 360 - 180

    def myround(self, x, base=1):
        return base * round(x/base)

    def get_wake_lims(self, bearing_to_finish):
        backbearing = ((bearing_to_finish - 180) + 360) % 360
        upper = ((backbearing+45) + 360) % 360
        lower  = ((backbearing-45) + 360) % 360
        return (upper,lower)

    def is_not_in_wake(self, wake_lims, bearing):
        in_wake = True
        upper,lower = wake_lims
        if upper > lower:
            if bearing <= upper:
                if bearing >= lower:
                    in_wake = False
        else:
            if bearing >= lower:
                in_wake = False
            if bearing <= upper:
                in_wake = False
        return in_wake
    
    def prune_slow(self, possible):
        arr = np.array(possible, dtype=object)
        keep = [True] * len(arr)
        for i in range(len(possible)):
            if keep[i] == True:
                for j in range(len(arr)):
                    if not i == j:
                        if keep[j] == True:
                            bearing = self.getBearing((arr[i][0], arr[i][1]), (arr[j][0], arr[j][1])) #inputting lat,lon of array i and j
                            bool_ = self.is_not_in_wake(self.get_wake_lims(arr[i][-1]), bearing) #inputting bearing to finish of i into get_wake_lims and bearing to other point
                            keep[j] = bool_       
        return arr[keep]
    

    def get_possible(self,lat_init, lon_init, route, bearing_end, t):
        possible = []
        twd, tws = self.get_wind(t, lat_init, lon_init)
        #route.append('dummy')
        upper = int(bearing_end)+135
        lower  = int(bearing_end) -135
        for heading in range(lower,upper,5):
            heading = ((int(heading) + 360) % 360)
            twa = self.getTWA_from_heading(heading, twd)
            speed = self.polar.getSpeed(tws,np.abs(twa))
            end_point = geopy.distance.great_circle(nautical=speed*self.step).destination((lat_init,lon_init), heading)
            lat,lon = end_point.latitude, end_point.longitude
            #route = route[:-1]
            route.append((lat,lon))
            if self.point_validity(lat, lon):
                bearing_end = self.getBearing((lat,lon), self.end_point)           
                possible.append([lat, lon, route, bearing_end])
        return possible
    
    def route(self):
        lat, lon = self.start_point
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
                        bearing_end = self.getBearing((lat,lon), self.end_point)
                        possible = self.get_possible(lat, lon, [self.start_point], bearing_end, t)
                    else:
                        self.isochrones.append(self.prune_slow(possible))
                        dist_wp = self.get_min_dist_wp(self.isochrones[-1])
                        if dist_wp > 30:
                            if step == len(self.time_steps)-1:
                                print('out of time')
                                not_done = False
                                break
                            else:
                                possible_at_t = []
                                for lat, lon, route, bearing_end in list(self.isochrones[-1]):
                                    possible_at_t.append(self.get_possible(lat, lon, route, bearing_end, t))
                        else:
                                print('reached dest')
                                not_done = False
                                break
                        possible = sum(possible_at_t,[]) 

    def get_isochrones(self):
        return self.isochrones

    def get_fastest_route(self):
        df = pd.DataFrame(self.isochrones[-1])
        df.columns =  ['lat', 'lon','route', 'brg']
        df['dist_wp'] = self.get_dist_wp(self.isochrones[-1])
        return df.iloc[pd.to_numeric(df['dist_wp']).idxmin()].route

