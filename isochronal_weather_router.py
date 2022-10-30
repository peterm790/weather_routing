import xarray
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
            self.point_validity = lsm.point_validity
        else:
            self.point_validity = point_validity


    def getDist_wp(self, lat, lon):
        return geopy.distance.great_circle((lat,lon), self.end_point).nm

    def getBearing_from_start(self, lat2,lon2):
        lat1, lon1 = self.start_point
        dLon = lon2 - lon1;
        y = math.sin(dLon) * math.cos(lat2);
        x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon);
        brng = np.rad2deg(math.atan2(y, x));
        if brng < 0: brng+= 360
        return brng

    def getBearing_to_end(self, lat2,lon2):
        lat1, lon1 = self.end_point
        dLon = lon2 - lon1;
        y = math.sin(dLon) * math.cos(lat2);
        x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon);
        brng = np.rad2deg(math.atan2(y, x));
        if brng < 0: brng+= 360
        return brng

    def getTWA_from_heading(self, bearing,TWD):
        TWA = bearing - TWD
        return (TWA + 180) % 360 - 180

    def myround(self, x, base=10):
        return base * round(x/base)

    def get_possible(self, inputs):
        t,arr = inputs
        lat_init, lon_init, route = arr
        possible = []
        twd, tws = self.get_wind(t, lat_init, lon_init)
        bearing_end = int(self.getBearing_to_end(lat_init,lon_init))
        route.append('dummy')
        for heading in range(0,360,10):
            twa = self.getTWA_from_heading(heading, twd)
            speed = self.polar.getSpeed(tws,np.abs(twa))
            end_point = geopy.distance.geodesic(nautical=speed*self.step).destination((lat_init,lon_init), heading)
            lat = end_point.latitude
            lon = end_point.longitude
            route = route[:-1]
            route.append((lat,lon))
            if self.point_validity(lat, lon):
                dist_wp = self.getDist_wp(lat, lon)
                bearing_start = self.myround(int(self.getBearing_from_start(lat,lon)))            
                #if dist_wp <= dist_wp_init+1:
                possible.append([end_point.latitude, end_point.longitude, route, dist_wp, bearing_start,bearing_end, speed, twa, twd, tws, heading,t])
        return possible
    
    def route(self):
        """
        Todo: 
            add error logging
            flag to adjust accuracy vs speed       
        """
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
                        possible = self.get_possible([t, [lat, lon, [self.start_point]]])
                    else:
                        arr = (np.array(possible, dtype=object))
                        #prune slow tracks
                        df = pd.DataFrame(arr)
                        df .columns = ['lat', 'lon','route', 'dist_wp','brg_start','brg_end','speed', 'twa', 'twd', 'tws', 'heading', 'time']
                        df = df[df['dist_wp'] == df.groupby('brg_start')['dist_wp'].transform('min')]
                        #df = df.sort_values('lat')
                        self.isochrones.append(df.to_numpy())
                        dist_wp = np.min(arr[:, 3]) 
                        if dist_wp > 30:
                            possible_at_t = []
                            if step == len(self.time_steps)-1:
                                print('out of time')
                                not_done = False
                                break
                            else:
                                inputs = [[t,x] for x in list(arr[:,:3])]
                                possible_at_t = []
                                for i in inputs:
                                    possible_at_t.append(self.get_possible(i))
                        else:
                                print('reached dest')
                                not_done = False
                                break
                        possible = sum(possible_at_t,[]) 

    def get_isochrones(self):
        return self.isochrones

    def get_fastest_route(self):
        df = pd.DataFrame(np.concatenate(self.isochrones))
        df .columns =  ['lat', 'lon','route', 'dist_wp','brg_start','brg_end','speed', 'twa', 'twd', 'tws', 'heading', 'time']
        return df.iloc[pd.to_numeric(df['dist_wp']).idxmin()].route

