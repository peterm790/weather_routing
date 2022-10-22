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
                start_point,
                end_point, 
                point_validity,
                ):
        """
        weather_router: class
            :param polar: class
                a class to return boat speed given tws and twa
            :param  get_wind: function
                supplied function to return tuple of (twd, tws) given (t,lat,lon)
            :param time_step: list[numpy.datetime64]
                list of time steps, time at which to start assumed to be at time_step[0]
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
        self.start_point = start_point
        self.end_point = end_point
        self.point_validity = point_validity


    def getDist_wp(lat, lon):
        return geopy.distance.great_circle((lat,lon), self.end_point).nm

    def getBearing_from_start(lat2,lon2):
        lat1, lon1 = self.start_point
        dLon = lon2 - lon1;
        y = math.sin(dLon) * math.cos(lat2);
        x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon);
        brng = np.rad2deg(math.atan2(y, x));
        if brng < 0: brng+= 360
        return brng

    def getBearing_to_end(lat2,lon2):
        lat1, lon1 = self.end_point
        dLon = lon2 - lon1;
        y = math.sin(dLon) * math.cos(lat2);
        x = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dLon);
        brng = np.rad2deg(math.atan2(y, x));
        if brng < 0: brng+= 360
        return brng

    def getTWA_from_heading(bearing,TWD):
        TWA = bearing - TWD
        return (TWA + 180) % 360 - 180

    def get_possible(inputs):
        t,arr = inputs
        lat_init, lon_init, route = arr
        possible = []
        twd, tws = self.getWind(t, lat_init, lon_init)
        #get bearing to finish
        bearing_end = int(self.getBearing_to_end(lat_init,lon_init))
        route.append(('dummy'))
        for heading in range(0,360,5):
        #for heading in range(bearing_end-175,bearing_end+175,10): #experiment with more than 180 deg spread
            twa = self.getTWA_from_heading(heading, twd)
            speed = self.polar.getSpeed(tws,math.radians(np.abs(twa)))
            end_point = geopy.distance.geodesic(nautical=speed*12).destination((lat_init,lon_init), heading)
            lat = end_point.latitude
            lon = end_point.longitude
            route = route[:-1]
            route.append((lat,lon))
            if self.point_vality(lat, lon):
                dist_wp = self.getDist_wp(lat, lon)
                bearing_start = self.myround(int(getBearing_from_start(lat,lon)))            
                #if dist_wp <= dist_wp_init+1:
                possible.append([end_point.latitude, end_point.longitude, route, dist_wp, bearing_start,bearing_end, speed, twa, twd, tws, heading,t])
        return possible
    
    def route(self):
        """
        Todo: 
            add error logging
            fix break 
            flag to adjust accuracy vs speed       
        """
        lat, lon = self.start_point
        if point_vality(lat,lon):
            step = 0
            #dist_wp_init = self.getDist_wp(lat, lon)
            isochrones = []
            not_done = True
            while not_done:
                for step,t in enumerate(self.time_steps):
                    print(step)
                    if step == 0:
                        if self.point_vality(lat,lon):
                            possible = self.get_possible([t, [lat, lon, [start_point]]])
                            isochrones.append(np.array(possible, dtype=object))
                        else:
                            print('start point error')
                    else:
                        arr = (np.array(possible, dtype=object))
                        #prune slow tracks
                        df = pd.DataFrame(arr)
                        df .columns = ['lat', 'lon','route', 'dist_wp','brg_start','brg_end','speed', 'twa', 'twd', 'tws', 'heading', 'time']
                        arr = df[df['dist_wp'] == df.groupby('brg_start')['dist_wp'].transform('min')].to_numpy()
                        isochrones.append(arr)
                        #del possible
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
                                #b = db.from_sequence(inputs, npartitions = 8)
                                #b = b.map(get_possible)
                                #possible_at_t = b.compute()
                        else:
                                print('reached dest')
                                not_done = False
                                break
                        possible = sum(possible_at_t,[]) 
        return isochrones


