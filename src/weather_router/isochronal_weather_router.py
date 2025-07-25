import numpy as np
import pandas as pd
import math
import geopy
import geopy.distance


class weather_router:
    def __init__(
                self,
                polar,
                get_wind,
                time_steps,
                step,
                start_point,
                end_point,
                spread=110,
                wake_lim=45,
                rounding=3,
                n_points=50,
                point_validity_extent=None,
                tack_penalty=0.5,
                finish_size=20,
                optimise_n_points=None,
                optimise_window=24
                ):
        """
        weather_router: class
            :param polar: class
                a class to return boat speed given tws and twa
            :param  get_wind: function
                supplied function to return tuple of (twd,tws) given(t,lat,lon)
            :param time_steps: list[numpy.datetime64]
                list of time steps, start_time=time_step[0]
            :param step: int,float64
                number of hours between time_steps
            :param start_point: (float64, float64)
                (lat,lon) start position
            :param end_point: (float64, float64)
                (lat,lon) end position
            :param n_points: int
                number of points to maintain in each isochrone for algorithm speed control
            :param point_validity_extent: list
                extent to trim point validity to. [lat1,lon1,lat2,lon2]
            :param tack_penalty: float
                speed penalty (0.0-1.0) applied when tacking (TWA changes sides)
            :param finish_size: int
                size of the finish area in nm
            :param optimise_n_points: int
                number of points to maintain in each isochrone during optimization (defaults to n_points*2)
            :param optimise_window: int
                time step window size (±) for optimization constraint search (default: 24)
        """

        self.end = False
        self.polar = polar
        self.track = []
        self.get_wind = get_wind
        self.time_steps = time_steps
        self.step = step
        self.start_point = start_point
        self.end_point = end_point
        self.spread = spread
        self.wake_lim = wake_lim
        self.rounding = rounding
        self.n_points = n_points
        self.tack_penalty = tack_penalty
        self.finish_size = finish_size
        self.optimise_n_points = optimise_n_points if optimise_n_points is not None else n_points * 2
        self.optimise_window = optimise_window

        from . import point_validity
        land_sea_mask = point_validity.land_sea_mask
        if point_validity_extent:
            lsm = land_sea_mask(point_validity_extent)
        else:
            lsm = land_sea_mask()
        self.point_validity = lsm.point_validity_arr

    def get_min_dist_wp(self, isochrones):
        dists = []
        for lat, lon in isochrones[:, :2]:
            dists.append(
                geopy.distance.great_circle((lat, lon), self.end_point).nm
                )
        return min(dists)

    def get_dist_wp(self, isochrones):
        dists = []
        for lat, lon in isochrones[:, :2]:
            dists.append(
                geopy.distance.great_circle((lat, lon), self.end_point).nm
                )
        return dists

    def getBearing(self, pointA, pointB):
        lat1 = math.radians(pointA[0])
        lat2 = math.radians(pointB[0])
        diffLong = math.radians(pointB[1] - pointA[1])
        x = math.sin(diffLong) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (
            math.sin(lat1) * math.cos(lat2) * math.cos(diffLong)
            )
        initial_bearing = math.atan2(x, y)
        return (math.degrees(initial_bearing) + 360) % 360

    def getTWA_from_heading(self, bearing, TWD):
        TWA = bearing - TWD
        return (TWA + 180) % 360 - 180

    def is_tacking(self, current_twa, previous_twa):
        """
        Determine if the boat is tacking (TWA changes from one side to another).
        Returns True if tacking, False otherwise.
        """
        if previous_twa is None:
            return False
        
        # Check if TWA crossed from port to starboard or vice versa
        # Port side: negative TWA, Starboard side: positive TWA
        previous_side = previous_twa >= 0  # True for starboard, False for port
        current_side = current_twa >= 0    # True for starboard, False for port
        
        return previous_side != current_side

    def myround(self, x, base=1):
        return base * round(x/base)

    def get_wake_lims(self, bearing_to_finish):
        backbearing = ((bearing_to_finish - 180) + 360) % 360
        upper = ((backbearing+self.wake_lim) + 360) % 360
        lower = ((backbearing-self.wake_lim) + 360) % 360
        return (upper, lower)

    def is_not_in_wake(self, wake_lims, bearing):
        in_wake = False
        upper, lower = wake_lims
        if upper > lower:
            if bearing <= upper:
                if bearing >= lower:
                    in_wake = True
        else:
            if bearing >= lower:
                in_wake = True
            if bearing <= upper:
                in_wake = True
        return in_wake

    def prune_slow(self, arr):
        keep = [True] * len(arr)
        for i in range(len(arr)):
            if keep[i] is True:
                wake = self.get_wake_lims(arr[i][3])
                for j in range(len(arr)):
                    if not i == j:
                        if keep[j] is True:
                            bearing = self.getBearing(
                                (arr[i][0], arr[i][1]), (arr[j][0], arr[j][1])
                                )  # inputting lat,lon of array i and j
                            if self.is_not_in_wake(wake, bearing):
                                keep[j] = False
        return arr[keep]

    def return_equidistant(self, isochrone):
        sort_iso = sorted(isochrone , key=lambda k: [k[1], k[0]])
        y = [x[0] for x in sort_iso]
        x = [x[1] for x in sort_iso]
        xd = np.diff(x)
        yd = np.diff(y)
        dist = np.sqrt(xd**2+yd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0],u])
        t = np.linspace(0,u.max(),self.n_points)
        xn = np.interp(t, u, x)
        yn = np.interp(t, u, y)
        return np.column_stack([yn,xn])

    def prune_equidistant(self, possible):
        """
        Prune points using equidistant spacing approach.
        Returns the original points closest to equidistant points along the isochrone curve.
        """
        arr = np.array(possible, dtype=object)
        df = pd.DataFrame(arr)
        df['dist_wp'] = self.get_dist_wp(arr)
        
        # Get equidistant points along the curve
        isochrone_points = [(row[0], row[1]) for row in possible]
        equidistant_points = self.return_equidistant(isochrone_points)
        
        # Find closest original points to each equidistant point
        selected_indices = []
        for eq_point in equidistant_points:
            min_dist = float('inf')
            closest_idx = 0
            for i, row in enumerate(possible):
                lat, lon = row[0], row[1]
                # Calculate distance between original point and equidistant point
                dist = np.sqrt((lat - eq_point[0])**2 + (lon - eq_point[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i
            if closest_idx not in selected_indices:
                selected_indices.append(closest_idx)
        
        # Return selected points and minimum distance to waypoint
        selected_points = [possible[i] for i in selected_indices]
        selected_arr = np.array(selected_points, dtype=object)
        selected_df = pd.DataFrame(selected_arr)
        selected_df['dist_wp'] = self.get_dist_wp(selected_arr)
        dist_wp_min = selected_df['dist_wp'].min()
        
        return selected_arr, dist_wp_min

    def prune_close_together(self, possible):
        arr = np.array(possible, dtype=object)
        df = pd.DataFrame(arr)
        df['dist_wp'] = self.get_dist_wp(arr)
        df = df.sort_values('dist_wp')
        df['round_lat'] = df.iloc[:, 0].apply(
            pd.to_numeric
            ).round(
            self.rounding
            )
        df['round_lon'] = df.iloc[:, 1].apply(
            pd.to_numeric
            ).round(
            self.rounding
            )
        df['tups'] = df[['round_lat', 'round_lon']].apply(tuple, axis=1)
        df = df.drop_duplicates(subset=['tups'])
        dit_wp_min = df['dist_wp'].min()
        return df.iloc[:, :5].to_numpy(), dit_wp_min

    def get_possible(self, lat_init, lon_init, route, bearing_end, t, previous_twa=None):
        possible = []
        twd, tws = self.get_wind(t, lat_init, lon_init)
        upper = int(bearing_end) + self.spread
        lower = int(bearing_end) - self.spread
        route.append('dummy')
        for heading in range(lower, upper, 10):
            heading = ((int(heading) + 360) % 360)
            twa = self.getTWA_from_heading(heading, twd)
            speed = self.polar.getSpeed(tws, np.abs(twa))
            
            # Apply tack penalty if tacking
            if self.is_tacking(twa, previous_twa):
                speed = speed * (1.0 - self.tack_penalty)
            
            end_point = geopy.distance.great_circle(
                nautical=speed*self.step
                ).destination(
                (lat_init, lon_init), heading
                )
            lat, lon = end_point.latitude, end_point.longitude
            route = route[:-1]
            route.append((lat, lon))
            if self.point_validity(lat, lon):
                bearing_end = self.getBearing((lat, lon), self.end_point)
                # Store current TWA with the possible position for next iteration
                possible.append([lat, lon, route, bearing_end, twa])
        return possible

    def route(self):
        lat, lon = self.start_point
        self.isochrones = []
        if not self.point_validity(lat, lon):
            print('start point error')
        else:
            step = 0
            not_done = True
            while not_done:
                possible_at_t = None
                possible = None
                for step, t in enumerate(self.time_steps):
                    #print(step)
                    if step == 0:
                        bearing_end = self.getBearing(
                                                    (lat, lon),
                                                    self.end_point
                                                    )
                        possible = self.get_possible(
                                                    lat, lon,
                                                    [self.start_point],
                                                    bearing_end,
                                                    t
                                                    )
                    else:
                        possible = self.prune_slow(np.array(possible, dtype=object))
                        possible, dist_wp = self.prune_equidistant(possible)
                        print('step', step, 'number of isochrone points', len(possible), 'dist to finish', f'{dist_wp:.1f}')
                        self.isochrones.append(possible)
                        if dist_wp > self.finish_size:
                            if step == len(self.time_steps)-1:
                                print('out of time')
                                not_done = False
                                break
                            else:
                                possible_at_t = []
                                for x in list(self.isochrones[-1]):
                                    lat, lon, route, bearing_end, previous_twa = x
                                    possible_at_t.append(
                                        self.get_possible(
                                            lat, lon, route, bearing_end, t, previous_twa)
                                        )
                        else:
                            print('reached dest')
                            not_done = False
                            break
                        if possible_at_t:
                            possible = sum(possible_at_t, [])

    def get_isochrones(self):
        return self.isochrones

    def get_isochrones_latlon(self):
        return [iso[:, :2] for iso in self.isochrones]
    
    def get_optimized_isochrones(self):
        """Get isochrones from optimization pass"""
        if hasattr(self, 'optimized_isochrones'):
            return self.optimized_isochrones
        else:
            return []
    
    def get_optimized_isochrones_latlon(self):
        """Get lat/lon coordinates of isochrones from optimization pass"""
        if hasattr(self, 'optimized_isochrones'):
            return [iso[:, :2] for iso in self.optimized_isochrones]
        else:
            return []

    def get_fastest_route(self, stats=True, use_optimized=False):
        """
        Get the fastest route from either regular or optimized isochrones.
        
        :param stats: whether to include detailed statistics
        :param use_optimized: if True, use optimized isochrones; if False, use regular isochrones
        """
        if use_optimized and hasattr(self, 'optimized_isochrones') and len(self.optimized_isochrones) > 0:
            isochrones_to_use = self.optimized_isochrones
        else:
            isochrones_to_use = self.isochrones
        
        df = pd.DataFrame(isochrones_to_use[-1])
        df.columns = ['lat', 'lon', 'route', 'brg', 'twa_at_arrival']
        df['dist_wp'] = self.get_dist_wp(isochrones_to_use[-1])
        wp_dists = df['dist_wp'].astype(float)
        fastest = df.iloc[wp_dists.idxmin()].route
        if stats:
            if fastest[-1] == 'dummy':
                df = pd.DataFrame(np.array(fastest[:-1]))
            else:
                df = pd.DataFrame(np.array(fastest))
            df.columns = ['lat', 'lon']
            df['time'] = self.time_steps[:len(df)]
            df[['twd', 'tws']] = pd.DataFrame(
                df.apply(lambda x: self.get_wind(x.time, x.lat, x.lon), axis=1
                         ).tolist(), index=df.index)
            df['pos'] = df[['lat', 'lon']].apply(tuple, axis=1)
            next_pos = list(df['pos'][1:])
            next_pos.append(self.end_point)
            df['next_pos'] = next_pos
            df['heading'] = df.apply(
                lambda x: self.getBearing(x.pos, x.next_pos), axis=1
                )
            df['twa'] = df.apply(
                lambda x: self.getTWA_from_heading(x.heading, x.twd), axis=1
                )
            
            # Calculate base boat speed and tack penalties
            df['base_boat_speed'] = df.apply(
                lambda x: self.polar.getSpeed(x.tws, np.abs(x.twa)), axis=1
                )
            
            # Determine if each leg involves tacking
            df['is_tacking'] = False
            df['boat_speed'] = df['base_boat_speed']
            
            for i in range(1, len(df)):
                current_twa = df.iloc[i]['twa']
                previous_twa = df.iloc[i-1]['twa']
                is_tacking = self.is_tacking(current_twa, previous_twa)
                df.iloc[i, df.columns.get_loc('is_tacking')] = is_tacking
                if is_tacking:
                    df.iloc[i, df.columns.get_loc('boat_speed')] = df.iloc[i]['base_boat_speed'] * (1.0 - self.tack_penalty)
            
            df['hours_elapsed'] = list(df.index)
            df['hours_elapsed'] = df['hours_elapsed']*self.step
            df['days_elapsed'] = df['hours_elapsed']/24
            fastest = df
        return fastest  # .set_index('time')

    def calculate_isochrone_spacing(self, isochrones):
        """
        Calculate spacing between isochrone points for optimization constraints.
        Returns spacing in nautical miles.
        """
        spacings = []
        for isochrone in isochrones:
            if len(isochrone) < 2:
                spacings.append(5.0)  # Default 5nm spacing
                continue
            
            # Extract lat/lon coordinates 
            points = [(float(row[0]), float(row[1])) for row in isochrone]
            
            # Sort points by longitude then latitude for consistent spacing calculation
            sort_points = sorted(points, key=lambda k: [k[1], k[0]])
            
            # Calculate cumulative distances using geographical distances
            total_distance = 0.0
            for i in range(len(sort_points) - 1):
                dist_nm = geopy.distance.great_circle(sort_points[i], sort_points[i+1]).nm
                total_distance += dist_nm
            
            # Average spacing between points in nautical miles
            avg_spacing = total_distance / (len(points) - 1) if len(points) > 1 else 5.0
            spacings.append(avg_spacing)
            
        return spacings

    def is_within_optimization_region(self, lat, lon, time_step_idx, route_points, spacings):
        """
        Check if a point is within the optimization region defined by the previous route.
        Uses closest constraint center within a window around the current time step.
        All distances calculated in nautical miles.
        """
        if len(route_points) == 0 or len(spacings) == 0:
            return True
        
        # Define search window around current time step
        window_size = self.optimise_window
        start_idx = max(0, time_step_idx - window_size)
        end_idx = min(len(route_points), time_step_idx + window_size + 1)
        
        # Find the closest route point within the window
        min_distance_to_center = float('inf')
        closest_spacing = spacings[min(time_step_idx, len(spacings) - 1)]  # fallback
        
        for i in range(start_idx, end_idx):
            center_lat, center_lon = route_points[i]
            # Use geographical distance in nautical miles
            distance_to_center = geopy.distance.great_circle((lat, lon), (center_lat, center_lon)).nm
            if distance_to_center < min_distance_to_center:
                min_distance_to_center = distance_to_center
                closest_spacing = spacings[min(i, len(spacings) - 1)]
        
        # Use the closest spacing as constraint radius (in nautical miles)
        constraint_radius = closest_spacing
        
        return min_distance_to_center <= constraint_radius

    def get_possible_optimized(self, lat_init, lon_init, route, bearing_end, t, time_step_idx, route_points, spacings, previous_twa=None):
        """
        Generate possible next positions with optimization region constraints.
        """
        possible = []
        twd, tws = self.get_wind(t, lat_init, lon_init)
        upper = int(bearing_end) + self.spread
        lower = int(bearing_end) - self.spread
        route.append('dummy')
        
        for heading in range(lower, upper, 10):
            heading = ((int(heading) + 360) % 360)
            twa = self.getTWA_from_heading(heading, twd)
            speed = self.polar.getSpeed(tws, np.abs(twa))
            
            # Apply tack penalty if tacking
            if self.is_tacking(twa, previous_twa):
                speed = speed * (1.0 - self.tack_penalty)
            
            end_point = geopy.distance.great_circle(
                nautical=speed*self.step
                ).destination(
                (lat_init, lon_init), heading
                )
            lat, lon = end_point.latitude, end_point.longitude
            route = route[:-1]
            route.append((lat, lon))
            
            # Check point validity and optimization region constraints
            if (self.point_validity(lat, lon) and 
                self.is_within_optimization_region(lat, lon, time_step_idx, route_points, spacings)):
                bearing_end = self.getBearing((lat, lon), self.end_point)
                possible.append([lat, lon, route, bearing_end, twa])
        
        return possible

    def optimize(self, previous_route, previous_isochrones):
        """
        Run optimization pass using previous route and isochrones to constrain search space.
        Uses only prune_slow and prune_close_together for finer granularity.
        
        :param previous_route: list of (lat, lon) tuples from previous routing
        :param previous_isochrones: list of isochrone arrays from previous routing
        :return: optimized route
        """
        # Calculate spacing constraints from previous isochrones
        base_spacings = self.calculate_isochrone_spacing(previous_isochrones)
        spacings = [s * 2 for s in base_spacings]

        if spacings[0] < self.finish_size:
            spacings = [self.finish_size] * len(spacings)
        
        # Extract route points for constraint centers
        route_points = [(float(point[0]), float(point[1])) for point in previous_route]

        lat, lon = self.start_point
        self.optimized_isochrones = []
        
        if not self.point_validity(lat, lon):
            print('start point error')
            return None
        
        step = 0
        not_done = True
        while not_done:
            possible_at_t = None
            possible = None
            for step, t in enumerate(self.time_steps):
                if step == 0:
                    bearing_end = self.getBearing((lat, lon), self.end_point)
                    possible = self.get_possible_optimized(
                        lat, lon,
                        [self.start_point],
                        bearing_end,
                        t,
                        step,
                        route_points,
                        spacings
                        )
                else:
                    # Apply prune_slow, prune_close_together, and equidistant pruning with double n_points
                    possible = self.prune_slow(np.array(possible, dtype=object))
                    possible, dist_wp = self.prune_close_together(possible)
                    
                    # Add equidistant pruning if we still have too many points
                    if len(possible) > self.optimise_n_points:
                        # Temporarily use optimise_n_points for pruning
                        original_n_points = self.n_points
                        self.n_points = self.optimise_n_points
                        possible, _ = self.prune_equidistant(possible)
                        self.n_points = original_n_points
                    
                    print('step', step, 'number of isochrone points', len(possible), 'dist to finish', f'{dist_wp:.1f}')
                    self.optimized_isochrones.append(possible)
                    
                    if dist_wp > self.finish_size:
                        if step == len(self.time_steps)-1:
                            not_done = False
                            break
                        else:
                            possible_at_t = []
                            for x in list(self.optimized_isochrones[-1]):
                                lat, lon, route, bearing_end, previous_twa = x
                                possible_at_t.append(
                                    self.get_possible_optimized(
                                        lat, lon, route, bearing_end, t, step, 
                                        route_points, spacings, previous_twa
                                        )
                                    )
                    else:
                        not_done = False
                        break
                    
                    if possible_at_t:
                        possible = sum(possible_at_t, [])
        
        return self.get_fastest_route(use_optimized=True)
