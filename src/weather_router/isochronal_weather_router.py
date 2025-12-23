import numpy as np
import pandas as pd
import math


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
            point_validity_file=None,
            point_validity_method = 'nearest',
            tack_penalty=0.5,
            finish_size=20,
            optimise_n_points=None,
            optimise_window=24,
            progress_callback=None,
            avoid_land_crossings: bool = False,
            leg_check_spacing_nm: float = 1.0,
            leg_check_max_samples: int = 25,
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
            :param point_validity_file: str or xarray.Dataset
                Path to land-sea mask netcdf file or xarray dataset.
            :param point_validity_method: str
                either nearest or arr, use arr if land sea mask is directly from weather data (mostly deprecated)
            :param tack_penalty: float
                speed penalty (0.0-1.0) applied when tacking (TWA changes sides)
            :param finish_size: int
                size of the finish area in nm
            :param optimise_n_points: int
                number of points to maintain in each isochrone during optimization (defaults to n_points*5)
            :param optimise_window: int
                time step window size (±) for optimization constraint search (default: 24)
            :param progress_callback: function, optional
                Callback function to report progress. Called with (step, dist_to_finish, isochrones).
            :param avoid_land_crossings: bool
                If True, reject any candidate leg that crosses land by sampling intermediate points.
            :param leg_check_spacing_nm: float
                Target sampling spacing (nautical miles) along each candidate leg.
                Must be >= 0.25nm.
            :param leg_check_max_samples: int
                Maximum number of intermediate samples per candidate leg (performance cap).
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
        self.optimise_n_points = optimise_n_points if optimise_n_points is not None else n_points * 5
        self.optimise_window = optimise_window
        self.progress_callback = progress_callback

        if leg_check_spacing_nm < 0.25:
            raise ValueError("leg_check_spacing_nm must be >= 0.25 (nautical miles)")
        if leg_check_max_samples < 1:
            raise ValueError("leg_check_max_samples must be >= 1")

        self.avoid_land_crossings = avoid_land_crossings
        self.leg_check_spacing_nm = float(leg_check_spacing_nm)
        self.leg_check_max_samples = int(leg_check_max_samples)


        from . import point_validity
        land_sea_mask = point_validity.land_sea_mask
        if point_validity_extent:
            lsm = land_sea_mask(point_validity_extent, file=point_validity_file, method = point_validity_method)
        else:
            lsm = land_sea_mask(file=point_validity_file, method = point_validity_method)
        self.point_validity = lsm.point_validity

    def haversine_vectorized(self, lat1, lon1, lat2, lon2):
        R = 3440.065
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        return R * c

    def destination_vectorized(self, lat1, lon1, dist, bearing):
        R = 3440.065
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        bearing_rad = np.radians(bearing)
        angular_dist = dist / R
        
        lat2_rad = np.arcsin(np.sin(lat1_rad) * np.cos(angular_dist) + 
                             np.cos(lat1_rad) * np.sin(angular_dist) * np.cos(bearing_rad))
        
        lon2_rad = lon1_rad + np.arctan2(np.sin(bearing_rad) * np.sin(angular_dist) * np.cos(lat1_rad),
                                         np.cos(angular_dist) - np.sin(lat1_rad) * np.sin(lat2_rad))
        
        return np.degrees(lat2_rad), np.degrees(lon2_rad)

    def bearing_vectorized(self, lat1, lon1, lat2, lon2):
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon_rad = np.radians(lon2 - lon1)
        
        y = np.sin(dlon_rad) * np.cos(lat2_rad)
        x = np.cos(lat1_rad) * np.sin(lat2_rad) - \
            np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(dlon_rad)
        
        bearing_rad = np.arctan2(y, x)
        return (np.degrees(bearing_rad) + 360) % 360

    def leg_is_clear(self, lat0, lon0, heading, distance_nm):
        """
        Return True if the great-circle leg stays on water.
        """
        if not self.avoid_land_crossings:
            return True

        if distance_nm is None or distance_nm <= 0:
            return True

        try:
            distance_nm = float(distance_nm)
        except Exception:
            return False

        if distance_nm <= 0:
            return True

        spacing = self.leg_check_spacing_nm

        if distance_nm <= spacing:
            sample_distances = [distance_nm * 0.5]
        else:
            n_intervals = int(math.ceil(distance_nm / spacing))
            n_interior = max(1, n_intervals - 1)
            if n_interior > self.leg_check_max_samples:
                n_intervals = self.leg_check_max_samples + 1
            sample_distances = [(distance_nm * i / n_intervals) for i in range(1, n_intervals)]

        # Vectorized check for sample points
        # lat0, lon0 are scalars here (single leg check)
        if len(sample_distances) > 0:
            d = np.array(sample_distances)
            lats, lons = self.destination_vectorized(lat0, lon0, d, heading)
            
            # Check validity of all points
            # point_validity is vectorized
            valid = self.point_validity(lats, lons)
            if not np.all(valid):
                return False
                
        return True

    def get_min_dist_wp(self, isochrones):
        if len(isochrones) == 0:
             return 0
        lats = isochrones[:, 0].astype(float)
        lons = isochrones[:, 1].astype(float)
        dists = self.haversine_vectorized(lats, lons, self.end_point[0], self.end_point[1])
        return np.min(dists)

    def get_dist_wp(self, isochrones):
        if len(isochrones) == 0:
             return []
        lats = isochrones[:, 0].astype(float)
        lons = isochrones[:, 1].astype(float)
        dists = self.haversine_vectorized(lats, lons, self.end_point[0], self.end_point[1])
        return dists

    def getBearing(self, pointA, pointB):
        # Kept for compatibility, but updated to use numpy
        return self.bearing_vectorized(pointA[0], pointA[1], pointB[0], pointB[1])

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
        """
        NOTE: Legacy naming — returns True if `bearing` is inside the wake sector.
        Supports scalar or vectorized `bearing`.
        """
        upper, lower = wake_lims

        # Scalar case
        if np.ndim(upper) == 0:
            if upper > lower:
                return (bearing <= upper) and (bearing >= lower)
            return (bearing >= lower) or (bearing <= upper)

        # Vectorized case
        mask_normal = upper > lower
        res = np.zeros_like(bearing, dtype=bool)

        res[mask_normal] = (bearing[mask_normal] <= upper[mask_normal]) & (bearing[mask_normal] >= lower[mask_normal])
        mask_wrapped = ~mask_normal
        res[mask_wrapped] = (bearing[mask_wrapped] >= lower[mask_wrapped]) | (bearing[mask_wrapped] <= upper[mask_wrapped])
        return res

    def prune_slow(self, arr):
        # Vectorized prune slow
        if len(arr) == 0:
            return arr
            
        lats = arr[:, 0].astype(float)
        lons = arr[:, 1].astype(float)
        bearings_to_finish = arr[:, 3].astype(float)
        
        # Calculate pairwise bearings matrix
        # bearings[i, j] is bearing FROM i TO j
        # shape (N, N)
        bearings_matrix = self.bearing_vectorized(lats[:, None], lons[:, None], lats[None, :], lons[None, :])
        
        # Calculate wake limits for each point
        uppers, lowers = self.get_wake_lims(bearings_to_finish)
        
        keep = np.ones(len(arr), dtype=bool)
        
        # We can iterate, but use vectorized checks inside
        # Or better: strictly sort by distance to finish (optional but good heuristic)
        # Assuming arr is not sorted, we just check pairwise.
        
        # Since 'is_not_in_wake' returns True if IN wake (bad naming), 
        # check if j is in wake of i.
        # Check: is_not_in_wake((uppers[i], lowers[i]), bearings_matrix[i, j])
        
        for i in range(len(arr)):
            if keep[i]:
                # Check which j are in wake of i
                # Vectorized check against all j
                in_wake_mask = self.is_not_in_wake((uppers[i], lowers[i]), bearings_matrix[i, :])
                
                # Don't prune self
                in_wake_mask[i] = False
                
                # Prune those in wake
                keep[in_wake_mask] = False
                
        return arr[keep]

    def return_equidistant(self, isochrone):
        # Use numpy operations
        # isochrone is list of [lat, lon, ...]
        pts = np.array([(row[0], row[1]) for row in isochrone], dtype=float)
        # Sort by lon, lat
        # lexicon sort
        ind = np.lexsort((pts[:,0], pts[:,1])) # sort by lat then lon? original key: [k[1], k[0]] -> lon, lat
        pts = pts[ind]
        
        y = pts[:, 0]
        x = pts[:, 1]
        
        xd = np.diff(x)
        yd = np.diff(y)
        dist = np.sqrt(xd**2+yd**2)
        u = np.cumsum(dist)
        u = np.hstack([[0],u])
        
        if u.max() == 0:
             return pts
             
        t = np.linspace(0,u.max(),self.n_points)
        xn = np.interp(t, u, x)
        yn = np.interp(t, u, y)
        return np.column_stack([yn,xn])

    def prune_equidistant(self, possible):
        """
        Prune points using equidistant spacing approach.
        """
        if len(possible) == 0:
            return possible, 0
            
        arr = np.array(possible, dtype=object)
        # Vectorized dist calc
        lats = arr[:, 0].astype(float)
        lons = arr[:, 1].astype(float)
        dists = self.haversine_vectorized(lats, lons, self.end_point[0], self.end_point[1])
        
        # Get equidistant points along the curve
        isochrone_points = [(row[0], row[1]) for row in possible]
        equidistant_points = self.return_equidistant(isochrone_points)
        
        # Find closest original points to each equidistant point
        # Vectorized distance matrix calculation
        eq_lats = equidistant_points[:, 0]
        eq_lons = equidistant_points[:, 1]
        
        # Distance matrix (M equidistant x N original)
        # Use simple Euclidean on lat/lon as approx or haversine
        # The original code used sqrt((lat-lat)**2...) -> Euclidean degrees
        diff_lat = lats[None, :] - eq_lats[:, None]
        diff_lon = lons[None, :] - eq_lons[:, None]
        dist_matrix = np.sqrt(diff_lat**2 + diff_lon**2)
        
        closest_indices = np.argmin(dist_matrix, axis=1)
        unique_indices = np.unique(closest_indices)
        
        selected_arr = arr[unique_indices]
        
        # Recalculate min dist
        min_dist = np.min(dists[unique_indices])
        
        return selected_arr, min_dist

    def prune_close_together(self, possible):
        """
        Remove points that are effectively the same location (within `self.rounding`),
        keeping the best representative (closest to finish) per bucket, then cap to
        `self.n_points`.
        """
        if len(possible) == 0:
            return np.array([], dtype=object), float("inf")

        arr = np.array(possible, dtype=object)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError("prune_close_together expected an array of rows with at least [lat, lon, ...]")

        lats = arr[:, 0].astype(float)
        lons = arr[:, 1].astype(float)
        dists = self.haversine_vectorized(lats, lons, self.end_point[0], self.end_point[1]).astype(float)

        # Group by rounded lat/lon; within each bucket keep the point closest to finish.
        df = pd.DataFrame(
            {
                "idx": np.arange(len(arr), dtype=int),
                "dist": dists,
                "round_lat": np.round(lats, self.rounding),
                "round_lon": np.round(lons, self.rounding),
            }
        )

        df = df.sort_values("dist", ascending=True)
        df = df.drop_duplicates(subset=["round_lat", "round_lon"], keep="first")

        # Hard cap for performance control.
        if len(df) > self.n_points:
            df = df.head(self.n_points)

        kept = df["idx"].to_numpy(dtype=int)
        result_arr = arr[kept]
        min_dist = float(df["dist"].min()) if len(df) else float("inf")
        return result_arr, min_dist

    def get_possible_batch(self, lats, lons, routes, bearings_end, t, previous_twas=None):
        N = len(lats)
        
        # Vectorized wind
        twd, tws = self.get_wind(t, lats, lons)
        
        # Ensure correct shapes for broadcasting
        twd = np.asarray(twd)
        tws = np.asarray(tws)
        
        if twd.ndim == 0:
            # Scalar case (single point)
            twd = np.broadcast_to(twd, (N,))
            tws = np.broadcast_to(tws, (N,))
        elif twd.shape != (N,):
             # Ensure 1D array of correct length
             twd = twd.flatten() if twd.size == N else np.broadcast_to(twd, (N,))
             tws = tws.flatten() if tws.size == N else np.broadcast_to(tws, (N,))
        
        # Headings grid
        heading_offsets = np.arange(-self.spread, self.spread + 1, 10) 
        M = len(heading_offsets)
        
        # headings (N, M)
        headings = bearings_end[:, None] + heading_offsets[None, :]
        headings = (headings + 360) % 360
        
        # TWD, TWS (N, M)
        twd_exp = twd[:, None]
        tws_exp = tws[:, None]
        
        # TWA
        twa = self.getTWA_from_heading(headings, twd_exp)
        
        # Speed
        speed = self.polar.getSpeed(tws_exp, np.abs(twa))
        
        # Tack Penalty
        if previous_twas is not None:
            prev_twa_exp = previous_twas[:, None]
            tacking = self.is_tacking(twa, prev_twa_exp)
            speed = np.where(tacking, speed * (1.0 - self.tack_penalty), speed)
        
        # Destination
        dist = speed * self.step
        new_lats, new_lons = self.destination_vectorized(lats[:, None], lons[:, None], dist, headings)
        
        # Flatten
        flat_lats = new_lats.ravel()
        flat_lons = new_lons.ravel()
        flat_headings = headings.ravel()
        flat_twa = twa.ravel()
        flat_bearing_end = np.zeros_like(flat_lats) # Computed later if valid
        
        # Validity Check
        valid_mask = self.point_validity(flat_lats, flat_lons)
        
        # Optimization Region (if provided - handled by caller logic? No, get_possible_optimized calls this?)
        # Current get_possible does not handle optimization region. 
        # get_possible_optimized does.
        # I should unify or keep separate.
        # For now, this replaces `get_possible`.
        
        valid_indices = np.where(valid_mask)[0]
        
        # Leg Clear Check
        if self.avoid_land_crossings and len(valid_indices) > 0:
            # Check subset
            # Reconstruct parent index
            # idx = p * M + m
            # lat0 = lats[p]
            
            p_indices = valid_indices // M
            
            # Vectorized leg check is harder to do in one massive call if sample counts vary.
            # But we can loop over valid candidates or do a batched check.
            # Given reduced set, a loop might be acceptable or a semi-vectorized approach.
            
            final_valid_indices = []
            for idx in valid_indices:
                 p = idx // M
                 if self.leg_is_clear(lats[p], lons[p], flat_headings[idx], dist.ravel()[idx]):
                     final_valid_indices.append(idx)
            valid_indices = np.array(final_valid_indices, dtype=int)
            
        # Construct Result
        if len(valid_indices) == 0:
            return []
            
        p_indices = valid_indices // M
        
        res_lats = flat_lats[valid_indices]
        res_lons = flat_lons[valid_indices]
        res_twa = flat_twa[valid_indices]
        
        # Calculate new bearings to end
        # Vectorized bearing
        res_bearings_end = self.bearing_vectorized(res_lats, res_lons, self.end_point[0], self.end_point[1])
        
        # Construct routes
        # This is the slow part (list copying).
        # Only do for survivors.
        new_routes = []
        for i, p in enumerate(p_indices):
            # Shallow copy parent route and append new point
            r = routes[p] + [(res_lats[i], res_lons[i])]
            new_routes.append(r)
            
        # Format: [lat, lon, route, bearing_end, twa]
        # Return list of lists
        results = []
        for i in range(len(res_lats)):
             results.append([res_lats[i], res_lons[i], new_routes[i], res_bearings_end[i], res_twa[i]])
             
        return results

    def get_possible(self, lat_init, lon_init, route, bearing_end, t, previous_twa=None):
        # Compatibility wrapper or deprecated?
        # The loop in `route` uses this. I will replace the loop in `route`.
        # But `get_possible` might be called externally?
        # I'll update it to use the batch logic for single point.
        lats = np.array([lat_init])
        lons = np.array([lon_init])
        routes = [route]
        bearings = np.array([bearing_end])
        prev_twas = np.array([previous_twa]) if previous_twa is not None else None
        
        return self.get_possible_batch(lats, lons, routes, bearings, t, prev_twas)

    def route(self):
        """
        Run the main isochronal routing pass.

        Uses vectorized candidate generation via `get_possible_batch()` and pruning via
        `prune_slow()` + `prune_equidistant()`.
        """
        lat, lon = self.start_point
        self.isochrones = []

        if not self.point_validity(lat, lon):
            print('start point error')
            return

        # Current frontier (isochrone) state
        current_lats = np.array([lat], dtype=float)
        current_lons = np.array([lon], dtype=float)
        current_routes = [[self.start_point]]
        current_bearings = np.array([self.getBearing((lat, lon), self.end_point)], dtype=float)
        current_twas = None  # no previous TWA for the first expansion

        for step, t in enumerate(self.time_steps):
            possible = self.get_possible_batch(
                current_lats,
                current_lons,
                current_routes,
                current_bearings,
                t,
                current_twas,
            )

            if step > 0:
                possible = self.prune_slow(np.array(possible, dtype=object))
                possible, dist_wp = self.prune_equidistant(possible)
            else:
                dist_wp = (
                    self.get_min_dist_wp(np.array(possible, dtype=object))
                    if len(possible) > 0
                    else float('inf')
                )

            print('step', step, 'number of isochrone points', len(possible), 'dist to finish', f'{dist_wp:.1f}')
            if self.progress_callback:
                self.progress_callback(step, dist_wp, possible)

            self.isochrones.append(possible)

            if len(possible) == 0:
                print('No valid routes found')
                return

            if dist_wp <= self.finish_size:
                print('reached dest')
                return

            if step == len(self.time_steps) - 1:
                print('out of time')
                return

            # Prepare next frontier
            arr = np.array(possible, dtype=object)
            current_lats = arr[:, 0].astype(float)
            current_lons = arr[:, 1].astype(float)
            current_routes = list(arr[:, 2])
            current_bearings = arr[:, 3].astype(float)
            current_twas = arr[:, 4].astype(float)


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
                # Use vectorized haversine (scalar here)
                dist_nm = self.haversine_vectorized(
                    sort_points[i][0], sort_points[i][1], 
                    sort_points[i+1][0], sort_points[i+1][1]
                )
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
            # Use vectorized haversine (scalar here)
            distance_to_center = self.haversine_vectorized(lat, lon, center_lat, center_lon)
            if distance_to_center < min_distance_to_center:
                min_distance_to_center = distance_to_center
                closest_spacing = spacings[min(i, len(spacings) - 1)]
        
        # Use the closest spacing as constraint radius (in nautical miles)
        constraint_radius = closest_spacing
        
        return min_distance_to_center <= constraint_radius

    def get_possible_batch_optimized(self, lats, lons, routes, bearings_end, t, time_step_idx, route_points, spacings, previous_twas=None):
        N = len(lats)
        
        # 1. Vectorized Wind
        twd, tws = self.get_wind(t, lats, lons)
        
        # Ensure correct shapes for broadcasting
        twd = np.asarray(twd)
        tws = np.asarray(tws)
        
        if twd.ndim == 0:
            # Scalar case
            twd = np.broadcast_to(twd, (N,))
            tws = np.broadcast_to(tws, (N,))
        elif twd.shape != (N,):
             # Ensure 1D array of correct length
             twd = twd.flatten() if twd.size == N else np.broadcast_to(twd, (N,))
             tws = tws.flatten() if tws.size == N else np.broadcast_to(tws, (N,))

        # 2. Heading Candidates
        heading_offsets = np.arange(-self.spread, self.spread + 1, 10)
        M = len(heading_offsets)
        
        # (N, M)
        headings = bearings_end[:, None] + heading_offsets[None, :]
        headings = (headings + 360) % 360
        
        # 3. TWA & Speed
        twd_exp = twd[:, None]
        tws_exp = tws[:, None]
        
        twa = self.getTWA_from_heading(headings, twd_exp)
        speed = self.polar.getSpeed(tws_exp, np.abs(twa))
        
        if previous_twas is not None:
             prev_twa_exp = previous_twas[:, None]
             tacking = self.is_tacking(twa, prev_twa_exp)
             speed = np.where(tacking, speed * (1.0 - self.tack_penalty), speed)
             
        # 4. Destination
        dist = speed * self.step
        new_lats, new_lons = self.destination_vectorized(lats[:, None], lons[:, None], dist, headings)
        
        # Flatten
        flat_lats = new_lats.ravel()
        flat_lons = new_lons.ravel()
        flat_headings = headings.ravel()
        flat_twa = twa.ravel()
        
        # 5. Point Validity (Water vs Land)
        valid_mask = self.point_validity(flat_lats, flat_lons)
        
        # 6. Optimization Region Constraint (Vectorized)
        # Check dist to optimization path
        # We need a vectorized version of is_within_optimization_region or loop over valid points
        # For now, let's filter by validity first to reduce count
        
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_indices) > 0:
            # Check Optimization Region
            # This is hard to fully vectorize without a KDTree or similar, but we can do a batched check against the window center
            # Find center for this time step
            # logic from is_within_optimization_region:
            # search window around time_step_idx
            
            window_size = self.optimise_window
            start_idx = max(0, time_step_idx - window_size)
            end_idx = min(len(route_points), time_step_idx + window_size + 1)
            
            # Extract relevant route points
            window_points = np.array(route_points[start_idx:end_idx]) # (W, 2)
            window_spacings = np.array(spacings[start_idx:end_idx])
            
            if len(window_points) > 0:
                 # Check distances from valid candidates to ALL window points
                 # valid_lats: (V,)
                 # window_lats: (W,)
                 # We need min dist to any window point <= corresponding spacing
                 
                 v_lats = flat_lats[valid_indices]
                 v_lons = flat_lons[valid_indices]
                 
                 # Haversine matrix (V, W)
                 # This might be heavy if V and W are large. 
                 # V ~ 1000s, W ~ 50. 50k calcs -> OK.
                 
                 dists = self.haversine_vectorized(v_lats[:, None], v_lons[:, None], 
                                                 window_points[:, 0][None, :], window_points[:, 1][None, :])
                 
                 # dists is (V, W)
                 # Find min dist and its index for each V
                 min_dists = np.min(dists, axis=1)
                 min_idxs = np.argmin(dists, axis=1)
                 
                 # Corresponding allowed spacing
                 allowed_spacing = window_spacings[min_idxs]
                 
                 # Filter
                 opt_mask = min_dists <= allowed_spacing
                 valid_indices = valid_indices[opt_mask]
        
        # 7. Leg Clear Check
        if self.avoid_land_crossings and len(valid_indices) > 0:
             p_indices = valid_indices // M
             final_valid_indices = []
             for idx in valid_indices:
                 p = idx // M
                 if self.leg_is_clear(lats[p], lons[p], flat_headings[idx], dist.ravel()[idx]):
                     final_valid_indices.append(idx)
             valid_indices = np.array(final_valid_indices, dtype=int)

        if len(valid_indices) == 0:
            return []
            
        # Reconstruct results
        p_indices = valid_indices // M
        res_lats = flat_lats[valid_indices]
        res_lons = flat_lons[valid_indices]
        res_twa = flat_twa[valid_indices]
        res_bearings_end = self.bearing_vectorized(res_lats, res_lons, self.end_point[0], self.end_point[1])
        
        new_routes = []
        for i, p in enumerate(p_indices):
            r = routes[p] + [(res_lats[i], res_lons[i])]
            new_routes.append(r)
            
        results = []
        for i in range(len(res_lats)):
             results.append([res_lats[i], res_lons[i], new_routes[i], res_bearings_end[i], res_twa[i]])
             
        return results

    def get_possible_optimized(self, lat_init, lon_init, route, bearing_end, t, time_step_idx, route_points, spacings, previous_twa=None):
        # Compatibility wrapper
        lats = np.array([lat_init])
        lons = np.array([lon_init])
        routes = [route]
        bearings = np.array([bearing_end])
        prev_twas = np.array([previous_twa]) if previous_twa is not None else None
        
        return self.get_possible_batch_optimized(lats, lons, routes, bearings, t, time_step_idx, route_points, spacings, prev_twas)

    def optimize(self, previous_route, previous_isochrones):
        """
        Run optimization pass using previous route and isochrones to constrain search space.
        Uses only prune_slow and prune_close_together for finer granularity.
        
        :param previous_route: list of (lat, lon) tuples from previous routing
        :param previous_isochrones: list of isochrone arrays from previous routing
        :return: optimized route
        """
        if previous_route is None or previous_isochrones is None:
            raise ValueError("optimize() requires previous_route and previous_isochrones (got None).")
        if len(previous_route) == 0 or len(previous_isochrones) == 0:
            raise ValueError(
                f"optimize() requires non-empty previous_route and previous_isochrones "
                f"(got {len(previous_route)=}, {len(previous_isochrones)=})."
            )
        if len(previous_route) != len(previous_isochrones):
            raise ValueError(
                "optimize() requires previous_route and previous_isochrones to have the same length "
                f"(got {len(previous_route)} route points vs {len(previous_isochrones)} isochrones)."
            )
        if len(self.time_steps) == 0:
            raise ValueError("optimize() requires non-empty self.time_steps.")

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

        # Initial State
        lats = np.array([lat])
        lons = np.array([lon])
        routes = [[self.start_point]]
        bearings = np.array([self.getBearing((lat, lon), self.end_point)])
        twas = None
        
        # Step 0
        t0 = self.time_steps[0]
        possible = self.get_possible_batch_optimized(
             lats, lons, routes, bearings, t0, 0, route_points, spacings, twas
        )
        if len(possible) == 0:
            raise RuntimeError("Optimization pass produced no valid candidates at step 0.")

        dist_wp = self.get_min_dist_wp(np.array(possible, dtype=object))
        print('step', 0, 'number of isochrone points', len(possible), 'dist to finish', f'{dist_wp:.1f}')
        if self.progress_callback:
            self.progress_callback(0, dist_wp, possible)
        self.optimized_isochrones.append(possible)

        if dist_wp <= self.finish_size:
            return self.get_fastest_route(use_optimized=True)

        for step, t in enumerate(self.time_steps[1:], start=1):
            
            # Prune
            possible = self.prune_slow(np.array(possible, dtype=object))
            possible, _ = self.prune_close_together(possible)
            
            if len(possible) > self.optimise_n_points:
                original_n_points = self.n_points
                self.n_points = self.optimise_n_points
                possible, _ = self.prune_equidistant(possible)
                self.n_points = original_n_points

            if len(possible) == 0:
                raise RuntimeError(f"Optimization pass produced no valid candidates at step {step}.")

            dist_wp = self.get_min_dist_wp(np.array(possible, dtype=object))
            print('step', step, 'number of isochrone points', len(possible), 'dist to finish', f'{dist_wp:.1f}')
            if self.progress_callback:
                self.progress_callback(step, dist_wp, possible)
            self.optimized_isochrones.append(possible)

            if dist_wp <= self.finish_size:
                break

            # Next Step
            arr = np.array(possible, dtype=object)
            lats = arr[:, 0].astype(float)
            lons = arr[:, 1].astype(float)
            routes = list(arr[:, 2])
            bearings = arr[:, 3].astype(float)
            twas = arr[:, 4].astype(float)
            
            possible = self.get_possible_batch_optimized(
                lats, lons, routes, bearings, t, step, route_points, spacings, twas
            )
            
            if len(possible) == 0:
                raise RuntimeError(f"Optimization pass produced no valid candidates after expanding step {step}.")
                    
            if step == len(self.time_steps) - 1:
                break
                
        return self.get_fastest_route(use_optimized=True)
