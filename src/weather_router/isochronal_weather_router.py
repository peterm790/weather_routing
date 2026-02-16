import numpy as np
import pandas as pd
import math
from numba import njit, prange

from .utils_geo import (
    _R_EARTH_NM,
    bearing_deg,
    gc_destination,
    haversine_nm_vec,
    haversine_nm_scalar,
)
from .utils_numba import polar_speed

@njit(cache=True, fastmath=True)
def _leg_finish_intersection_numba(
    lat0: float,
    lon0: float,
    heading_deg: float,
    leg_distance_nm: float,
    finish_lat: float,
    finish_lon: float,
    finish_size_nm: float,
):
    """
    Numba-accelerated finish-circle intersection.
    Returns (intersects, hit_lat, hit_lon, hit_distance_nm).
    If no hit, intersects=False and hit_* values are 0.0.
    """
    if leg_distance_nm < 0.0 or finish_size_nm <= 0.0:
        return False, 0.0, 0.0, 0.0

    # Great-circle distance from start to finish center (nm)
    dist_to_center_nm = haversine_nm_scalar(lat0, lon0, finish_lat, finish_lon)

    # Already inside finish circle
    if dist_to_center_nm <= finish_size_nm:
        return True, lat0, lon0, 0.0

    # Quick necessary condition: to hit a circle of radius R, the center must be within L+R
    if dist_to_center_nm > leg_distance_nm + finish_size_nm:
        return False, 0.0, 0.0, 0.0

    # Convert to angular distances (radians)
    delta13 = dist_to_center_nm / _R_EARTH_NM
    delta12 = leg_distance_nm / _R_EARTH_NM
    deltaR = finish_size_nm / _R_EARTH_NM

    # Bearings (radians)
    theta12 = math.radians(heading_deg % 360.0)
    theta13 = math.radians(bearing_deg(lat0, lon0, finish_lat, finish_lon))

    xt_arg = math.sin(delta13) * math.sin(theta13 - theta12)
    if xt_arg < -1.0:
        xt_arg = -1.0
    if xt_arg > 1.0:
        xt_arg = 1.0
    delta_xt = math.asin(xt_arg)

    if abs(delta_xt) > deltaR:
        return False, 0.0, 0.0, 0.0

    delta_at = math.atan2(
        math.sin(delta13) * math.cos(theta13 - theta12),
        math.cos(delta13),
    )

    cos_delta_xt = math.cos(delta_xt)
    if cos_delta_xt == 0.0:
        return False, 0.0, 0.0, 0.0

    h_arg = math.cos(deltaR) / cos_delta_xt
    if h_arg < -1.0:
        h_arg = -1.0
    if h_arg > 1.0:
        h_arg = 1.0
    delta_h = math.acos(h_arg)

    delta_i1 = delta_at - delta_h
    delta_i2 = delta_at + delta_h

    # Pick the first intersection within the segment [0, delta12]
    hit = False
    delta_hit = 0.0
    if 0.0 <= delta_i1 <= delta12:
        hit = True
        delta_hit = delta_i1
    if 0.0 <= delta_i2 <= delta12:
        if not hit or delta_i2 < delta_hit:
            hit = True
            delta_hit = delta_i2

    if not hit:
        return False, 0.0, 0.0, 0.0

    hit_distance_nm = delta_hit * _R_EARTH_NM
    hit_lat, hit_lon = gc_destination(lat0, lon0, heading_deg, hit_distance_nm)
    return True, hit_lat, hit_lon, hit_distance_nm

@njit(cache=True, fastmath=True, parallel=True)
def _check_route_proximity_numba(
    candidates_arr,    # Nx2 array (lat, lon)
    window_arr,        # Mx2 array (lat, lon) of route points
    current_step,
    window_start_idx,
    threshold
):
    """
    Returns a boolean mask of candidates that are within the 'behind' threshold.
    """
    n_candidates = len(candidates_arr)
    n_window = len(window_arr)
    keep = np.empty(n_candidates, dtype=np.bool_)

    for i in prange(n_candidates):
        c_lat = candidates_arr[i, 0]
        c_lon = candidates_arr[i, 1]
        
        min_dist = 1e9  # arbitrary large float
        best_window_idx = -1
        
        # Find closest point in the route window
        for j in range(n_window):
            w_lat = window_arr[j, 0]
            w_lon = window_arr[j, 1]
            
            # Using your existing imported scalar haversine
            dist = haversine_nm_scalar(c_lat, c_lon, w_lat, w_lon)
            
            if dist < min_dist:
                min_dist = dist
                best_window_idx = j
        
        # Logic: if closest point index is too far back in time, prune
        if best_window_idx != -1:
            closest_global_idx = window_start_idx + best_window_idx
            # If (current_step - closest_idx) < threshold, we keep it
            keep[i] = (current_step - closest_global_idx) < threshold
        else:
            keep[i] = False # Should not happen if window is populated

    return keep

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
            twa_change_penalty=0.02,
            twa_change_threshold=5.0,
            finish_size=20,
            optimise_n_points=None,
            optimise_window=24,
            progress_callback=None,
            avoid_land_crossings=False,
            leg_check_spacing_nm: float = 1.0,
                leg_check_max_samples: int = 25,
                land_threshold: float = 0.5,
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
            :param twa_change_penalty: float
                speed penalty (0.0-1.0) applied when TWA changes more than threshold (default 0.02)
            :param twa_change_threshold: float
                threshold in degrees for applying TWA change penalty (default 5.0)
            :param finish_size: int
                size of the finish area in nm
            :param optimise_n_points: int
                number of points to maintain in each isochrone during optimization (defaults to n_points*2)
            :param optimise_window: int
                time step window size (±) for optimization constraint search (default: 24)
            :param progress_callback: function, optional
                Callback function to report progress. Called with (step, dist_to_finish, isochrones).
            :param avoid_land_crossings: bool | str
                Land-crossing mode (leg validation):
                - 'point': validate only start+end points (does not detect crossings between them)
                - 'step': sample intermediate points along the leg (fast, can miss thin crossings)
                - 'strict': reject if ANY intersected land-sea-mask grid cell is land (conservative/slower)

                Backward-compatible bool forms:
                - False -> 'point'
                - True  -> 'step'
            :param leg_check_spacing_nm: float
                Target sampling spacing (nautical miles) along each candidate leg.
                Must be >= 0.25nm.
            :param leg_check_max_samples: int
                Maximum number of intermediate samples per candidate leg (performance cap).
            :param land_threshold: float
                Fractional land coverage threshold for 'strict' mode leg checks (0.0-1.0).
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
        self.twa_change_penalty = twa_change_penalty
        self.twa_change_threshold = twa_change_threshold
        self.finish_size = finish_size
        self.optimise_n_points = optimise_n_points if optimise_n_points is not None else n_points * 2
        self.optimise_window = optimise_window
        self.progress_callback = progress_callback
        # Fast polar lookup params for Numba
        self._polar_speed_table = self.polar.speedTable
        self._polar_tws_max = float(self.polar._tws_max)
        self._polar_tws_min = float(self.polar._tws_min)
        self._polar_twa_step = float(self.polar._twa_step)
        self._polar_twa_max = float(self.polar._twa_max)
        self._polar_twa_min = float(self.polar._twa_min)

        if leg_check_spacing_nm < 0.25:
            raise ValueError("leg_check_spacing_nm must be >= 0.25 (nautical miles)")
        if leg_check_max_samples < 1:
            raise ValueError("leg_check_max_samples must be >= 1")

        # Land-crossing mode:
        # - 'point': endpoint validity only (no segment crossing detection)
        # - 'step' : sparse intermediate-point sampling via point_validity()
        # - 'strict': full grid-cell intersection check using the underlying land-sea mask
        if isinstance(avoid_land_crossings, bool):
            self.avoid_land_crossings = 'step' if avoid_land_crossings else 'point'
        elif isinstance(avoid_land_crossings, str):
            mode = avoid_land_crossings.lower()
            if mode not in ('point', 'step', 'strict'):
                raise ValueError("avoid_land_crossings must be one of: 'point', 'step', 'strict'")
            self.avoid_land_crossings = mode
        else:
            raise ValueError("avoid_land_crossings must be a bool or one of: 'point', 'step', 'strict'")

        # Performance tweak:
        # - During the *main routing* pass, cap land-crossing checks at 'step' (never 'strict').
        # - During the *optimisation* pass, allow 'strict' when requested.
        if self.avoid_land_crossings == 'strict':
            self.avoid_land_crossings_route = 'step'
            self.avoid_land_crossings_optimise = 'strict'
        else:
            self.avoid_land_crossings_route = self.avoid_land_crossings
            self.avoid_land_crossings_optimise = self.avoid_land_crossings
        self.leg_check_spacing_nm = float(leg_check_spacing_nm)
        self.leg_check_max_samples = int(leg_check_max_samples)
        self.land_threshold = float(land_threshold)


        from . import point_validity
        land_sea_mask = point_validity.land_sea_mask
        if point_validity_extent:
            lsm = land_sea_mask(point_validity_extent, file=point_validity_file, method = point_validity_method)
        else:
            lsm = land_sea_mask(file=point_validity_file, method = point_validity_method)
        self._lsm = lsm
        self.point_validity = lsm.point_validity

    def leg_is_clear(self, lat0, lon0, heading, distance_nm, *, mode: str | None = None):
        """
        Return True if the great-circle leg stays on water.

        We check intermediate points along the same initial bearing used to compute the endpoint.
        Endpoint validity should be checked separately by the caller.
        """
        mode_to_use = mode if mode is not None else self.avoid_land_crossings_route

        if mode_to_use == 'point':
            return self._lsm.leg_is_clear(lat0, lon0, heading, distance_nm, mode='point')

        if distance_nm is None:
            return False

        try:
            distance_nm = float(distance_nm)
        except Exception:
            return False

        if distance_nm <= 0:
            return True

        if mode_to_use == 'strict':
            return self._lsm.leg_is_clear(
                lat0, lon0, heading, distance_nm,
                mode='strict',
                land_threshold=self.land_threshold,
            )

        # Step-wise sparse sampling mode
        return self._lsm.leg_is_clear(
            lat0, lon0, heading, distance_nm,
            mode='step',
            spacing_nm=self.leg_check_spacing_nm,
            max_samples=self.leg_check_max_samples,
        )

    def get_min_dist_wp(self, isochrones):
        lats = isochrones[:, 0].astype(float)
        lons = isochrones[:, 1].astype(float)
        d = haversine_nm_vec(lats, lons, self.end_point[0], self.end_point[1])
        return float(np.min(d))

    def get_dist_wp(self, isochrones):
        lats = np.asarray(isochrones)[:, 0].astype(float)
        lons = np.asarray(isochrones)[:, 1].astype(float)
        d = haversine_nm_vec(lats, lons, self.end_point[0], self.end_point[1])
        return d.tolist()

    def getBearing(self, pointA, pointB):
        return bearing_deg(float(pointA[0]), float(pointA[1]), float(pointB[0]), float(pointB[1]))

    def get_average_bearing(self, b1, b2):
        """
        Calculate the average of two bearings, handling the 0/360 wrap-around.
        """
        rad1 = math.radians(b1)
        rad2 = math.radians(b2)
        
        # Sum of unit vectors
        s = math.sin(rad1) + math.sin(rad2)
        c = math.cos(rad1) + math.cos(rad2)
        
        avg_rad = math.atan2(s, c)
        avg_deg = math.degrees(avg_rad)
        
        return (avg_deg + 360) % 360

    def getTWA_from_heading(self, bearing, TWD):
        TWA = bearing - TWD
        return (TWA + 180) % 360 - 180

    def leg_finish_intersection(self, lat0, lon0, heading_deg, leg_distance_nm):
        """
        Heading-aware check whether a great-circle leg intersects the finish circle.

        We treat the finish as a circle of radius `self.finish_size` (nautical miles)
        around `self.end_point`. If the *segment* (not the infinite great-circle) intersects,
        return the first intersection point along the segment.

        Returns:
            (intersects: bool, hit_lat: float|None, hit_lon: float|None, hit_distance_nm: float|None)
        """
        # Validate inputs
        try:
            lat0 = float(lat0)
            lon0 = float(lon0)
            heading_deg = float(heading_deg)
            leg_distance_nm = float(leg_distance_nm)
        except Exception:
            raise ValueError("leg_finish_intersection(): lat/lon/heading/leg_distance must be numeric.")

        if leg_distance_nm < 0:
            raise ValueError("leg_finish_intersection(): leg_distance_nm must be >= 0.")
        if self.finish_size <= 0:
            raise ValueError("leg_finish_intersection(): finish_size must be > 0.")
        finish_lat, finish_lon = self.end_point
        intersects, hit_lat, hit_lon, hit_distance_nm = _leg_finish_intersection_numba(
            lat0,
            lon0,
            heading_deg,
            leg_distance_nm,
            float(finish_lat),
            float(finish_lon),
            float(self.finish_size),
        )

        if not intersects:
            return False, None, None, None
        return True, hit_lat, hit_lon, hit_distance_nm

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

    def has_intersected_finish(self, lat_start, lon_start, lat_end, lon_end):
        """
        Check if the segment from start to end intersects the finish area.
        Approximates using Cartesian coordinates on a flat earth projection.
        """
        finish_lat, finish_lon = self.end_point
        radius_nm = self.finish_size
        
        # Approximate scale factor for longitude based on mean latitude
        mean_lat = (lat_start + finish_lat) / 2.0
        lon_scale = math.cos(math.radians(mean_lat))
        
        # Convert to local Cartesian coordinates (nm) relative to finish point
        # 1 degree lat = 60 nm
        y1 = (lat_start - finish_lat) * 60.0
        x1 = (lon_start - finish_lon) * 60.0 * lon_scale
        
        y2 = (lat_end - finish_lat) * 60.0
        x2 = (lon_end - finish_lon) * 60.0 * lon_scale
        
        # Vector from p1 to p2
        dx = x2 - x1
        dy = y2 - y1
        
        # Segment length squared
        l2 = dx*dx + dy*dy
        
        if l2 == 0:
            return x1*x1 + y1*y1 <= radius_nm*radius_nm
            
        # Project finish point (0,0) onto line containing segment
        # The vector from p1 to (0,0) is (-x1, -y1)
        # t = dot(p1_to_origin, p1_to_p2) / l2
        t = ( -x1 * dx - y1 * dy ) / l2
        
        # Clamp t to segment [0, 1] to find closest point on segment
        t_clamped = max(0, min(1, t))
        
        # Closest point coordinates
        closest_x = x1 + t_clamped * dx
        closest_y = y1 + t_clamped * dy
        
        # Distance squared to closest point
        dist_sq = closest_x*closest_x + closest_y*closest_y
        
        return dist_sq <= radius_nm*radius_nm

    def is_twa_change(self, current_twa, previous_twa):
        """
        Determine if the TWA changed significantly (exceeds threshold), but not a tack.
        """
        if previous_twa is None:
            return False
        
        # If tacking, let the tack penalty handle it
        if self.is_tacking(current_twa, previous_twa):
            return False

        diff = abs(current_twa - previous_twa)
        if diff > 180:
            diff = 360 - diff
            
        return diff > self.twa_change_threshold

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

    def prune_slow(self, arr, waypoint=None):
        keep = [True] * len(arr)
        for i in range(len(arr)):
            if keep[i] is True:
                bearing_to_finish = arr[i][3]
                if waypoint:
                    bearing_to_wp = self.getBearing((arr[i][0], arr[i][1]), waypoint)
                    #target_bearing = self.get_average_bearing(bearing_to_finish, bearing_to_wp)
                    target_bearing = bearing_to_wp
                    wake = self.get_wake_lims(target_bearing)
                else:
                    wake = self.get_wake_lims(bearing_to_finish)

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
        # Sort points by bearing from start point to ensure they are ordered along the wavefront
        points = isochrone
        bearings = [self.getBearing(self.start_point, p) for p in points]
        
        # Handle wrap-around at 0/360 degrees if the sector spans across North
        if bearings and (max(bearings) - min(bearings) > 180):
                bearings = [b + 360 if b < 180 else b for b in bearings]
        
        # Sort points based on the calculated bearings
        points_with_bearings = sorted(zip(points, bearings), key=lambda x: x[1])
        sort_iso = [p for p, _ in points_with_bearings]

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
        Selects a subset of the original points whose cumulative distances along the
        isochrone curve are closest to uniformly spaced targets (in nautical miles).
        """
        if len(possible) == 0:
            return np.array([], dtype=object), float("inf")

        arr = np.array(possible, dtype=object)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("prune_equidistant expected rows like [lat, lon, route, brg, twa].")

        # Extract original indices and coordinates
        pts = [(float(row[0]), float(row[1]), i) for i, row in enumerate(arr)]
        if len(pts) < 2:
            raise ValueError("prune_equidistant(): need at least 2 points to compute spacing.")

        # Order points by unwrapped bearing from the start point to reduce discontinuities at 0/360
        # Compute angles in radians
        angles = [math.radians(self.getBearing(self.start_point, (lat, lon))) for (lat, lon, _) in pts]
        # Circular mean
        s = sum(math.sin(a) for a in angles)
        c = sum(math.cos(a) for a in angles)
        mean_ang = math.atan2(s, c)

        def unwrap_near_mean(a, mean):
            # Shift a by multiples of 2*pi so it lies within +/-pi of mean
            while a - mean > math.pi:
                a -= 2.0 * math.pi
            while mean - a > math.pi:
                a += 2.0 * math.pi
            return a

        unwrapped = [unwrap_near_mean(a, mean_ang) for a in angles]
        ordered = [p for _, p in sorted(zip(unwrapped, pts), key=lambda t: t[0])]

        # Cumulative geodesic distances (nm) along the ordered polyline
        cum = [0.0]
        for i in range(len(ordered) - 1):
            (lat1, lon1, _), (lat2, lon2, _) = ordered[i], ordered[i + 1]
            d_nm = haversine_nm_scalar(lat1, lon1, lat2, lon2)
            cum.append(cum[-1] + float(d_nm))

        total_len = cum[-1]
        if not math.isfinite(total_len) or total_len <= 0.0:
            raise ValueError("prune_equidistant(): degenerate isochrone (zero total length).")

        # Number of desired samples limited by available unique points
        n_target = min(int(self.n_points), len(ordered))
        if n_target < 1:
            raise ValueError("prune_equidistant(): invalid n_points computed.")

        targets = np.linspace(0.0, total_len, n_target)
        cum_arr = np.asarray(cum, dtype=float)

        # For each target distance, pick the original vertex closest in cumulative distance
        # Map back to ORIGINAL indices from input 'possible'
        selected_orig_indices = []
        for t in targets:
            idx_along = int(np.argmin(np.abs(cum_arr - t)))
            orig_idx = ordered[idx_along][2]
            if len(selected_orig_indices) == 0 or selected_orig_indices[-1] != orig_idx:
                selected_orig_indices.append(orig_idx)

        if len(selected_orig_indices) == 0:
            raise ValueError("prune_equidistant(): failed to select any indices.")

        selected_arr = arr[selected_orig_indices]
        dists_wp = self.get_dist_wp(selected_arr)
        dist_wp_min = float(np.min(dists_wp)) if len(dists_wp) else float("inf")
        return selected_arr, dist_wp_min

    def prune_behind_route_step(self, possible, step, route_points, window_size, behind_threshold_steps=3):
        """
        Prune candidates that are too far behind the previous route.

        For each candidate point, find the closest `route_points[i]` (searching within a bounded
        window around the current `step` for performance). If the closest route step index is
        `behind_threshold_steps` or more behind the current `step`, prune the candidate.

        Notes:
        - If `step >= len(route_points)`, this pruning is disabled and `possible` is returned unchanged.
        - No implicit fallbacks: invalid inputs raise.
        """
        if route_points is None:
            raise ValueError("prune_behind_route_step(): route_points must not be None.")
        if step is None or step < 0:
            raise ValueError(f"prune_behind_route_step(): step must be a non-negative int (got {step}).")
        if window_size is None or window_size < 0:
            raise ValueError(
                f"prune_behind_route_step(): window_size must be a non-negative int (got {window_size})."
            )
        if behind_threshold_steps is None or behind_threshold_steps < 0:
            raise ValueError(
                "prune_behind_route_step(): behind_threshold_steps must be a non-negative int "
                f"(got {behind_threshold_steps})."
            )

        n_route = len(route_points)
        if n_route == 0:
            raise ValueError("prune_behind_route_step(): route_points must be non-empty.")
        if step >= n_route:
            return possible

        arr = np.array(possible, dtype=object)
        if arr.size == 0:
            return arr
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(
                "prune_behind_route_step(): expected candidates shaped like [lat, lon, ...] per row "
                f"(got arr.ndim={arr.ndim}, arr.shape={arr.shape})."
            )

        start_idx = max(0, step - window_size)
        end_idx = min(n_route, step + window_size + 1)
        if start_idx >= end_idx:
            raise ValueError(
                "prune_behind_route_step(): empty search window. "
                f"Computed [{start_idx}, {end_idx}) for {step=}, {window_size=}, {n_route=}."
            )

        # Pre-slice the window to avoid slicing inside the loop
        # and ensure we are working with simple floats
        window_subset = route_points[start_idx:end_idx]

        # Extract candidate coordinates as Nx2 array
        candidates = np.array([[float(row[0]), float(row[1])] for row in arr], dtype=np.float64)
        
        # Convert window subset to Mx2 array
        window_arr = np.array([[float(r_lat), float(r_lon)] for r_lat, r_lon in window_subset], dtype=np.float64)
        
        # Use Numba-accelerated function
        keep_mask = _check_route_proximity_numba(
            candidates,
            window_arr,
            step,
            start_idx,
            behind_threshold_steps
        )

        return arr[keep_mask]

    def get_possible(self, lat_init, lon_init, route, bearing_end, t, previous_twa=None):
        possible = []
        twd, tws = self.get_wind(t, lat_init, lon_init)
        # Distance to finish center (nm) used for a quick necessary-condition check
        dist_to_finish = haversine_nm_scalar(lat_init, lon_init, self.end_point[0], self.end_point[1])

        upper = int(bearing_end) + self.spread
        lower = int(bearing_end) - self.spread
        route.append('dummy')
        for heading in range(lower, upper, 10):
            heading = ((int(heading) + 360) % 360)
            twa = self.getTWA_from_heading(heading, twd)
            speed = polar_speed(
                tws,
                abs(twa),
                self._polar_speed_table,
                self._polar_tws_max,
                self._polar_tws_min,
                self._polar_twa_step,
                self._polar_twa_max,
                self._polar_twa_min,
            )

            # Apply tack penalty if tacking
            if self.is_tacking(twa, previous_twa):
                speed = speed * (1.0 - self.tack_penalty)
            
            lat, lon = gc_destination(lat_init, lon_init, heading, speed * self.step)

            route = route[:-1]
            route.append((lat, lon))
            
            # Default: theoretical leg length for this step
            leg_distance_nm = speed * self.step

            # Heading-aware finish-circle intersection:
            # Only attempt the more expensive intersection math if the finish circle is reachable in principle.
            if dist_to_finish <= leg_distance_nm + self.finish_size:
                intersects, hit_lat, hit_lon, hit_dist_nm = _leg_finish_intersection_numba(
                    float(lat_init),
                    float(lon_init),
                    float(heading),
                    float(leg_distance_nm),
                    float(self.end_point[0]),
                    float(self.end_point[1]),
                    float(self.finish_size),
                )
                if intersects:
                    lat, lon = hit_lat, hit_lon
                    route = route[:-1]
                    route.append((lat, lon))
                    leg_distance_nm = hit_dist_nm

            if self.point_validity(lat, lon) and self.leg_is_clear(lat_init, lon_init, heading, leg_distance_nm):
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
                        if self.progress_callback:
                            self.progress_callback(step, dist_wp, possible)
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
            df['is_twa_change'] = False
            df['boat_speed'] = df['base_boat_speed']
            
            for i in range(1, len(df)):
                current_twa = df.iloc[i]['twa']
                previous_twa = df.iloc[i-1]['twa']
                is_tacking = self.is_tacking(current_twa, previous_twa)
                is_twa_change = self.is_twa_change(current_twa, previous_twa)

                df.iloc[i, df.columns.get_loc('is_tacking')] = is_tacking
                df.iloc[i, df.columns.get_loc('is_twa_change')] = is_twa_change

                if is_tacking:
                    df.iloc[i, df.columns.get_loc('boat_speed')] = df.iloc[i]['base_boat_speed'] * (1.0 - self.tack_penalty)
                elif is_twa_change:
                    df.iloc[i, df.columns.get_loc('boat_speed')] = df.iloc[i]['base_boat_speed'] * (1.0 - self.twa_change_penalty)
            
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
            
            # Sort points by bearing from start point to ensure they are ordered along the wavefront
            bearings = [self.getBearing(self.start_point, p) for p in points]
            
            # Handle wrap-around at 0/360 degrees if the sector spans across North
            if bearings and (max(bearings) - min(bearings) > 180):
                bearings = [b + 360 if b < 180 else b for b in bearings]
            
            # Sort points based on the calculated bearings
            points_with_bearings = sorted(zip(points, bearings), key=lambda x: x[1])
            sort_points = [p for p, _ in points_with_bearings]
            
            # Calculate cumulative distances using geographical distances
            total_distance = 0.0
            for i in range(len(sort_points) - 1):
                lat1, lon1 = sort_points[i]
                lat2, lon2 = sort_points[i + 1]
                dist_nm = haversine_nm_scalar(lat1, lon1, lat2, lon2)
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
            distance_to_center = haversine_nm_scalar(lat, lon, center_lat, center_lon)
            if distance_to_center < min_distance_to_center:
                min_distance_to_center = distance_to_center
                closest_spacing = spacings[min(i, len(spacings) - 1)]
        
        # Use the closest spacing as constraint radius (in nautical miles)
        constraint_radius = closest_spacing * 2
        
        return min_distance_to_center <= constraint_radius

    def get_possible_optimized(self, lat_init, lon_init, route, bearing_end, t, time_step_idx, route_points, spacings, previous_twa=None):
        """
        Generate possible next positions with optimization region constraints.
        """
        possible = []
        twd, tws = self.get_wind(t, lat_init, lon_init)
        
        # Distance to finish center (nm) used for a quick necessary-condition check
        dist_to_finish = haversine_nm_scalar(lat_init, lon_init, self.end_point[0], self.end_point[1])

        upper = int(bearing_end) + self.spread
        lower = int(bearing_end) - self.spread
        route.append('dummy')
        
        for heading in range(lower, upper, 10):
            heading = ((int(heading) + 360) % 360)
            twa = self.getTWA_from_heading(heading, twd)
            speed = polar_speed(
                tws,
                abs(twa),
                self._polar_speed_table,
                self._polar_tws_max,
                self._polar_tws_min,
                self._polar_twa_step,
                self._polar_twa_max,
                self._polar_twa_min,
            )
            
            # Apply tack penalty if tacking
            if self.is_tacking(twa, previous_twa):
                speed = speed * (1.0 - self.tack_penalty)
            elif self.is_twa_change(twa, previous_twa):
                speed = speed * (1.0 - self.twa_change_penalty)
            
            lat, lon = gc_destination(lat_init, lon_init, heading, speed * self.step)

            route = route[:-1]
            route.append((lat, lon))
            
            # Check point validity and optimization region constraints
            leg_distance_nm = speed * self.step

            intersects = False
            if dist_to_finish <= leg_distance_nm + self.finish_size:
                intersects, hit_lat, hit_lon, hit_dist_nm = _leg_finish_intersection_numba(
                    float(lat_init),
                    float(lon_init),
                    float(heading),
                    float(leg_distance_nm),
                    float(self.end_point[0]),
                    float(self.end_point[1]),
                    float(self.finish_size),
                )
                if intersects:
                    lat, lon = hit_lat, hit_lon
                    route = route[:-1]
                    route.append((lat, lon))
                    leg_distance_nm = hit_dist_nm

            if (self.point_validity(lat, lon) and
                self.leg_is_clear(lat_init, lon_init, heading, leg_distance_nm, mode=self.avoid_land_crossings_optimise) and
                (intersects or self.is_within_optimization_region(lat, lon, time_step_idx, route_points, spacings))):
                bearing_end = self.getBearing((lat, lon), self.end_point)
                possible.append([lat, lon, route, bearing_end, twa])
        
        return possible

    def get_possible_batch_optimized(self, lats, lons, routes, bearings_end, t, time_step_idx, route_points, spacings, previous_twas=None):
        """
        Batch wrapper around `get_possible_optimized`.

        The routing logic in this module is pointwise (non-vectorized). This method exists so
        `optimize()` can expand a whole frontier in one call.
        """
        if previous_twas is None:
            previous_twas = [None] * len(lats)
        if not (len(lats) == len(lons) == len(routes) == len(bearings_end) == len(previous_twas)):
            raise ValueError("get_possible_batch_optimized(): input arrays/lists must all have the same length.")

        possible_at_t = []
        for lat_i, lon_i, route_i, brg_i, prev_twa_i in zip(lats, lons, routes, bearings_end, previous_twas):
            possible_at_t.append(
                self.get_possible_optimized(
                    lat_i, lon_i, route_i, brg_i, t, time_step_idx, route_points, spacings, prev_twa_i
                )
            )
        return sum(possible_at_t, [])

    def prune_close_together(self, possible):
        """
        Remove points that are very close together (same rounded lat/lon bucket),
        keeping the best (closest-to-finish) representative per bucket.
        """
        if len(possible) == 0:
            return np.array([], dtype=object), float("inf")

        arr = np.array(possible, dtype=object)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError("prune_close_together expected rows like [lat, lon, route, brg, twa].")

        df = pd.DataFrame(arr)
        df['dist_wp'] = self.get_dist_wp(arr)
        df = df.sort_values('dist_wp')

        df['tups'] = df.apply(lambda x: (round(float(x[0]), self.rounding), round(float(x[1]), self.rounding)), axis=1)
        df = df.drop_duplicates(subset=['tups'], keep='first')

        dist_wp_min = float(df['dist_wp'].min()) if len(df) else float("inf")
        return df.iloc[:, :5].to_numpy(dtype=object), dist_wp_min


    def _normalize_route_points(self, route):
        """
        Normalize various route representations into a clean list of (lat, lon) float tuples.
        - Accepts: list/ndarray of (lat, lon) pairs, possibly mixed with markers like 'dummy'
                   or a pandas DataFrame with 'lat'/'lon' columns.
        - Drops any non-numeric or non-2D entries.
        """
        cleaned = []
        try:
            if isinstance(route, pd.DataFrame):
                if 'lat' in route.columns and 'lon' in route.columns:
                    lats = route['lat'].astype(float).tolist()
                    lons = route['lon'].astype(float).tolist()
                    cleaned = list(zip(lats, lons))
                else:
                    # Fallback: use first two columns if present
                    if route.shape[1] >= 2:
                        lats = route.iloc[:, 0].astype(float).tolist()
                        lons = route.iloc[:, 1].astype(float).tolist()
                        cleaned = list(zip(lats, lons))
                    else:
                        cleaned = []
            else:
                for item in list(route) if isinstance(route, (list, tuple, np.ndarray)) else []:
                    if isinstance(item, (list, tuple, np.ndarray)) and len(item) >= 2:
                        try:
                            lat = float(item[0])
                            lon = float(item[1])
                            cleaned.append((lat, lon))
                        except Exception:
                            continue
        except Exception:
            cleaned = []
        return cleaned

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
        if len(self.time_steps) == 0:
            raise ValueError("optimize() requires non-empty self.time_steps.")

        # Outer loop: repeat passes if equidistant pruning occurred, up to a safety cap
        max_passes = 5
        pass_idx = 0
        curr_route = previous_route
        curr_isochrones = previous_isochrones
        while True:
            clamped_all_this_pass = False
            # Sanitize input route for this pass (remove markers like 'dummy', coerce DataFrame, etc.)
            route_clean = self._normalize_route_points(curr_route)
            if len(route_clean) == 0:
                raise ValueError("optimize(): previous_route normalized to empty list; cannot proceed.")
            
            # Calculate spacing constraints from the current isochrones
            spacings = self.calculate_isochrone_spacing(curr_isochrones)
            #spacings = [s * 1 for s in base_spacings]

            # Clamp on all passes if any spacing drops below finish_size
            if len(spacings) > 0 and spacings[0] < self.finish_size:
                spacings = [self.finish_size] * len(spacings)
                clamped_all_this_pass = True
        
            # Extract route points for constraint centers.
            # Common convention: routes include the start point, so they are often 1 longer than isochrones.
            n_route = len(route_clean)
            n_iso = len(curr_isochrones)
            if n_route == n_iso + 1:
                # Align: isochrone index 0 corresponds to "after the first move" => route[1]
                aligned_route = route_clean[1:]
            elif n_route == n_iso:
                aligned_route = route_clean
            else:
                raise ValueError(
                    "optimize() requires len(previous_route) to be either equal to len(previous_isochrones) "
                    "or exactly one longer (to include the start point). "
                    f"Got {n_route} route points vs {n_iso} isochrones."
                )

            route_points = [(float(point[0]), float(point[1])) for point in aligned_route]
            if len(route_points) != len(spacings):
                raise ValueError(
                    "optimize() internal alignment error: expected route_points and spacings to be the same length "
                    f"(got {len(route_points)} route points vs {len(spacings)} spacings)."
                )

            lat, lon = self.start_point
            self.optimized_isochrones = []
            used_equidistant = False
            
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
                try:
                    self.progress_callback(0, dist_wp, possible, pass_idx=pass_idx)
                except TypeError:
                    # Backward-compat: older callbacks without pass_idx
                    self.progress_callback(0, dist_wp, possible)
            self.optimized_isochrones.append(possible)

            if dist_wp <= self.finish_size:
                return self.get_fastest_route(use_optimized=True)

            for step, t in enumerate(self.time_steps[1:], start=1):
                
                # Prune
                window_size = self.optimise_window * 2  # wider window for robustness during optimisation

                possible = self.prune_behind_route_step(
                    possible,
                    step=step,
                    route_points=route_points,
                    window_size=window_size,
                    behind_threshold_steps=3,
                )
                if len(possible) == 0:
                    raise RuntimeError(
                        f"Optimization pass produced no valid candidates at step {step} after step-behind pruning."
                    )

                # Find the best point (closest to finish) in the current set
                arr_possible = np.array(possible, dtype=object)
                dists = self.get_dist_wp(arr_possible)
                best_idx = np.argmin(dists)
                best_lat, best_lon = arr_possible[best_idx, 0], arr_possible[best_idx, 1]

                # Find closest point in previous route within window
                # We search a window around the current step to handle overtaking/falling behind
                start_idx = max(0, step - window_size)
                end_idx = min(len(route_points), step + window_size + 1)
                
                min_dist = float('inf')
                closest_idx = step if step < len(route_points) else len(route_points) - 1 # Default fallback
                
                for i in range(start_idx, end_idx):
                    r_lat, r_lon = route_points[i]
                    dist = haversine_nm_scalar(r_lat, r_lon, best_lat, best_lon)
                    if dist < min_dist:
                        min_dist = dist
                        closest_idx = i
                
                # Target a waypoint further ahead to stabilize the bearing during tacking
                # +5 steps ahead provides a better "general direction" than the immediate next point
                lookahead_steps = 5
                target_idx = closest_idx + lookahead_steps
                
                # If we are near the end of the previous route, target the final destination
                if target_idx < len(route_points):
                    waypoint = route_points[target_idx]
                else:
                    waypoint = self.end_point
                
                possible = self.prune_slow(arr_possible, waypoint=waypoint)
                possible, dist_wp = self.prune_close_together(possible)
                
                if len(possible) > self.optimise_n_points:
                    used_equidistant = True
                    print('pruning in optimise!')
                    original_n_points = self.n_points
                    self.n_points = self.optimise_n_points
                    possible, _ = self.prune_equidistant(possible)
                    self.n_points = original_n_points # a hack must fix

                if len(possible) == 0:
                    raise RuntimeError(f"Optimization pass produced no valid candidates at step {step}.")

                print('step', step, 'number of isochrone points', len(possible), 'dist to finish', f'{dist_wp:.1f}')
                if self.progress_callback:
                    try:
                        self.progress_callback(step, dist_wp, possible, pass_idx=pass_idx)
                    except TypeError:
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
            
            # If we completed a pass without needing equidistant pruning, return the result
            if not used_equidistant or clamped_all_this_pass:
                return self.get_fastest_route(use_optimized=True)
            
            # End of pass: emit a preliminary route snapshot before starting next pass
            if self.progress_callback:
                try:
                    self.progress_callback(step, dist_wp, possible, pass_idx=pass_idx, emit_prelim=True)
                except TypeError:
                    try:
                        self.progress_callback(step, dist_wp, possible)
                    except Exception:
                        pass
            
            # Prepare for next pass using the optimized results
            pass_idx += 1
            if pass_idx >= max_passes:
                raise RuntimeError("optimize(): exceeded max passes without eliminating equidistant pruning")
            
            # Feed back optimized isochrones and route as the new constraints
            curr_isochrones = self.optimized_isochrones
            curr_route = self.get_fastest_route(stats=False, use_optimized=True)
