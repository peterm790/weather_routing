"""passing in point validity is marginally faster not sure why
point validity is the slowest portion of the code"""



import xarray as xr
import os
import math
from numba import njit

from .utils_geo import gc_destination
from .utils_numba import nearest_index, wrap_lon_to_domain


LAND_BORDER_PAD_CELLS = 100


@njit(cache=True, fastmath=True)
def _dda_cells_clear_core(lsm_arr, lat_origin, lon_origin, dlat, dlon, lat0, lon0, lat1, lon1, land_threshold) -> bool:
    """
    Numba-accelerated DDA traversal on a regular grid in (lat,lon) space.
    lsm_arr: 2D numpy array [lat, lon]
    Returns False if any intersected cell is land or traversal goes OOB.
    """
    nlat = lsm_arr.shape[0]
    nlon = lsm_arr.shape[1]

    # Map to continuous index space where cell edges are integers.
    y0 = (float(lat0) - lat_origin) / dlat
    x0 = (float(lon0) - lon_origin) / dlon
    y1 = (float(lat1) - lat_origin) / dlat
    x1 = (float(lon1) - lon_origin) / dlon

    iy = int(math.floor(y0))
    ix = int(math.floor(x0))
    end_iy = int(math.floor(y1))
    end_ix = int(math.floor(x1))

    # Out-of-bounds => treat as land (reject)
    if ix < 0 or ix >= nlon or iy < 0 or iy >= nlat:
        return False
    if end_ix < 0 or end_ix >= nlon or end_iy < 0 or end_iy >= nlat:
        return False

    if float(lsm_arr[iy, ix]) >= float(land_threshold):
        return False

    dx = x1 - x0
    dy = y1 - y0

    if dx == 0.0 and dy == 0.0:
        return True

    stepx = 0 if dx == 0.0 else (1 if dx > 0.0 else -1)
    stepy = 0 if dy == 0.0 else (1 if dy > 0.0 else -1)

    inf = 1e30
    if dx == 0.0:
        tMaxX = inf
        tDeltaX = inf
    else:
        next_x_boundary = (ix + 1) if stepx > 0 else ix
        tMaxX = (next_x_boundary - x0) / dx
        tDeltaX = 1.0 / abs(dx)

    if dy == 0.0:
        tMaxY = inf
        tDeltaY = inf
    else:
        next_y_boundary = (iy + 1) if stepy > 0 else iy
        tMaxY = (next_y_boundary - y0) / dy
        tDeltaY = 1.0 / abs(dy)

    max_iters = int(abs(end_ix - ix) + abs(end_iy - iy) + 10)
    for _ in range(max_iters):
        if ix == end_ix and iy == end_iy:
            return True

        if tMaxX < tMaxY:
            ix += stepx
            tMaxX += tDeltaX
        elif tMaxY < tMaxX:
            iy += stepy
            tMaxY += tDeltaY
        else:
            # corner hit: step both
            ix += stepx
            iy += stepy
            tMaxX += tDeltaX
            tMaxY += tDeltaY

        if ix < 0 or ix >= nlon or iy < 0 or iy >= nlat:
            return False
        if float(lsm_arr[iy, ix]) >= float(land_threshold):
            return False

    # If we didn't converge, be conservative.
    return False

def _pad_with_land_border(lsm: xr.DataArray, pad: int = 1, land_value: float = 1.0) -> xr.DataArray:
    """
    Return a new DataArray with a constant land border of width `pad`.

    This is used to prevent `method="nearest"` selection from snapping out-of-domain
    points to an edge *water* cell (which can let routes leak outside a bbox).
    """
    if pad <= 0:
        return lsm

    if 'latitude' not in lsm.coords or 'longitude' not in lsm.coords:
        raise ValueError("Land-sea mask must have 'latitude' and 'longitude' coordinates")

    lats = lsm['latitude'].values
    lons = lsm['longitude'].values

    if lats.size < 2 or lons.size < 2:
        raise ValueError("Land-sea mask must have at least 2 latitude and 2 longitude points to pad")

    # Infer step sizes (supports descending latitude and either lon direction).
    lat_step = float(lats[0] - lats[1])
    lon_step = float(lons[1] - lons[0])
    if lat_step == 0.0 or lon_step == 0.0:
        raise ValueError("Cannot infer grid step from land-sea mask coordinates")

    new_lats = (
        [float(lats[0] + lat_step * i) for i in range(pad, 0, -1)]
        + [float(x) for x in lats]
        + [float(lats[-1] - lat_step * i) for i in range(1, pad + 1)]
    )
    new_lons = (
        [float(lons[0] - lon_step * i) for i in range(pad, 0, -1)]
        + [float(x) for x in lons]
        + [float(lons[-1] + lon_step * i) for i in range(1, pad + 1)]
    )

    # Pure-xarray: reindex onto the expanded grid, filling the new border with land.
    return lsm.reindex(latitude=new_lats, longitude=new_lons, fill_value=land_value)


class land_sea_mask():
    def __init__ (self, extent=None, file=None, method = 'nearest'):
        """
            :param extent [lat1,lon1,lat2,lon2]
            :param file: path to nc file or xarray dataset
        """
        self.method = method

        if file is None:
            lsm = xr.open_dataset(os.path.join(os.path.dirname(__file__), 'data/era5_land-sea-mask.nc'))
        elif isinstance(file, xr.Dataset):
            lsm = file
        else:
            lsm = xr.open_dataset(file)
        
        if 'longitude' in lsm.coords:
            lsm.coords['longitude'] = (lsm.coords['longitude'] + 180) % 360 - 180
            lsm = lsm.sortby(lsm.longitude)
        
        if 'time' in lsm:
            lsm = lsm.isel(time = 0)
        
        if 'lsm' in lsm:
            lsm = lsm.lsm

        if extent:
            lat1,lon1,lat2,lon2 = extent
            lsm = lsm.sel(latitude = slice(max([lat1, lat2]),min([lat1, lat2]))).sel(longitude = slice(min([lon1, lon2]),max([lon1, lon2])))

        # Pad the mask by 1 cell on each side with "land" so points that drift outside
        # the requested extent snap to land when using method='nearest'.
        lsm = _pad_with_land_border(lsm, pad=LAND_BORDER_PAD_CELLS, land_value=1.0)
        
        lsm = lsm.load()

        self.lsm = lsm
        self.lsm_arr = lsm.values
        self.lats = list(lsm.latitude.values)
        self.lons = list(lsm.longitude.values)
        self._lat_vals = lsm.latitude.values.astype('float64')
        self._lon_vals = lsm.longitude.values.astype('float64')
        self._lat_asc = bool(self._lat_vals[0] <= self._lat_vals[-1])
        self._lon_asc = bool(self._lon_vals[0] <= self._lon_vals[-1])
        self._lon_min = float(self._lon_vals.min())
        self._lon_max = float(self._lon_vals.max())

    @staticmethod
    def _wrap_lon(lon: float) -> float:
        """Wrap longitude to [-180, 180)."""
        return (float(lon) + 180.0) % 360.0 - 180.0

    def _grid_params(self):
        lats = self.lsm['latitude'].values
        lons = self.lsm['longitude'].values
        if lats.size < 2 or lons.size < 2:
            raise ValueError("Land-sea mask must have at least 2 latitude and 2 longitude points")
        lat_step = float(lats[1] - lats[0])
        lon_step = float(lons[1] - lons[0])
        if lat_step == 0.0 or lon_step == 0.0:
            raise ValueError("Cannot infer grid step from land-sea mask coordinates")
        return float(lats[0]), float(lons[0]), lat_step, lon_step, int(lats.size), int(lons.size)

    def _cell_is_land(self, iy: int, ix: int, land_threshold: float) -> bool:
        # iy: latitude index, ix: longitude index
        return float(self.lsm_arr[iy, ix]) >= float(land_threshold)

    def _dda_cells_clear(self, lat0: float, lon0: float, lat1: float, lon1: float, land_threshold: float) -> bool:
        """
        Check all grid cells intersected by a straight segment in (lat,lon) space.
        Returns False if any intersected cell is land, or if traversal goes OOB.
        """
        lat_origin, lon_origin, dlat, dlon, _, _ = self._grid_params()
        return _dda_cells_clear_core(self.lsm_arr, lat_origin, lon_origin, dlat, dlon, lat0, lon0, lat1, lon1, land_threshold)

    def _segment_cells_clear_wrapped(self, lat0: float, lon0: float, lat1: float, lon1: float, land_threshold: float) -> bool:
        """
        Dateline-safe wrapper around _dda_cells_clear(): splits segments that cross ±180°.
        """
        eps = 1e-9
        lon0w = self._wrap_lon(lon0)
        lon1w = self._wrap_lon(lon1)

        # Make lon1 close to lon0 (short way around) for splitting.
        while lon1w - lon0w > 180.0:
            lon1w -= 360.0
        while lon1w - lon0w < -180.0:
            lon1w += 360.0

        if -180.0 <= lon1w <= 180.0:
            return self._dda_cells_clear(lat0, lon0w, lat1, lon1w, land_threshold)

        # Segment crosses the dateline; split into two parts.
        if lon1w > 180.0:
            boundary1 = 180.0 - eps
            boundary2 = -180.0 + eps
            denom = (lon1w - lon0w)
            if denom == 0.0:
                return False
            t = (boundary1 - lon0w) / denom
            latm = float(lat0) + t * (float(lat1) - float(lat0))
            # First: lon0 -> +180, Second: -180 -> lon1-360
            if not self._dda_cells_clear(lat0, lon0w, latm, boundary1, land_threshold):
                return False
            return self._dda_cells_clear(latm, boundary2, lat1, lon1w - 360.0, land_threshold)

        if lon1w < -180.0:
            boundary1 = -180.0 + eps
            boundary2 = 180.0 - eps
            denom = (lon1w - lon0w)
            if denom == 0.0:
                return False
            t = (boundary1 - lon0w) / denom
            latm = float(lat0) + t * (float(lat1) - float(lat0))
            if not self._dda_cells_clear(lat0, lon0w, latm, boundary1, land_threshold):
                return False
            return self._dda_cells_clear(latm, boundary2, lat1, lon1w + 360.0, land_threshold)

        return False

    def leg_is_clear_strict(
        self,
        lat0: float,
        lon0: float,
        heading_deg: float,
        distance_nm: float,
        land_threshold: float = 0.5,
        max_subsegments: int = 10000,
    ) -> bool:
        """
        Strictly check that a great-circle leg does not touch land.

        This rejects the leg if ANY intersected land-sea-mask grid cell has lsm >= land_threshold.
        If the traversal goes out of the loaded mask domain, it is treated as land (reject).
        """
        if distance_nm is None:
            return False
        try:
            distance_nm = float(distance_nm)
            heading_deg = float(heading_deg)
            lat0 = float(lat0)
            lon0 = float(lon0)
        except Exception:
            return False

        if distance_nm <= 0.0:
            return True

        # Choose subsegment length based on the mask native latitude step (deg -> nm).
        # 1 degree latitude ~= 60 nm.
        lat_origin, lon_origin, dlat, dlon, nlat, nlon = self._grid_params()
        step_nm = 0.5 * 60.0 * abs(float(dlat))
        if step_nm <= 0.0:
            raise ValueError("Invalid land-sea mask grid step; cannot compute strict sampling step.")

        n_steps = int(math.ceil(distance_nm / step_nm))
        if n_steps < 1:
            n_steps = 1
        if n_steps > int(max_subsegments):
            raise ValueError(
                f"Strict land check would require {n_steps} subsegments (distance_nm={distance_nm:.3f}, step_nm={step_nm:.3f}). "
                f"Increase max_subsegments if you really want this."
            )

        # Generate polyline points along the great-circle using the same bearing model as the router.
        pts = []
        for k in range(n_steps + 1):
            d = distance_nm * (k / n_steps)
            lat_p, lon_p = gc_destination(lat0, lon0, heading_deg, float(d))
            pts.append((float(lat_p), float(lon_p)))

        for (a_lat, a_lon), (b_lat, b_lon) in zip(pts[:-1], pts[1:]):
            if not self._segment_cells_clear_wrapped(a_lat, a_lon, b_lat, b_lon, land_threshold):
                return False

        return True

    def leg_is_clear_sparse(
        self,
        lat0: float,
        lon0: float,
        heading_deg: float,
        distance_nm: float,
        spacing_nm: float = 1.0,
        max_samples: int = 25,
    ) -> bool:
        """
        Sparse land-crossing check: sample intermediate great-circle points along the leg and
        reject if any sampled point is on land per point_validity().

        This mirrors the router's historical behavior (fast, can miss thin land crossings).
        """
        if distance_nm is None:
            return False
        try:
            distance_nm = float(distance_nm)
            heading_deg = float(heading_deg)
            lat0 = float(lat0)
            lon0 = float(lon0)
        except Exception:
            return False

        if distance_nm <= 0.0:
            return True

        spacing = float(spacing_nm)
        if spacing <= 0.0:
            raise ValueError("spacing_nm must be > 0")
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ValueError("max_samples must be >= 1")

        # Always check at least the midpoint for short legs (distance < spacing),
        # otherwise sample at approximately `spacing` nm intervals.
        if distance_nm <= spacing:
            sample_distances = [distance_nm * 0.5]
        else:
            n_intervals = int(math.ceil(distance_nm / spacing))
            n_interior = max(1, n_intervals - 1)
            if n_interior > max_samples:
                n_intervals = max_samples + 1
            sample_distances = [(distance_nm * i / n_intervals) for i in range(1, n_intervals)]

        for d in sample_distances:
            lat_p, lon_p = gc_destination(lat0, lon0, heading_deg, float(d))
            if not self.point_validity(lat_p, lon_p):
                return False
        return True

    def leg_is_clear_point(
        self,
        lat0: float,
        lon0: float,
        heading_deg: float,
        distance_nm: float,
    ) -> bool:
        """
        Point-wise land-crossing check: validate only the start and end points of the leg.

        This does NOT attempt to detect land crossings between the endpoints.
        """
        if distance_nm is None:
            return False
        try:
            distance_nm = float(distance_nm)
            heading_deg = float(heading_deg)
            lat0 = float(lat0)
            lon0 = float(lon0)
        except Exception:
            return False

        if distance_nm <= 0.0:
            return True

        # Start and end point validity only.
        if not self.point_validity(lat0, lon0):
            return False

        lat_p, lon_p = gc_destination(lat0, lon0, heading_deg, float(distance_nm))
        return self.point_validity(float(lat_p), float(lon_p))

    def leg_is_clear(
        self,
        lat0: float,
        lon0: float,
        heading_deg: float,
        distance_nm: float,
        mode,
        spacing_nm: float = 1.0,
        max_samples: int = 25,
        land_threshold: float = 0.5,
    ) -> bool:
        """
        Unified leg-clear API.

        mode:
          - 'strict': grid-cell intersection check (uses land_threshold)
          - 'step': intermediate-point sampling (uses point_validity thresholding)
          - 'sparse': same as 'step' (kept for backward compatibility)
          - 'point': start+end point validity only
        """
        if isinstance(mode, str) and mode.lower() == 'point':
            return self.leg_is_clear_point(lat0, lon0, heading_deg, distance_nm)
        if isinstance(mode, str) and mode.lower() == 'strict':
            return self.leg_is_clear_strict(
                lat0, lon0, heading_deg, distance_nm, land_threshold=land_threshold
            )
        if isinstance(mode, str) and mode.lower() in ('step', 'sparse'):
            return self.leg_is_clear_sparse(
                lat0, lon0, heading_deg, distance_nm, spacing_nm=spacing_nm, max_samples=max_samples
            )
        raise ValueError("mode must be one of: 'point', 'step', 'strict'")

    def point_validity(self, lat, lon):
        if self.method == 'arr':
            try:
                x = self.lats.index(round(lat*4)/4)
                y = self.lons.index(round(lon*4)/4)
                res = self.lsm_arr[x,y] <= 0.1
            except:
                res = False
            return res
        if self.method == 'nearest':
            # method = nearest is problematic at low resolutions
            try:
                lat_f = float(lat)
                lon_f = float(lon)
            except Exception:
                return False
            lon_wrapped = wrap_lon_to_domain(lon_f, self._lon_min, self._lon_max)
            yi = int(nearest_index(self._lat_vals, lat_f, self._lat_asc))
            xi = int(nearest_index(self._lon_vals, lon_wrapped, self._lon_asc))
            return float(self.lsm_arr[yi, xi]) <= 0.1
