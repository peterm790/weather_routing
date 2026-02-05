import math
import numpy as np
from numba import njit

_R_EARTH_NM = 3440.065


@njit(cache=True, fastmath=True)
def wrap_lon180(lon_deg: float) -> float:
    return (float(lon_deg) + 180.0) % 360.0 - 180.0


@njit(cache=True, fastmath=True)
def bearing_deg(lat1_deg: float, lon1_deg: float, lat2_deg: float, lon2_deg: float) -> float:
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    diff_long = math.radians(lon2_deg - lon1_deg)
    x = math.sin(diff_long) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_long))
    initial_bearing = math.atan2(x, y)
    brg = (math.degrees(initial_bearing) + 360.0) % 360.0
    return brg


@njit(cache=True, fastmath=True)
def gc_destination(lat_deg: float, lon_deg: float, bearing_deg_val: float, distance_nm: float):
    """
    Great-circle destination on a sphere, returning (lat, lon) in degrees.
    """
    if distance_nm == 0.0:
        return float(lat_deg), wrap_lon180(float(lon_deg))
    lat1 = math.radians(float(lat_deg))
    lon1 = math.radians(float(lon_deg))
    theta = math.radians((float(bearing_deg_val) % 360.0))
    delta = float(distance_nm) / _R_EARTH_NM

    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_delta = math.sin(delta)
    cos_delta = math.cos(delta)
    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)

    sin_lat2 = sin_lat1 * cos_delta + cos_lat1 * sin_delta * cos_theta
    sin_lat2 = max(-1.0, min(1.0, sin_lat2))
    lat2 = math.asin(sin_lat2)

    y = sin_theta * sin_delta * cos_lat1
    x = cos_delta - sin_lat1 * math.sin(lat2)
    lon2 = lon1 + math.atan2(y, x)

    return math.degrees(lat2), wrap_lon180(math.degrees(lon2))


@njit(cache=True, fastmath=True)
def haversine_nm_vec(lat1_arr, lon1_arr, lat2, lon2):
    """
    Vectorized haversine distance (nm) from arrays lat1/lon1 to a single lat2/lon2.
    Accepts numpy arrays or array-likes for lat1_arr/lon1_arr.
    Equivalent to geopy.distance.great_circle(...).nm (spherical Earth).
    """
    lat1 = np.radians(np.asarray(lat1_arr, dtype=np.float64))
    lon1 = np.radians(np.asarray(lon1_arr, dtype=np.float64))
    lat2r = math.radians(float(lat2))
    lon2r = math.radians(float(lon2))
    dlat = lat2r - lat1
    dlon = lon2r - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * math.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return _R_EARTH_NM * c


@njit(cache=True, fastmath=True)
def haversine_nm_scalar(lat1, lon1, lat2, lon2) -> float:
    """
    Scalar haversine distance (nm).
    Equivalent to geopy.distance.great_circle((lat1, lon1), (lat2, lon2)).nm (spherical Earth).
    """
    lat1r = math.radians(float(lat1))
    lon1r = math.radians(float(lon1))
    lat2r = math.radians(float(lat2))
    lon2r = math.radians(float(lon2))
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1r) * math.cos(lat2r) * math.sin(dlon / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return _R_EARTH_NM * c
