from numba import njit


@njit(cache=True)
def nearest_index_numba(vals, x, asc):
    left = 0
    right = vals.size
    if asc:
        while left < right:
            mid = (left + right) // 2
            if vals[mid] < x:
                left = mid + 1
            else:
                right = mid
    else:
        while left < right:
            mid = (left + right) // 2
            if vals[mid] > x:
                left = mid + 1
            else:
                right = mid

    i = left
    if i == 0:
        return 0
    n = vals.size
    if i >= n:
        return n - 1
    dr = abs(vals[i] - x)
    dl = abs(x - vals[i - 1])
    return i if dr <= dl else i - 1


@njit(cache=True)
def wrap_lon_to_domain_numba(lon, lon_min_v, lon_max_v):
    width = lon_max_v - lon_min_v
    if width <= 0.0:
        return lon

    lon_rel = lon - lon_min_v
    wrapped_rel = lon_rel % 360.0
    wrapped = lon_min_v + wrapped_rel
    if wrapped < lon_min_v:
        wrapped = lon_min_v
    if wrapped > lon_max_v:
        wrapped = lon_max_v
    return wrapped


@njit(cache=True)
def wind_indices_nearest_numba(
    time_vals_ns,
    lat_vals_f64,
    lon_vals_f64,
    t_ns,
    lat,
    lon,
    lat_asc,
    lon_asc,
    lon_min,
    lon_max,
):
    ti = nearest_index_numba(time_vals_ns, t_ns, True)
    yi = nearest_index_numba(lat_vals_f64, lat, lat_asc)
    wl = wrap_lon_to_domain_numba(lon, lon_min, lon_max)
    xi = nearest_index_numba(lon_vals_f64, wl, lon_asc)
    return ti, yi, xi
