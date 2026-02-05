from numba import njit


@njit(cache=True)
def nearest_index(vals, x, asc):
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
def wrap_lon_to_domain(lon, lon_min_v, lon_max_v):
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


@njit(cache=True, fastmath=True)
def polar_speed(tws, twa, speed_table, tws_max, twa_step, twa_max):
    """
    Fast polar lookup mirroring Polar.getSpeed rounding/clamping.
    - tws rounded to nearest int and clamped to tws_max
    - twa rounded to nearest twa_step and clamped to [0, twa_max]
    """
    step = float(twa_step)
    rtwa = int(round(float(twa) / step)) * step
    if rtwa < 0.0:
        rtwa = 0.0
    if rtwa > twa_max:
        rtwa = twa_max
    twa_idx = int(round(rtwa / step))

    rtws = int(round(float(tws)))
    if rtws > int(tws_max):
        rtws = int(tws_max)
    if rtws < 0:
        rtws = 0
    tws_idx = rtws

    return speed_table[twa_idx][tws_idx]
