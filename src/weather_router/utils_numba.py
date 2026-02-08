from numba import njit


@njit(cache=True)
def nearest_index(vals, x, asc):
    """
    Return nearest index in a sorted 1D array.
    Equivalent to `xarray.sel(..., method="nearest")` on that coordinate.
    """
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
    """
    Wrap longitude into the data domain [lon_min_v, lon_max_v].
    Equivalent to modulo wrap used before nearest-neighbor selection.
    """
    if lon_max_v <= lon_min_v:
        return lon

    # Prefer the representation that's already inside the domain, or that requires
    # the smallest +/-360 adjustment to get inside. This avoids pathological
    # behavior when the domain has been padded beyond 360 degrees.
    best = lon
    best_delta = 1e30
    found = False
    cand0 = lon
    cand1 = lon + 360.0
    cand2 = lon - 360.0

    if lon_min_v <= cand0 <= lon_max_v:
        best = cand0
        best_delta = 0.0
        found = True
    if lon_min_v <= cand1 <= lon_max_v:
        d = abs(cand1 - lon)
        if d < best_delta:
            best = cand1
            best_delta = d
            found = True
    if lon_min_v <= cand2 <= lon_max_v:
        d = abs(cand2 - lon)
        if d < best_delta:
            best = cand2
            found = True

    if found:
        return best

    # Fallback: clamp (treat out-of-domain as edge).
    if lon < lon_min_v:
        return lon_min_v
    if lon > lon_max_v:
        return lon_max_v
    return lon


@njit(cache=True, fastmath=True)
def polar_speed(tws, twa, speed_table, tws_max, tws_min, twa_step, twa_max, twa_min):
    """
    Fast polar lookup mirroring Polar.getSpeed rounding/clamping.
    - tws rounded to nearest int and clamped to [tws_min, tws_max]
    - twa rounded to nearest twa_step and clamped to [twa_min, twa_max]
    Equivalent to Polar.getSpeed(tws, abs(twa)) but Numba-friendly.
    """
    step = float(twa_step)
    rtwa = int(round(float(twa) / step)) * step
    if rtwa < twa_min:
        rtwa = twa_min
    if rtwa > twa_max:
        rtwa = twa_max
    # Calculate index: (rtwa - twa_min) / step
    twa_idx = int(round((rtwa - twa_min) / step))

    rtws = int(round(float(tws)))
    if rtws > int(tws_max):
        rtws = int(tws_max)
    if rtws < int(tws_min):
        rtws = int(tws_min)
    # Calculate index: rtws - tws_min
    tws_idx = rtws - int(tws_min)

    return speed_table[twa_idx][tws_idx]
