"""passing in point validity is marginally faster not sure why
point validity is the slowest portion of the code"""



import xarray as xr
import os


LAND_BORDER_PAD_CELLS = 100


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
            return self.lsm.sel(latitude = lat,longitude = lon, method = 'nearest').values <= 0.1



