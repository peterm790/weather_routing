"""passing in point validity is marginally faster not sure why
point validity is the slowest portion of the code"""



import xarray as xr
import os


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
        
        self.lsm = lsm.load()
        self.lsm_arr = self.lsm.values
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
            mask = self.lsm.sel(latitude = lat,longitude = lon, method = 'nearest').values <= 0.1
            return mask.compute()



