"""passing in point validity is marginally faster not sure why
point validity is the slowest portion of the code"""



import xarray as xr
import os


class land_sea_mask():
    def __init__ (self, extent=None):
        """
            :param extent [lat1,lon1,lat2,lon2]
        """
        lsm = xr.open_dataset(os.path.join(os.path.dirname(__file__), 'data/era5_land-sea-mask.nc'))
        lsm.coords['longitude'] = (lsm.coords['longitude'] + 180) % 360 - 180
        lsm = lsm.sortby(lsm.longitude)
        lsm = lsm.lsm[0].load()
        if extent:
            lat1,lon1,lat2,lon2 = extent
            lsm = lsm.sel(latitude = slice(max([lat1, lat2]),min([lat1, lat2]))).sel(longitude = slice(min([lon1, lon2]),max([lon1, lon2])))
        self.lsm = lsm
        self.lsm_arr = self.lsm.values
        self.lats = list(lsm.latitude.values)
        self.lons = list(lsm.longitude.values)

    def point_validity_arr(self, lat, lon):
        try:
            x = self.lats.index(round(lat*4)/4)
            y = self.lons.index(round(lon*4)/4)
            res = self.lsm_arr[x,y] <= 0.1
        except:
            res = False
        return res

    def point_validity(self, lat, lon):
        return self.lsm.sel(latitude = lat,longitude = lon, method = 'nearest').values <= 0.1



