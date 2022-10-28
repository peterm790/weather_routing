import xarray as xr
import os

class land_sea_mask():
    def __init__ (self):
        self.lsm = xr.open_dataset(os.path.join(os.path.dirname(__file__), 'data/era5_land-sea-mask.nc'))
        self.lsm.coords['longitude'] = (self.lsm.coords['longitude'] + 180) % 360 - 180
        self.lsm = self.lsm.sortby(self.lsm.longitude)
        self.lsm = self.lsm.lsm[0]

    def point_validity(self, lat, lon):
        return self.lsm.sel(latitude = lat,longitude = lon, method = 'nearest').values <= 0.1



