import fsspec
import xarray as xr
import zarr
import numpy as np

land = xr.open_dataset('era5_land-sea-mask.nc')
land.coords['longitude'] = (land.coords['longitude'] + 180) % 360 - 180
land = land.sortby(land.longitude)
lsm = land.lsm[0]


dic = lsm.to_dict()

print(dic)
