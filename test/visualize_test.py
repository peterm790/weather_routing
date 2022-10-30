import xarray as xr
import zarr
import numpy as np
import sys
sys.path.insert(0, '../')

from visualize import visualize
import pandas as pd


ds = xr.open_zarr('/home/peter/Documents/weather_routing/test/test_ds.zarr')

isochrones = []
start,finish = (-34,17),(-24,-45)

vis_class = visualize(ds,start,finish,isochrones)

vis_class.make_plot()