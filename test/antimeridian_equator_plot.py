import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# Create a path that crosses both the equator and antimeridian
lons = np.array([170, 175, 179, -179, -175, -170])
lats = np.array([-5, -2, 1, 3, 5, 7])

fig = plt.figure(figsize=(7, 4))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
ax.set_global()

ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
ax.add_feature(cfeature.OCEAN, facecolor="#d7e7f3")
ax.add_feature(cfeature.COASTLINE, linewidth=0.6)

ax.plot(lons, lats, transform=ccrs.PlateCarree(), color="black", linewidth=2)
ax.scatter(lons, lats, transform=ccrs.PlateCarree(), color="red", s=15)

ax.set_title("Antimeridian + Equator Crossing Test")
plt.tight_layout()
plt.savefig("/Users/petermarsh/Documents/petes/weather_routing/antimeridian_equator_test.png", dpi=150)
plt.close(fig)
