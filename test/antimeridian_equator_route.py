"""
Run a real routing example that crosses the antimeridian and equator using GFS data.

Outputs:
  - antimeridian_equator_route.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib
matplotlib.use("Agg")  # headless-safe PNG output
import matplotlib.pyplot as plt


def _select_lon_crossing(ds: xr.Dataset, lon_name: str, min_lon: float, max_lon: float) -> xr.Dataset:
    lon_vals = ds[lon_name].values
    if min_lon <= max_lon:
        mask = (lon_vals >= min_lon) & (lon_vals <= max_lon)
    else:
        mask = (lon_vals >= min_lon) | (lon_vals <= max_lon)
    idx = np.where(mask)[0]
    return ds.isel({lon_name: idx})


def _load_weather(cache_path: Path, init_time_index: int, lead_hours: int) -> xr.Dataset:
    if cache_path.exists():
        ds_cached = xr.open_zarr(str(cache_path))
        if ds_cached.sizes.get("time", 0) >= lead_hours + 1:
            return ds_cached

    last_err = None
    for _ in range(3):
        try:
            ds = xr.open_zarr(
                "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
                decode_timedelta=True,
            )
            ds = ds.rename({"latitude": "lat", "longitude": "lon"})
            ds = ds[["wind_u_10m", "wind_v_10m"]]

            # Wrap to [-180, 180) to make antimeridian handling explicit.
            ds["lon"] = ((ds["lon"] + 180) % 360) - 180
            ds = ds.sortby("lon")

            # Target bbox: lat [-10, 10], lon [170, -170] (crosses antimeridian)
            min_lat, max_lat = -10, 10
            min_lon, max_lon = 170, -170
            ds = ds.sel(lat=slice(max_lat, min_lat))
            ds = _select_lon_crossing(ds, "lon", min_lon, max_lon)

            ds = ds.isel(init_time=init_time_index)
            ds = ds.isel(lead_time=slice(0, lead_hours + 1))

            ds = ds.assign_coords(time=ds.init_time + ds.lead_time)
            ds = ds.swap_dims({"lead_time": "time"})
            ds = ds.drop_vars(
                [
                    "expected_forecast_length",
                    "ingested_forecast_length",
                    "init_time",
                    "lead_time",
                    "spatial_ref",
                    "valid_time",
                ],
                errors="ignore",
            )

            ds = ds.load()

            u10 = ds["wind_u_10m"]
            v10 = ds["wind_v_10m"]
            tws = np.hypot(v10, u10) * 1.94384  # m/s -> knots
            twd = (180 + np.rad2deg(np.arctan2(u10, v10))) % 360
            ds_processed = tws.to_dataset(name="tws")
            ds_processed["twd"] = twd

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            ds_processed.to_zarr(str(cache_path), mode="w")
            return xr.open_zarr(str(cache_path))
        except Exception as exc:
            last_err = exc
    raise last_err


def _fallback_cache(cache_dir: Path, lead_hours: int) -> xr.Dataset | None:
    candidates = sorted(cache_dir.glob("antimeridian_equator_gfs_2024-07-10T20Z*.zarr"))
    for path in candidates:
        try:
            ds_cached = xr.open_zarr(str(path))
            if ds_cached.sizes.get("time", 0) >= 2:
                if ds_cached.sizes.get("time", 0) < lead_hours + 1:
                    print(
                        f"Warning: using shorter cached weather window from {path.name} "
                        f"({ds_cached.sizes.get('time', 0)} hours)."
                    )
                return ds_cached
        except Exception:
            continue
    return None


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        import sys
        sys.path.insert(0, str(src_dir))

    from weather_router import isochronal_weather_router, polar

    lead_hours = 120
    cache_path = repo_root / "cache" / f"antimeridian_equator_gfs_2024-07-10T20Z_{lead_hours}h.zarr"
    try:
        ds = _load_weather(cache_path=cache_path, init_time_index=6964, lead_hours=lead_hours)
    except Exception as exc:
        fallback = _fallback_cache(cache_path.parent, lead_hours=lead_hours)
        if fallback is None:
            raise exc
        ds = fallback

    def get_wind(t, lat, lon):
        tws_sel = ds.tws.sel(time=t, method="nearest")
        tws_sel = tws_sel.sel(lat=lat, lon=lon, method="nearest")
        twd_sel = ds.twd.sel(time=t, method="nearest")
        twd_sel = twd_sel.sel(lat=lat, lon=lon, method="nearest")
        return (np.float32(twd_sel.values), np.float32(tws_sel.values))

    # Land mask (real, local). We use the ERA5 land-sea mask shipped with the repo
    # because it's lightweight and avoids network + huge GEBCO reads for this test.
    ds_lsm = xr.open_dataset(repo_root / "src" / "weather_router" / "data" / "era5_land-sea-mask.nc")
    ds_lsm["longitude"] = ((ds_lsm["longitude"] + 180) % 360) - 180
    ds_lsm = ds_lsm.sortby("longitude")
    ds_lsm = ds_lsm.fillna(0)
    ds_lsm = ds_lsm.sel(latitude=slice(10, -10))
    ds_lsm = _select_lon_crossing(ds_lsm, "longitude", 170, -170)
    ds_lsm = ds_lsm.load()

    # Start/end across antimeridian and equator
    # Route: 1°S just WEST of the antimeridian to 1°N just EAST of the antimeridian.
    #
    # With longitude wrapped to [-180, 180), "west of 180" is ~ +179, and "east of 180"
    # is ~ -179. (Remember: 181 == -179 in wrapped coordinates.)
    start = (-1.0, 179.0)
    end = (1.0, -179.0)

    weatherrouter = isochronal_weather_router.weather_router(
        polar.Polar(str(repo_root / "test/volvo70.pol")),
        get_wind,
        time_steps=ds.time.values,
        step=1,
        start_point=start,
        end_point=end,
        point_validity_extent=None,
        point_validity_file=ds_lsm,
        avoid_land_crossings="strict",
        leg_check_spacing_nm=2.0,
        spread=140,
        wake_lim=35,
        n_points=50,
        rounding=3,
        finish_size=5.0,
        optimise_n_points=60,
        optimise_window=24,
        leg_check_max_samples=10,
        point_validity_method="nearest",
        twa_change_penalty=0.02,
        twa_change_threshold=5.0,
    )

    weatherrouter.route()
    route_df = weatherrouter.get_fastest_route(stats=True)
    isochrones = weatherrouter.get_isochrones_latlon()

    fig = plt.figure(figsize=(7, 4))
    # Center the projection on the antimeridian (dateline).
    proj = ccrs.PlateCarree(central_longitude=180)
    ax = plt.axes(projection=proj)
    # For dateline-crossing extents, Cartopy behaves best when the "east" value
    # is > 180 (e.g. 190 == -170), rather than passing west>east (170, -170).
    # This gives a true +/-10° window around the antimeridian.
    ax.set_extent([170, 190, -10, 10], crs=ccrs.PlateCarree())

    ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
    ax.add_feature(cfeature.OCEAN, facecolor="#d7e7f3")
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6)

    # Visual reference lines inside the window (not the full globe).
    ax.plot([180, 180], [-10, 10], transform=ccrs.PlateCarree(), color="black", alpha=0.25, linewidth=1.0, linestyle="--")
    ax.plot([170, 190], [0, 0], transform=ccrs.PlateCarree(), color="black", alpha=0.25, linewidth=1.0, linestyle="--")

    def _lon_0_360(lon):
        # Keep longitude continuous around the antimeridian for plotting.
        # Example: -179 -> 181 so it sits next to +179 in a 170..190° window.
        return (float(lon) + 360.0) % 360.0

    # Plot isochrones to visualize wavefront even if route doesn't finish.
    for iso in isochrones:
        if iso is None or len(iso) == 0:
            continue
        ax.plot(
            [_lon_0_360(x) for x in iso[:, 1]],
            iso[:, 0],
            transform=ccrs.PlateCarree(),
            color="#444444",
            linewidth=0.6,
            alpha=0.6,
        )

    # Plot final route (if any)
    ax.plot(
        [_lon_0_360(x) for x in route_df["lon"]],
        route_df["lat"],
        transform=ccrs.PlateCarree(),
        color="black",
        linewidth=2,
    )
    ax.scatter(
        [_lon_0_360(start[1]), _lon_0_360(end[1])],
        [start[0], end[0]],
        transform=ccrs.PlateCarree(),
        c=["red", "green"],
        s=30,
        zorder=5,
    )
    ax.set_title("Antimeridian + Equator Routing Test")
    plt.tight_layout()
    out_path = repo_root / "antimeridian_equator_route.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote route plot to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
