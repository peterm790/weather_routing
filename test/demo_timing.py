"""
Run the README demo using dynamical GFS data, time it, and write a markdown report.

Usage:
  uv run python test/demo_timing.py
"""

from __future__ import annotations

import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import io
import urllib.request

import numpy as np
import xarray as xr


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {msg}")
    return result.stdout.strip()


def _ensure_latest_main(repo_root: Path) -> str:
    toplevel = _run_git(repo_root, "rev-parse", "--show-toplevel")
    if Path(toplevel).resolve() != repo_root.resolve():
        raise RuntimeError(f"Expected repo root {repo_root}, got {toplevel}")

    _run_git(repo_root, "fetch", "origin", "main")

    branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
    if branch != "main":
        raise RuntimeError(f"Must be on branch 'main' to run demo, found '{branch}'")

    head = _run_git(repo_root, "rev-parse", "HEAD")
    origin_main = _run_git(repo_root, "rev-parse", "origin/main")
    if head != origin_main:
        raise RuntimeError(
            "Repo is not at latest origin/main. "
            f"HEAD={head}, origin/main={origin_main}"
        )

    return head


def _load_weather(freq: str, crank_step: int, lead_time_start: int) -> xr.Dataset:
    ds = xr.open_zarr(
        "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
        decode_timedelta=True,
    )
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds[["wind_u_10m", "wind_v_10m"]]

    ds = ds.sel(lat=slice(40, 35)).sel(lon=slice(-7, 4))
    if "init_time" in ds.dims:
        ds = ds.isel(init_time=-1)

    if "lead_time" in ds.dims:
        if freq == "1hr":
            ds = ds.isel(lead_time=slice(lead_time_start, 120))
        elif freq == "3hr":
            hourly_indices = list(range(0, 121, 3))
            three_hourly_indices = list(range(121, ds.sizes["lead_time"]))
            indices = hourly_indices + three_hourly_indices
            ds = ds.isel(lead_time=indices)
            start_index = int(round(lead_time_start / 3))
            ds = ds.isel(lead_time=slice(start_index, None))
        else:
            raise ValueError("freq must be '1hr' or '3hr'")
    elif "time" in ds.dims:
        ds = ds.isel(time=slice(lead_time_start, lead_time_start + 120))

    if "lead_time" in ds.dims and "init_time" in ds.coords:
        ds = ds.assign_coords(time=ds.init_time + ds.lead_time)
        ds = ds.swap_dims({"lead_time": "time"})

    ds = ds.load()

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

    if "time" not in ds.dims or ds.sizes.get("time", 0) < 2:
        raise ValueError("Weather dataset has insufficient time steps to interpolate")

    time_vals = ds.time.values
    dt = np.timedelta64(int(crank_step), "m")
    new_times = np.arange(time_vals[0], time_vals[-1], dt)
    if new_times.size == 0 or new_times[-1] != time_vals[-1]:
        new_times = np.concatenate([new_times, np.array([time_vals[-1]], dtype=time_vals.dtype)])

    ds_interp = ds[["wind_u_10m", "wind_v_10m"]].interp(time=new_times)

    u10 = ds_interp["wind_u_10m"]
    v10 = ds_interp["wind_v_10m"]
    tws = np.hypot(v10, u10)
    tws = tws * 1.94384  # convert m/s to knots
    twd = (180 + np.rad2deg(np.arctan2(u10, v10))) % 360

    ds_processed = tws.to_dataset(name="tws")
    ds_processed["twd"] = twd
    return ds_processed


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
    try:
        head_sha = _ensure_latest_main(repo_root)
    except RuntimeError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2

    from weather_router import isochronal_weather_router, polar

    freq = "1hr"
    crank_step = 30
    lead_time_start = 0
    min_lat, min_lon, max_lat, max_lon = 35, -7, 40, 4
    polar_file = "volvo70"

    start = time.perf_counter()
    ds_processed = _load_weather(freq=freq, crank_step=crank_step, lead_time_start=lead_time_start)

    def get_wind(t, lat, lon):
        tws_sel = ds_processed.tws.sel(time=t, method="nearest")
        tws_sel = tws_sel.sel(lat=lat, lon=lon, method="nearest")
        twd_sel = ds_processed.twd.sel(time=t, method="nearest")
        twd_sel = twd_sel.sel(lat=lat, lon=lon, method="nearest")
        return (np.float32(twd_sel.values), np.float32(tws_sel.values))

    print("fetching polar")
    url = f"https://data.offshoreweatherrouting.com/polars/{polar_file}.pol"
    try:
        with urllib.request.urlopen(url) as response:
            polar_data = response.read().decode("utf-8")
    except Exception:
        polar_path = repo_root / "test" / f"{polar_file}.pol"
        if not polar_path.exists():
            raise
        polar_data = polar_path.read_text()

    polar_obj = polar.Polar(f=io.StringIO(polar_data))

    print("loading lsm zarr")
    ds_lsm = xr.open_zarr(
        "https://data.offshoreweatherrouting.com/GEBCO_2025_land_mask_sharded.zarr",
        consolidated=True,
    )
    ds_lsm = ds_lsm.rename({"lat": "latitude", "lon": "longitude"})
    ds_lsm = ds_lsm.sortby("latitude", ascending=False)
    ds_lsm = ds_lsm.fillna(0)
    ds_lsm = ds_lsm.sel(latitude=slice(max_lat, min_lat)).sel(longitude=slice(min_lon, max_lon))
    ds_lsm = ds_lsm.load()

    Palma = (39.283014, 2.527704)
    Gibraltar = (36.073, -5.354)

    weatherrouter = isochronal_weather_router.weather_router(
        polar_obj,
        get_wind,
        time_steps=ds_processed.time.values,
        step=float(crank_step) / 60.0,
        start_point=Palma,
        end_point=Gibraltar,
        point_validity_extent=[min_lat, min_lon, max_lat, max_lon],
        point_validity_file=ds_lsm,
        avoid_land_crossings="step",
        leg_check_spacing_nm=2.0,
        spread=140,
        wake_lim=35,
        rounding=3,
        n_points=50,
        optimise_n_points=60,
        finish_size=5.0,
        tack_penalty=0.5,
        optimise_window=24,
        leg_check_max_samples=10,
        point_validity_method="nearest",
        twa_change_penalty=0.02,
        twa_change_threshold=5.0,
    )

    weatherrouter.route()
    route_df = weatherrouter.get_fastest_route(stats=True)
    elapsed = time.perf_counter() - start

    init_time = ds_processed.time.values[0]
    report_lines = [
        "# Weather Routing Demo Timing",
        "",
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Repo SHA: {head_sha}",
        "- Data source: https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
        f"- Time steps: {len(ds_processed.time.values)}",
        f"- Forecast start: {init_time}",
        f"- Route points: {route_df.shape[0]}",
        f"- Total runtime seconds: {elapsed:.2f}",
        "",
    ]

    report_path = repo_root / "demo_timing.md"
    report_path.write_text("\n".join(report_lines))

    print("\n".join(report_lines))
    print(f"Wrote report to {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
