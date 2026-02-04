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


def _load_weather() -> xr.Dataset:
    ds = xr.open_zarr(
        "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
        decode_timedelta=True,
    )
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds[["wind_u_10m", "wind_v_10m"]]

    ds = ds.sel(lat=slice(40, 35)).sel(lon=slice(-7, 4))
    ds = ds.isel(init_time=-1)
    ds = ds.isel(lead_time=slice(0, 120))

    ds = ds.assign_coords(time=ds.init_time + ds.lead_time)
    ds = ds.swap_dims({"lead_time": "time"})

    u10 = ds.wind_u_10m
    v10 = ds.wind_v_10m
    tws = np.sqrt(v10**2 + u10**2)
    tws = tws * 1.94384  # convert m/s to knots
    twd = np.mod(180 + np.rad2deg(np.arctan2(u10, v10)), 360)

    ds = tws.to_dataset(name="tws")
    ds["twd"] = twd
    return ds.load()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    try:
        head_sha = _ensure_latest_main(repo_root)
    except RuntimeError as exc:
        print(f"Preflight failed: {exc}", file=sys.stderr)
        return 2

    from weather_router import isochronal_weather_router, polar, point_validity

    start = time.perf_counter()
    ds = _load_weather()

    def get_wind(t, lat, lon):
        tws_sel = ds.tws.sel(time=t, method="nearest")
        tws_sel = tws_sel.sel(lat=lat, lon=lon, method="nearest")
        twd_sel = ds.twd.sel(time=t, method="nearest")
        twd_sel = twd_sel.sel(lat=lat, lon=lon, method="nearest")
        return (np.float32(twd_sel.values), np.float32(tws_sel.values))

    point_valid = point_validity.land_sea_mask(
        extent=[40, -7, 35, 4]
    ).point_validity_arr

    Palma = (39.283014, 2.527704)
    Gibraltar = (36.073, -5.354)

    weatherrouter = isochronal_weather_router.weather_router(
        polar.Polar(str(repo_root / "test/volvo70.pol")),
        get_wind,
        time_steps=ds.time.values,
        step=1,
        start_point=Palma,
        end_point=Gibraltar,
        point_validity=point_valid,
        spread=140,
        wake_lim=35,
        n_points=50,
        rounding=3,
    )

    weatherrouter.route()
    route_df = weatherrouter.get_fastest_route(stats=True)
    elapsed = time.perf_counter() - start

    init_time = ds.time.values[0]
    report_lines = [
        "# Weather Routing Demo Timing",
        "",
        f"- Timestamp (UTC): {datetime.now(timezone.utc).isoformat()}",
        f"- Repo SHA: {head_sha}",
        "- Data source: https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
        f"- Time steps: {len(ds.time.values)}",
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
