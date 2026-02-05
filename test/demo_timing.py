"""
Run the README demo using dynamical GFS data, time it, and write a markdown report.

Usage:
  uv run python test/demo_timing.py
"""

from __future__ import annotations

import csv
import io
import os
import platform
import subprocess
import sys
import time
import urllib.request
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


def _git_info(repo_root: Path) -> tuple[str, str]:
    try:
        toplevel = _run_git(repo_root, "rev-parse", "--show-toplevel")
        if Path(toplevel).resolve() != repo_root.resolve():
            return ("unknown", "unknown")
        sha = _run_git(repo_root, "rev-parse", "HEAD")
        branch = _run_git(repo_root, "rev-parse", "--abbrev-ref", "HEAD")
        return (sha, branch)
    except Exception:
        return ("unknown", "unknown")


def _system_info() -> dict[str, str]:
    info: dict[str, str] = {}
    info["os"] = platform.platform()
    info["python"] = sys.version.split()[0]
    info["machine"] = platform.machine()
    info["cpu_cores"] = str(os.cpu_count() or "unknown")

    cpu_model = "unknown"
    mem_gb = "unknown"
    if sys.platform == "darwin":
        try:
            cpu_model = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
            mem_bytes = int(subprocess.check_output(["sysctl", "-n", "hw.memsize"]).decode().strip())
            mem_gb = f"{mem_bytes / (1024**3):.1f}"
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        cpu_model = line.split(":", 1)[1].strip()
                        break
            with open("/proc/meminfo", "r", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        mem_kb = int(line.split()[1])
                        mem_gb = f"{mem_kb / (1024**2):.1f}"
                        break
        except Exception:
            pass
    info["cpu_model"] = cpu_model
    info["mem_gb"] = mem_gb
    return info


def _load_weather(
    cache_path: Path,
    init_time_index: int,
    lead_time_hours: int,
    min_lat: float,
    min_lon: float,
    max_lat: float,
    max_lon: float,
    freq: str,
    crank_step: int,
    lead_time_start: int,
) -> xr.Dataset:
    if cache_path.exists():
        return xr.open_zarr(str(cache_path))

    ds = xr.open_zarr(
        "https://data.dynamical.org/noaa/gfs/forecast/latest.zarr",
        decode_timedelta=True,
    )
    ds = ds.rename({"latitude": "lat", "longitude": "lon"})
    ds = ds[["wind_u_10m", "wind_v_10m"]]

    ds = ds.sel(lat=slice(max_lat, min_lat)).sel(lon=slice(min_lon, max_lon))
    if "init_time" in ds.dims:
        ds = ds.isel(init_time=init_time_index)

    if "lead_time" in ds.dims:
        if freq == "1hr":
            ds = ds.isel(lead_time=slice(lead_time_start, lead_time_start + lead_time_hours + 1))
        elif freq == "3hr":
            hourly_indices = list(range(0, lead_time_hours + 1, 3))
            three_hourly_indices = list(range(121, ds.sizes["lead_time"]))
            indices = hourly_indices + three_hourly_indices
            ds = ds.isel(lead_time=indices)
            start_index = int(round(lead_time_start / 3))
            ds = ds.isel(lead_time=slice(start_index, None))
        else:
            raise ValueError("freq must be '1hr' or '3hr'")
    elif "time" in ds.dims:
        ds = ds.isel(time=slice(lead_time_start, lead_time_start + lead_time_hours + 1))

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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    ds_processed.to_zarr(str(cache_path), mode="w")
    return xr.open_zarr(str(cache_path))


def _haversine_nm(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_earth_nm = 3440.065
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return float(r_earth_nm * c)


def _write_markdown_table(path: Path, headers: list[str], values: list[str]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        "| " + " | ".join(values) + " |",
        "",
    ]
    path.write_text("\n".join(lines))


def _append_csv(path: Path, headers: list[str], row: list[str]) -> None:
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(headers)
        writer.writerow(row)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if src_dir.exists():
        sys.path.insert(0, str(src_dir))
    head_sha, branch = _git_info(repo_root)
    sys_info = _system_info()

    from weather_router import isochronal_weather_router, polar

    freq = "1hr"
    crank_step = 60
    lead_time_start = 0
    min_lat, min_lon, max_lat, max_lon = 35, -7, 40, 4
    polar_file = "volvo70"
    init_time_index = 6964
    lead_time_hours = 48
    cache_path = repo_root / "cache" / "version_benchmark_gfs_2024-07-10T20Z.zarr"

    start = time.perf_counter()
    ds_processed = _load_weather(
        cache_path=cache_path,
        init_time_index=init_time_index,
        lead_time_hours=lead_time_hours,
        min_lat=min_lat,
        min_lon=min_lon,
        max_lat=max_lat,
        max_lon=max_lon,
        freq=freq,
        crank_step=crank_step,
        lead_time_start=lead_time_start,
    )

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

    Gibraltar = (36.073, -5.354)
    Palma = (39.283014, 2.527704)

    weatherrouter = isochronal_weather_router.weather_router(
        polar_obj,
        get_wind,
        time_steps=ds_processed.time.values,
        step=float(crank_step) / 60.0,
        start_point=Gibraltar,
        end_point=Palma,
        point_validity_extent=[min_lat, min_lon, max_lat, max_lon],
        point_validity_file=ds_lsm,
        avoid_land_crossings="strict",
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

    algo_start = time.perf_counter()
    weatherrouter.route()
    algo_elapsed = time.perf_counter() - algo_start
    route_df = weatherrouter.get_fastest_route(stats=True)
    elapsed = time.perf_counter() - start

    route_points = route_df.shape[0]
    if "time" in route_df.columns:
        eta = route_df.iloc[-1].time
    else:
        eta = "unknown"
    if "hours_elapsed" in route_df.columns:
        duration_hours = float(route_df.iloc[-1].hours_elapsed)
    else:
        duration_hours = float(len(ds_processed.time.values) - 1)

    if route_points > 0 and "lat" in route_df.columns and "lon" in route_df.columns:
        last_lat = float(route_df.iloc[-1].lat)
        last_lon = float(route_df.iloc[-1].lon)
        remaining_nm = _haversine_nm(last_lat, last_lon, Palma[0], Palma[1])
    else:
        remaining_nm = float("nan")

    init_time = ds_processed.time.values[0]
    timestamp_utc = datetime.now(timezone.utc).isoformat()
    report_path = repo_root / "version_benchmark.md"
    csv_path = repo_root / "version_benchmark.csv"
    map_path = repo_root / "version_benchmark.png"

    headers = [
        "timestamp_utc",
        "git_sha",
        "cpu_model",
        "algo_runtime_seconds",
    ]
    values = [
        timestamp_utc,
        head_sha,
        sys_info["cpu_model"],
        f"{algo_elapsed:.2f}",
    ]

    _write_markdown_table(report_path, headers, values)

    csv_headers = [
        "timestamp_utc",
        "git_sha",
        "git_branch",
        "cpu_model",
        "cpu_cores",
        "mem_gb",
        "os",
        "python",
        "data_cache",
        "forecast_init_time",
        "lead_time_hours",
        "route_points",
        "route_duration_hours",
        "eta_utc",
        "remaining_nm",
        "algo_runtime_seconds",
        "total_runtime_seconds",
    ]
    csv_values = [
        timestamp_utc,
        head_sha,
        branch,
        sys_info["cpu_model"],
        sys_info["cpu_cores"],
        sys_info["mem_gb"],
        sys_info["os"],
        sys_info["python"],
        str(cache_path),
        str(init_time),
        str(lead_time_hours),
        str(route_points),
        f"{duration_hours:.2f}",
        str(eta),
        f"{remaining_nm:.2f}",
        f"{algo_elapsed:.2f}",
        f"{elapsed:.2f}",
    ]
    _append_csv(csv_path, csv_headers, csv_values)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(7, 6))
        ax = plt.axes(projection=ccrs.Mercator())
        ax.set_extent([min_lon, max_lon, min_lat, max_lat], crs=ccrs.PlateCarree())

        ax.add_feature(cfeature.LAND, facecolor="#f2efe9")
        ax.add_feature(cfeature.OCEAN, facecolor="#d7e7f3")
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4, linestyle=":")

        ax.plot(
            route_df["lon"],
            route_df["lat"],
            transform=ccrs.PlateCarree(),
            color="black",
            linewidth=2,
        )
        ax.scatter(
            [Gibraltar[1], Palma[1]],
            [Gibraltar[0], Palma[0]],
            transform=ccrs.PlateCarree(),
            c=["red", "green"],
            s=30,
            zorder=5,
        )

        ax.set_title("Weather Routing Benchmark")
        plt.tight_layout()
        plt.savefig(map_path, dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Map render failed: {exc}", file=sys.stderr)

    print(f"Wrote report to {report_path}")
    print(f"Appended CSV to {csv_path}")
    print(f"Wrote route map to {map_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
