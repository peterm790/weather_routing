"""
Lightweight test to ensure we can evaluate legs that cross both the equator
and the antimeridian using the real land-sea mask (no routing run required).
"""

from __future__ import annotations

from pathlib import Path
import sys

import xarray as xr

# Ensure local `src/` is importable without requiring installation.
_TEST_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TEST_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
if not _SRC_DIR.exists():
    raise FileNotFoundError(f"Expected src directory at: {_SRC_DIR}")

sys.path.insert(0, str(_SRC_DIR))

from weather_router.point_validity import land_sea_mask  # noqa: E402
from weather_router.utils_geo import bearing_deg, haversine_nm_scalar  # noqa: E402


def _load_lsm() -> land_sea_mask:
    lsm_path = _REPO_ROOT / "src" / "weather_router" / "data" / "era5_land-sea-mask.nc"
    ds_lsm = xr.open_dataset(lsm_path)
    return land_sea_mask(file=ds_lsm, method="nearest")


def _fixed_ocean_pair():
    """
    Fixed water points (checked against the ERA5 land-sea mask).
    South of the equator, west of the antimeridian -> north of the equator, east of the antimeridian.
    """
    return (-0.5, 179.0), (0.5, -179.0)


def test_leg_crosses_equator_and_antimeridian():
    lsm = _load_lsm()
    start, end = _fixed_ocean_pair()

    assert start[0] < 0.0
    assert end[0] > 0.0
    assert abs(end[1] - start[1]) > 180.0

    assert lsm.point_validity(start[0], start[1]) is True
    assert lsm.point_validity(end[0], end[1]) is True

    heading = bearing_deg(start[0], start[1], end[0], end[1])
    distance_nm = haversine_nm_scalar(start[0], start[1], end[0], end[1])

    # Strict land check should be stable across equivalent longitude representations.
    ok = lsm.leg_is_clear_strict(start[0], start[1], heading, distance_nm)
    end_equiv = (end[0], end[1] + 360.0)
    heading_equiv = bearing_deg(start[0], start[1], end_equiv[0], end_equiv[1])
    distance_equiv = haversine_nm_scalar(start[0], start[1], end_equiv[0], end_equiv[1])
    ok_equiv = lsm.leg_is_clear_strict(start[0], start[1], heading_equiv, distance_equiv)

    assert isinstance(ok, bool)
    assert ok == ok_equiv
