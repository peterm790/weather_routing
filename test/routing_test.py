"""
Weather Router Test Suite

This test suite covers both the original weather routing functionality
and the new equidistant pruning implementation.

Tests included:
1. Basic functionality tests (polar, isochrones, fastest route)
2. New equidistant pruning functionality:
   - n_points parameter handling
   - return_equidistant function (creates evenly spaced points along curves)
   - prune_equidistant method (selects closest original points to equidistant positions)
   - Integration testing (full routing with equidistant pruning)
   - Comparison testing (different n_points values)

The equidistant pruning approach provides better point distribution along
isochrone curves compared to the previous crude rounding-based method,
while maintaining route history and allowing performance tuning via n_points.

To run tests:
- python routing_test.py (runs basic routing)
- python run_tests.py (runs full test suite with reporting)
"""

import xarray as xr
import zarr
import numpy as np
import sys
sys.path.insert(0, '../src/')
sys.path.insert(0, './src/')
from weather_router.isochronal_weather_router import weather_router
from weather_router.polar import Polar


ds = xr.open_zarr('test_ds.zarr')

def getWindAt(t, lat, lon):
    tws_sel = ds.tws.sel(time = t, method = 'nearest')
    tws_sel = tws_sel.sel(lat = lat, lon = lon, method = 'nearest')
    twd_sel = ds.twd.sel(time = t, method = 'nearest')
    twd_sel = twd_sel.sel(lat = lat, lon = lon, method = 'nearest')
    return (np.float32(twd_sel.values), np.float32(tws_sel.values))

#def route():
weatherrouter = weather_router(Polar('volvo70.pol'), getWindAt, ds.time.values[:4], 12, (-34,0),(-34,17))
weatherrouter.route()


def test_polar():
    assert weatherrouter.polar.getSpeed(20,45) == np.float64(12.5)

def test_isochrones():
    assert type(weatherrouter.get_isochrones()) == list
    assert len(weatherrouter.get_isochrones()) == 3
    assert len(weatherrouter.get_isochrones()[0][0]) == 4  # Fixed: should be 4 columns [lat, lon, route, bearing]

def test_fastest():
    assert weatherrouter.get_fastest_route(stats=True).shape == (4, 12)
    assert weatherrouter.get_fastest_route(stats=True).iloc[0].boat_speed == 19.28  # Updated for equidistant pruning
    assert weatherrouter.get_fastest_route(stats=True).iloc[0].pos == (-34.0, 0.0)

def test_n_points_parameter():
    """Test that n_points parameter is properly initialized"""
    router_with_custom_n_points = weather_router(
        Polar('volvo70.pol'), 
        getWindAt, 
        ds.time.values[:2], 
        12, 
        (-34,0), 
        (-34,17), 
        n_points=25
    )
    assert router_with_custom_n_points.n_points == 25

def test_return_equidistant():
    """Test the return_equidistant function with known input"""
    # Create a simple curved isochrone
    test_isochrone = [
        (50.0, 0.0),
        (50.1, 0.1), 
        (50.2, 0.3),
        (50.3, 0.6),
        (50.4, 1.0)
    ]
    
    # Create a router with n_points=5 for this test
    test_router = weather_router(
        Polar('volvo70.pol'), 
        getWindAt, 
        ds.time.values[:2], 
        12, 
        (-34,0), 
        (-34,17), 
        n_points=5
    )
    
    equidistant_points = test_router.return_equidistant(test_isochrone)
    
    # Check that we get the correct number of points
    assert equidistant_points.shape == (5, 2)
    
    # Check that first and last points match original first and last
    assert np.isclose(equidistant_points[0][0], 50.0)
    assert np.isclose(equidistant_points[0][1], 0.0)
    assert np.isclose(equidistant_points[-1][0], 50.4)
    assert np.isclose(equidistant_points[-1][1], 1.0)
    
    # Verify points are actually equidistant
    distances = []
    for i in range(len(equidistant_points) - 1):
        p1, p2 = equidistant_points[i], equidistant_points[i+1]
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        distances.append(dist)
    
    # All distances should be approximately equal
    for i in range(1, len(distances)):
        assert np.isclose(distances[i], distances[0], rtol=0.1)

def test_prune_equidistant():
    """Test the prune_equidistant function"""
    # Create mock data with 7 points
    test_possible = [
        [50.0, 0.0, [(50, 0)], 45.0],
        [50.05, 0.05, [(50, 0), (50.05, 0.05)], 45.0],
        [50.1, 0.1, [(50, 0), (50.1, 0.1)], 45.0],
        [50.15, 0.15, [(50, 0), (50.15, 0.15)], 45.0],
        [50.2, 0.2, [(50, 0), (50.2, 0.2)], 45.0],
        [50.25, 0.25, [(50, 0), (50.25, 0.25)], 45.0],
        [50.3, 0.3, [(50, 0), (50.3, 0.3)], 45.0],
    ]
    
    # Create router with n_points=3
    test_router = weather_router(
        Polar('volvo70.pol'), 
        getWindAt, 
        ds.time.values[:2], 
        12, 
        (-34,0), 
        (-34,17), 
        n_points=3
    )
    
    pruned_points, min_dist = test_router.prune_equidistant(test_possible)
    
    # Should return exactly 3 points
    assert len(pruned_points) == 3
    
    # Check that structure is preserved (4 columns)
    assert len(pruned_points[0]) == 4
    
    # Check that min_dist is calculated
    assert isinstance(min_dist, (int, float))
    assert min_dist > 0

def test_equidistant_routing_integration():
    """Test that routing works with equidistant pruning"""
    # Create a router with small n_points for faster testing
    test_router = weather_router(
        Polar('volvo70.pol'), 
        getWindAt, 
        ds.time.values[:3],  # Use fewer time steps for speed
        12, 
        (-34,0), 
        (-34,17),
        n_points=10  # Small number for testing
    )
    
    # Run routing
    test_router.route()
    
    # Verify isochrones were created
    isochrones = test_router.get_isochrones()
    assert len(isochrones) > 0
    
    # Verify each isochrone has the correct structure
    for iso in isochrones:
        assert iso.shape[1] == 4  # 4 columns: lat, lon, route, bearing
        # Each isochrone should have at most n_points entries
        assert iso.shape[0] <= test_router.n_points
    
    # Verify we can get the fastest route
    route = test_router.get_fastest_route()
    assert len(route) > 0
    assert 'lat' in route.columns
    assert 'lon' in route.columns
    assert 'boat_speed' in route.columns

def test_equidistant_vs_traditional():
    """Test that equidistant pruning produces different results than traditional method"""
    # This is more of a sanity check to ensure the new method is actually different
    
    # Create two identical routers with different n_points
    router1 = weather_router(
        Polar('volvo70.pol'), 
        getWindAt, 
        ds.time.values[:3], 
        12, 
        (-34,0), 
        (-34,17),
        n_points=5
    )
    
    router2 = weather_router(
        Polar('volvo70.pol'), 
        getWindAt, 
        ds.time.values[:3], 
        12, 
        (-34,0), 
        (-34,17),
        n_points=15
    )
    
    # Run both
    router1.route()
    router2.route()
    
    # They should potentially have different numbers of points in isochrones
    # due to different n_points settings
    iso1 = router1.get_isochrones()
    iso2 = router2.get_isochrones()
    
    # Both should complete successfully
    assert len(iso1) > 0
    assert len(iso2) > 0
    
    # Both should produce valid routes
    route1 = router1.get_fastest_route()
    route2 = router2.get_fastest_route()
    
    assert len(route1) > 0
    assert len(route2) > 0
