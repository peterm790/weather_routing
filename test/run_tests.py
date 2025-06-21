#!/usr/bin/env python3
"""
Test runner for weather routing equidistant pruning functionality.
Run this script to execute all tests for the new and existing functionality.
"""

import routing_test
import sys

def run_all_tests():
    """Run all tests and report results"""
    tests = [
        ("n_points parameter", routing_test.test_n_points_parameter),
        ("return_equidistant function", routing_test.test_return_equidistant),
        ("prune_equidistant function", routing_test.test_prune_equidistant),
        ("equidistant routing integration", routing_test.test_equidistant_routing_integration),
        ("polar functionality", routing_test.test_polar),
        ("isochrones structure", routing_test.test_isochrones),
        ("fastest route calculation", routing_test.test_fastest),
        ("equidistant vs traditional comparison", routing_test.test_equidistant_vs_traditional),
    ]
    
    passed = 0
    failed = 0
    
    print("🧪 Running Weather Router Tests")
    print("=" * 50)
    
    for test_name, test_func in tests:
        try:
            print(f"Testing {test_name}...", end=" ")
            test_func()
            print("✅ PASSED")
            passed += 1
        except Exception as e:
            print(f"❌ FAILED: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All tests passed! The equidistant pruning implementation is working correctly.")
        return True
    else:
        print(f"⚠️  {failed} test(s) failed. Please review the implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 