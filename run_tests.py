#!/usr/bin/env python3
"""
Single test runner for this repo.

Usage:
  python run_tests.py
  python run_tests.py -k tack_penalty
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    try:
        import pytest
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "pytest is required to run the test suite. "
            "Install dev deps (e.g. `pip install -r requirements-dev.txt`)."
        ) from e

    repo_root = Path(__file__).resolve().parent
    test_dir = repo_root / "test"
    if not test_dir.exists():
        raise FileNotFoundError(f"Expected test directory at: {test_dir}")

    # Run the full suite under ./test. Forward any extra CLI args to pytest.
    return pytest.main([str(test_dir), *argv])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


