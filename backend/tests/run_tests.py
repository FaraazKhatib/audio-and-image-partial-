"""Runs every test module without pytest, so the suite works on a bare Python install.

    python tests/run_tests.py

pytest also collects these files if you have it.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import test_audio
import test_core
import test_engine

MODULES = [("core", test_core), ("engine", test_engine), ("audio", test_audio)]


def main() -> int:
    failures = 0
    for name, module in MODULES:
        print(f"\n{name}")
        failures += module._run_all()
    print("\nall modules passed" if failures == 0 else f"\n{failures} module(s) had failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
