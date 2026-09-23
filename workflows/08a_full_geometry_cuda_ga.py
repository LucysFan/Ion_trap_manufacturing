"""Compatibility launcher for the renamed CPU/CUDA workflow."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    target = Path(__file__).with_name("08a_full_geometry_ga.py")
    runpy.run_path(str(target), run_name="__main__")
