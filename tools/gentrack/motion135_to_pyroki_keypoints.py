#!/usr/bin/env python3
"""Run the package-owned motion-135 to PyRoki keypoint converter."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


if __name__ == "__main__":
    runpy.run_module(
        "motius.motion.retarget.gentrack_pyroki",
        run_name="__main__",
    )
