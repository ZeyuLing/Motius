#!/usr/bin/env python3
"""CLI wrapper for the legacy motion-135 to G1 retarget pipeline."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.models.gentrack.pipeline_motion_to_robot import main


if __name__ == "__main__":
    main()
