#!/usr/bin/env python3
"""CLI wrapper for GenTrack's ProtoMotions MuJoCo rollout runtime."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motius.simulators.gentrack.protomotions import main


if __name__ == "__main__":
    main()
