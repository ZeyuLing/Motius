"""Headless scene assets; clips are external data, not bundled model weights."""

import math
from pathlib import Path

import warp as wp

from motius.configuration.motionbricks_runtime_contract import ModelRig

# Presets hold relative paths, rebased per scene. No upstream checkout is required.
ASSETS_ROOT = Path(".")


def model_to_runtime(rig):
    if ModelRig(rig) is ModelRig.SOMA:
        return wp.transform(
            wp.vec3(0.0, 0.0, 0.0),
            wp.quat_from_axis_angle(wp.vec3(1.0, 0.0, 0.0), math.pi / 2),
        )
    return wp.transform_identity()
