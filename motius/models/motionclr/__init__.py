"""MotionCLR model bundle and official-compatible network."""

from .bundle import (
    DEFAULT_NETWORK_CONFIG,
    MOTIONCLR_DIM,
    MOTIONCLR_FPS,
    MOTIONCLR_MAX_FRAMES,
    MOTIONCLR_REPO_ID,
    MOTIONCLR_SOURCE_REVISION,
    MotionCLRBundle,
)
from .network import MotionCLR

__all__ = [
    "DEFAULT_NETWORK_CONFIG",
    "MOTIONCLR_DIM",
    "MOTIONCLR_FPS",
    "MOTIONCLR_MAX_FRAMES",
    "MOTIONCLR_REPO_ID",
    "MOTIONCLR_SOURCE_REVISION",
    "MotionCLR",
    "MotionCLRBundle",
]
