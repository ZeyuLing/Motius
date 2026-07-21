"""Self-contained EDGE network and motion math."""

from .model import DanceDecoder
from .motion import (
    EDGE_CONTACT_DIM,
    EDGE_MOTION_DIM,
    EDGE_REPR_DIM,
    edge_motion_to_aistpp_joints,
    edge_motion_to_motion135,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)

__all__ = [
    "DanceDecoder",
    "EDGE_CONTACT_DIM",
    "EDGE_MOTION_DIM",
    "EDGE_REPR_DIM",
    "edge_motion_to_aistpp_joints",
    "edge_motion_to_motion135",
    "matrix_to_rotation_6d",
    "rotation_6d_to_matrix",
]
