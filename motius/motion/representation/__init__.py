"""Public motion representation library."""

from .specs import (
    DART276,
    G1_38,
    HML263,
    HYMOTION201,
    INTERHUMAN262,
    MOTION135,
    MS272,
    SPECS,
    MotionRepresentationSpec,
)
from .interhuman262 import (
    interhuman262_to_foot_contacts,
    interhuman262_to_joint_velocities,
    interhuman262_to_joints,
    interhuman262_to_local_rot6d,
    interhuman262_to_local_rotmat,
    joints_pair_to_interhuman262,
    joints_to_interhuman262,
)
from .g1 import (
    G1_MOTION_DIM,
    G1_QPOS_DIM,
    decode_g1_to_qpos,
    encode_g1_motion,
    encode_g1_qpos,
)
from .convert import (
    convert_motion,
    joints_to_hml263,
    motion135_to_hml263,
    motion135_to_interhuman262,
    motion272_to_hml263,
    smpl_to_hml263,
    smpl_to_humanml263,
    smpl_to_joints,
    smpl_to_motion135,
)


def get_spec(name: str) -> MotionRepresentationSpec:
    """Return a representation spec from a normalized public alias."""

    key = name.lower().replace("-", "").replace("_", "")
    aliases = {
        "humanml3d263": "hml263",
        "humanml263": "hml263",
        "motionstreamer272": "ms272",
        "motion272": "ms272",
        "hymotion201": "hymotion201",
        "interhuman262": "interhuman262",
        "interhuman": "interhuman262",
        "motion135": "motion135",
        "dart276": "dart276",
        "g138": "g1_38",
        "g1motion38": "g1_38",
    }
    canonical = aliases.get(key, key)
    try:
        return SPECS[canonical]
    except KeyError as exc:
        raise KeyError(f"unknown motion representation {name!r}; available: {sorted(SPECS)}") from exc

__all__ = [
    "MotionRepresentationSpec",
    "HML263",
    "MS272",
    "MOTION135",
    "HYMOTION201",
    "INTERHUMAN262",
    "DART276",
    "G1_38",
    "SPECS",
    "get_spec",
    "G1_MOTION_DIM",
    "G1_QPOS_DIM",
    "encode_g1_motion",
    "encode_g1_qpos",
    "decode_g1_to_qpos",
    "interhuman262_to_foot_contacts",
    "interhuman262_to_joint_velocities",
    "interhuman262_to_joints",
    "interhuman262_to_local_rot6d",
    "interhuman262_to_local_rotmat",
    "joints_pair_to_interhuman262",
    "joints_to_interhuman262",
    "convert_motion",
    "joints_to_hml263",
    "motion135_to_hml263",
    "motion135_to_interhuman262",
    "motion272_to_hml263",
    "smpl_to_hml263",
    "smpl_to_humanml263",
    "smpl_to_joints",
    "smpl_to_motion135",
]
