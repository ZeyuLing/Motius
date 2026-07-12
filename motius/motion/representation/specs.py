"""Motion representation channel specs.

The specs in this module are intentionally small, import-light descriptors used
by docs, converters, and model cards. Heavy conversion code lives in sibling
modules such as :mod:`motius.motion.representation.motion272` and
:mod:`motius.motion.representation.dart276`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class MotionRepresentationSpec:
    """Static metadata for one motion representation."""

    name: str
    dim: int
    fps: float | None
    layout: Tuple[Tuple[str, int, int, str], ...]
    coordinate_frame: str
    rotation_convention: str
    notes: str = ""


HML263 = MotionRepresentationSpec(
    name="hml263",
    dim=263,
    fps=20.0,
    coordinate_frame="HumanML3D canonical Y-up, root-yaw-relative RIC features",
    rotation_convention="root yaw velocity + local joint rotations/positions",
    layout=(
        ("root_yaw_velocity", 0, 1, "scalar angular velocity"),
        ("root_xz_velocity", 1, 3, "root planar velocity in heading frame"),
        ("root_height", 3, 4, "absolute root Y height"),
        ("ric_positions", 4, 67, "21 non-root joints relative to root"),
        ("local_rotations", 67, 193, "21 joints * 6D continuous rotations"),
        ("local_velocities", 193, 259, "22 joints * XYZ velocities"),
        ("foot_contacts", 259, 263, "binary foot contact channels"),
    ),
    notes="Native HumanML3D/T2M-GPT/MoMask representation.",
)


MS272 = MotionRepresentationSpec(
    name="ms272",
    dim=272,
    fps=30.0,
    coordinate_frame="MotionStreamer humanml3d_272 canonical Y-up frame",
    rotation_convention="first-two-rows 6D: R[:2, :].reshape(6)",
    layout=(
        ("root_local_xz_velocity", 0, 2, "heading-removed root xz velocity"),
        ("heading_velocity_6d", 2, 8, "heading angular velocity as 6D rotation"),
        ("joint_positions", 8, 74, "22 joints * XYZ, heading/root aligned"),
        ("joint_velocities", 74, 140, "22 joints * XYZ velocity"),
        ("local_rotations_rows6d", 140, 272, "22 joints * R[:2, :].reshape(6)"),
    ),
    notes="Native MotionStreamer / GoToZero evaluator representation.",
)


MOTION135 = MotionRepresentationSpec(
    name="motion135",
    dim=135,
    fps=30.0,
    coordinate_frame="SMPL-22 root translation plus local rotations",
    rotation_convention="row-interleaved first-two-column 6D: R[:, :2].reshape(6)",
    layout=(
        ("root_translation", 0, 3, "SMPL root translation in metres"),
        ("local_rotations_row6d", 3, 135, "22 joints * R[:, :2].reshape(6)"),
    ),
    notes="Repository-canonical SMPL body representation for mesh viewers and bridges.",
)


HYMOTION201 = MotionRepresentationSpec(
    name="hymotion201",
    dim=201,
    fps=30.0,
    coordinate_frame="HY-Motion O6DP Y-up frame with absolute root translation",
    rotation_convention="row-interleaved first-two-column local 6D rotations",
    layout=(
        ("root_translation", 0, 3, "absolute pelvis translation in metres"),
        ("root_rotation_row6d", 3, 9, "SMPL root/global orientation"),
        ("body_rotations_row6d", 9, 135, "21 parent-relative body rotations"),
        ("root_invariant_joints", 135, 201, "22 pelvis-relative XYZ joints"),
    ),
    notes=(
        "Official HY-Motion-201. Channels 135:138 are the pelvis RIC triplet "
        "and should be zero. The internal 198-dim variant removes that triplet."
    ),
)


DART276 = MotionRepresentationSpec(
    name="dart276",
    dim=276,
    fps=20.0,
    coordinate_frame=(
        "DART / ViMoGen canonical Z-up frame: first-frame pelvis centred, "
        "first-frame hips/shoulders used to align body facing"
    ),
    rotation_convention=(
        "row-interleaved first-two-column 6D: [R00,R01,R10,R11,R20,R21], "
        "matching convention='row' in motius.motion.representation.rotation"
    ),
    layout=(
        ("body_pose_rot6d", 0, 126, "21 SMPL body joints * DART/row 6D local rotations"),
        ("joint_positions", 126, 192, "22 canonical joints * XYZ"),
        ("joint_velocities", 192, 258, "22 canonical joints * XYZ velocity"),
        ("root_orient_rot6d", 258, 264, "root/global orientation as DART/row 6D"),
        ("root_orient_velocity_rot6d", 264, 270, "R[t+1] @ R[t]^T as DART/row 6D"),
        ("root_translation", 270, 273, "canonical root translation"),
        ("root_translation_velocity", 273, 276, "root translation velocity"),
    ),
    notes=(
        "The stored sequence has length T-1 for an original T-frame SMPL/joint clip. "
        "Use equal_length=True when decoding back to T frames."
    ),
)


G1_38 = MotionRepresentationSpec(
    name="g1_38",
    dim=38,
    fps=30.0,
    coordinate_frame="Unitree G1 MuJoCo Z-up canonical frame",
    rotation_convention="row-interleaved first-two-column root rotation 6D",
    layout=(
        ("root_xy_velocity_and_height", 0, 3, "default: vx, vy, absolute z"),
        ("root_rotation_row6d", 3, 9, "floating-base root orientation"),
        ("joint_angles", 9, 38, "29 G1 DOF angles in radians"),
    ),
    notes="Decodes to 36-d MuJoCo qpos: root xyz + quaternion wxyz + 29 DOF.",
)


SPECS = {
    spec.name: spec
    for spec in (
        HML263,
        MS272,
        MOTION135,
        HYMOTION201,
        DART276,
        G1_38,
    )
}


__all__ = [
    "MotionRepresentationSpec",
    "HML263",
    "MS272",
    "MOTION135",
    "HYMOTION201",
    "DART276",
    "G1_38",
    "SPECS",
]
