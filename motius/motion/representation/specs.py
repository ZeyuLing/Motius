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


INTERHUMAN262 = MotionRepresentationSpec(
    name="interhuman262",
    dim=262,
    fps=30.0,
    coordinate_frame=(
        "InterHuman canonical Y-up world frame shared by both people; person one "
        "starts at the origin facing +Z and person two retains relative placement"
    ),
    rotation_convention=(
        "21 non-root SMPL local rotations as row-interleaved first-two-column 6D: "
        "R[:, :2].reshape(6); root rotation is represented implicitly by global "
        "joint positions"
    ),
    layout=(
        ("joint_positions", 0, 66, "22 global SMPL joints * XYZ"),
        ("joint_velocities", 66, 132, "22 global SMPL joints * XYZ displacement"),
        ("local_rotations", 132, 258, "21 non-root joints * 6D local rotations"),
        ("foot_contacts", 258, 262, "left heel/toe and right heel/toe contacts"),
    ),
    notes=(
        "Per-person feature width is 262. A complete two-person sample is (T,2,262). "
        "Canonicalize the pair jointly to preserve interaction geometry."
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


ARDY_330 = MotionRepresentationSpec(
    name="ardy_330",
    dim=330,
    fps=20.0,
    coordinate_frame="ARDY 27-joint Y-up world frame used by released ARDY checkpoints",
    rotation_convention="ARDY continuous 6D global rotations via matrix_to_cont6d",
    layout=(
        ("root_position", 0, 3, "absolute XYZ root position"),
        ("global_root_heading", 3, 5, "cosine and sine of global heading"),
        ("root_local_joint_positions", 5, 83, "26 non-root ARDY joints * XYZ"),
        ("global_joint_rotations_6d", 83, 245, "27 global joint rotations * 6D"),
        ("global_joint_velocities", 245, 326, "27 joints * XYZ velocity"),
        ("foot_contacts", 326, 330, "left/right heel and toe contacts"),
    ),
    notes=(
        "Native explicit tensor for NVIDIA ARDY's 27-joint skeleton checkpoints. "
        "ARDY-330 is not an SMPL-family body model; Motius exposes a named ARDY-27 <-> "
        "SMPL-22 joint-position bridge for viewers and joint evaluators."
    ),
)


# Backward-compatible symbol for older internal imports.
ARDY_CORE330 = ARDY_330


ARDY_G1_414 = MotionRepresentationSpec(
    name="ardy_g1_414",
    dim=414,
    fps=25.0,
    coordinate_frame="Unitree G1 Y-up world frame used by released ARDY G1 checkpoints",
    rotation_convention="ARDY continuous 6D global rotations via matrix_to_cont6d",
    layout=(
        ("root_position", 0, 3, "absolute XYZ pelvis position"),
        ("global_root_heading", 3, 5, "cosine and sine of global heading"),
        ("root_local_joint_positions", 5, 104, "33 non-root G1 joints * XYZ"),
        ("global_joint_rotations_6d", 104, 308, "34 global joint rotations * 6D"),
        ("global_joint_velocities", 308, 410, "34 joints * XYZ velocity"),
        ("foot_contacts", 410, 414, "left/right heel and toe contacts"),
    ),
    notes=(
        "Native explicit tensor for ARDY's Unitree G1 checkpoints. This is the "
        "same robot family as the public Unitree G1 representation; it is not "
        "listed as a separate body model."
    ),
)


MOTIONBRICKS_G1_414 = MotionRepresentationSpec(
    name="motionbricks_g1_414",
    dim=414,
    fps=30.0,
    coordinate_frame="MotionBricks Unitree G1 Y-up motion space, Z-forward",
    rotation_convention="continuous 6D global rotations for 34 G1 joints",
    layout=(
        ("global_root_pos", 0, 3, "XYZ pelvis position in motion space"),
        ("global_root_heading", 3, 5, "cosine and sine of Y-axis heading"),
        ("ric_data", 5, 104, "33 non-root G1 joint positions minus projected root XZ"),
        ("global_rot_data", 104, 308, "34 global joint rotations * 6D"),
        ("local_vel", 308, 410, "34 world-frame joint velocities * XYZ"),
        ("foot_contacts", 410, 414, "left/right ankle and toe contacts"),
    ),
    notes=(
        "Global subset used by the MotionBricks root model and data loader. "
        "It shares the 409D body block with the local 413D subset."
    ),
)


MOTIONBRICKS_G1_413 = MotionRepresentationSpec(
    name="motionbricks_g1_413",
    dim=413,
    fps=30.0,
    coordinate_frame="MotionBricks Unitree G1 local-root subset",
    rotation_convention="continuous 6D global rotations for the shared 409D body block",
    layout=(
        ("local_root_rot_vel", 0, 1, "Y-axis angular velocity"),
        ("local_root_vel", 1, 3, "heading-frame XZ root velocity"),
        ("global_root_y", 3, 4, "absolute root height"),
        ("body_features", 4, 413, "shared 409D body block: positions, rotations, velocities, contacts"),
    ),
    notes="Local-root subset consumed by MotionBricks pose/tokenizer modules.",
)


MOTIONBRICKS_G1_418 = MotionRepresentationSpec(
    name="motionbricks_g1_418",
    dim=418,
    fps=30.0,
    coordinate_frame="MotionBricks full dual-root Unitree G1 representation",
    rotation_convention="continuous 6D global rotations for the shared 409D body block",
    layout=(
        ("global_root", 0, 5, "global root subset"),
        ("local_root", 5, 9, "local root subset"),
        ("body_features", 9, 418, "shared G1 body features"),
    ),
    notes="Full dual-root representation; global 414D and local 413D subsets are losslessly convertible.",
)


SPECS = {
    spec.name: spec
    for spec in (
        HML263,
        MS272,
        MOTION135,
        HYMOTION201,
        DART276,
        INTERHUMAN262,
        G1_38,
        ARDY_330,
        ARDY_G1_414,
        MOTIONBRICKS_G1_414,
        MOTIONBRICKS_G1_413,
        MOTIONBRICKS_G1_418,
    )
}

# Backward-compatible lookup key.
SPECS["ardy_core330"] = ARDY_330


__all__ = [
    "MotionRepresentationSpec",
    "HML263",
    "MS272",
    "MOTION135",
    "HYMOTION201",
    "DART276",
    "INTERHUMAN262",
    "G1_38",
    "ARDY_330",
    "ARDY_CORE330",
    "ARDY_G1_414",
    "MOTIONBRICKS_G1_414",
    "MOTIONBRICKS_G1_413",
    "MOTIONBRICKS_G1_418",
    "SPECS",
]
