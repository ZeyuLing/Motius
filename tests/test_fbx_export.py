from pathlib import Path

import numpy as np
import pytest

from motius.motion.fbx import SMPLAnimation, SMPL_TO_BLENDER
from motius.motion.fbx._mapping import SMPL22_BONE_NAMES, resolve_bone_map
from motius.motion.fbx.api import _prepare_payload
from motius.motion.representation.rotation import matrix_to_rotation_6d
from motius.motion.skeleton.names import SMPL22_NAMES, SMPL22_PARENTS


def _identity_motion135(frames: int) -> np.ndarray:
    identity6 = matrix_to_rotation_6d(np.eye(3), convention="row")
    motion = np.zeros((frames, 135), dtype=np.float64)
    motion[:, 3:] = np.tile(identity6, 22)
    return motion


def _synthetic_smpl(path: Path) -> np.ndarray:
    offsets = np.asarray(
        [
            [0.0, 0.9, 0.0],
            [0.09, -0.08, 0.0], [-0.09, -0.08, 0.0], [0.0, 0.12, 0.0],
            [0.0, -0.38, 0.0], [0.0, -0.38, 0.0], [0.0, 0.13, 0.0],
            [0.0, -0.38, 0.0], [0.0, -0.38, 0.0], [0.0, 0.13, 0.0],
            [0.0, 0.0, 0.13], [0.0, 0.0, 0.13], [0.0, 0.17, 0.0],
            [0.10, 0.04, 0.0], [-0.10, 0.04, 0.0], [0.0, 0.15, 0.0],
            [0.14, 0.0, 0.0], [-0.14, 0.0, 0.0], [0.24, 0.0, 0.0],
            [-0.24, 0.0, 0.0], [0.22, 0.0, 0.0], [-0.22, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    rest = np.zeros((22, 3), dtype=np.float64)
    for joint, parent in enumerate(SMPL22_PARENTS):
        rest[joint] = offsets[joint] if parent < 0 else rest[parent] + offsets[joint]
    shapedirs = np.zeros((22, 3, 2), dtype=np.float64)
    shapedirs[:, 0, 0] = np.linspace(0.0, 0.02, 22)
    faces = np.asarray([[0, 1, 3], [0, 3, 2], [1, 4, 3]], dtype=np.int32)
    np.savez(
        path,
        v_template=rest,
        shapedirs=shapedirs,
        J_regressor=np.eye(22, dtype=np.float64),
        kintree_table=np.stack([SMPL22_PARENTS, np.arange(22)]),
        weights=np.eye(22, dtype=np.float64),
        f=faces,
    )
    return rest


def test_smpl_animation_from_motion135_uses_row_convention() -> None:
    motion = _identity_motion135(4)
    motion[:, :3] = np.arange(12).reshape(4, 3) / 10
    animation = SMPLAnimation.from_motion135(motion, betas=[0.2, -0.1], fps=20)
    expected = np.broadcast_to(np.eye(3), animation.local_rotations.shape)
    np.testing.assert_allclose(animation.local_rotations, expected, atol=1e-7)
    np.testing.assert_array_equal(animation.translations, motion[:, :3])
    assert animation.frames == 4
    assert animation.fps == 20


def test_smpl_animation_rejects_varying_shape() -> None:
    frames = 2
    with pytest.raises(ValueError, match="constant betas"):
        SMPLAnimation.from_smpl(
            np.zeros((frames, 3)),
            np.zeros((frames, 21, 3)),
            np.zeros((frames, 3)),
            betas=np.asarray([[0.0], [1.0]]),
        )


def test_payload_builds_shaped_skin_and_blender_coordinates(tmp_path: Path) -> None:
    model_path = tmp_path / "synthetic_smpl.npz"
    rest = _synthetic_smpl(model_path)
    motion = _identity_motion135(3)
    motion[:, 2] = [0.0, 0.2, 0.4]
    animation = SMPLAnimation.from_motion135(motion, betas=[1.0, 0.0])
    payload, resolved = _prepare_payload(
        animation, model_path=model_path, model_type="smpl", gender="neutral"
    )

    shaped = rest.copy()
    shaped[:, 0] += np.linspace(0.0, 0.02, 22)
    np.testing.assert_allclose(payload["vertices"], shaped @ SMPL_TO_BLENDER.T)
    np.testing.assert_allclose(payload["weights"], np.eye(22))
    expected = np.broadcast_to(np.eye(3), payload["global_rotations"].shape)
    np.testing.assert_allclose(payload["global_rotations"], expected, atol=1e-7)
    np.testing.assert_allclose(payload["joints"][:, 0, 1], [0.0, -0.2, -0.4])
    assert resolved == model_path.resolve()


def test_auto_mapping_supports_namespaces_and_mixamo_names() -> None:
    mixamo = (
        "Hips", "LeftUpLeg", "RightUpLeg", "Spine", "LeftLeg", "RightLeg",
        "Spine1", "LeftFoot", "RightFoot", "Spine2", "LeftToeBase",
        "RightToeBase", "Neck", "LeftShoulder", "RightShoulder", "Head",
        "LeftArm", "RightArm", "LeftForeArm", "RightForeArm", "LeftHand",
        "RightHand",
    )
    names = [f"mixamorig:{name}" for name in mixamo]
    mapping = resolve_bone_map(names)
    assert mapping["Pelvis"] == "mixamorig:Hips"
    assert mapping["L_Wrist"] == "mixamorig:LeftHand"
    assert len(mapping) == 22


def test_mapping_requires_complete_unambiguous_target_by_default() -> None:
    assert tuple(SMPL22_NAMES) == SMPL22_BONE_NAMES
    with pytest.raises(ValueError, match="missing required"):
        resolve_bone_map(["Pelvis"])
    assert resolve_bone_map(["Pelvis"], strict=False) == {"Pelvis": "Pelvis"}
    with pytest.raises(ValueError, match="do not exist"):
        resolve_bone_map(SMPL22_BONE_NAMES, {"Pelvis": "missing"})
