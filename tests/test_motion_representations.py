import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from motius.motion.retarget import GMR_Y_UP_FROM_Z_UP, GMR_Z_UP_FROM_Y_UP
from motius.motion.retarget._hml263_smpl_impl import _resolve_smplx_model_root

from motius.motion.representation.convert import (
    convert_motion,
    smpl_to_hml263,
    smpl_to_joints,
)
from motius.motion.representation.humanml import joints_to_hml263
from motius.motion.representation.dart276 import (
    DART276_DIM,
    dart276_to_motion135,
    dart276_to_smpl_params,
    smpl_params_and_joints_to_dart276,
)
from motius.motion.representation.hymotion import (
    hymotion201_to_joints,
    hymotion201_to_motion135,
    motion135_to_hymotion201,
)
from motius.motion.representation.motion272 import (
    encode_smpl_to_272,
    motion272_to_motion135,
    recover_272_stored_positions,
    recover_local_rotations_and_root,
)
from motius.motion.representation.rotation import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from motius.motion.skeleton.fk import motion135_to_fk
from motius.motion.skeleton.names import SMPL22_PARENTS
from tools.convert_hml263_predictions import _relative_offsets


def _identity_motion135(frames: int) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    identity6 = matrix_to_rotation_6d(matrix, convention="row")
    motion = np.zeros((frames, 135), dtype=np.float32)
    motion[:, 3:] = np.tile(identity6, 22)
    return motion


def test_smpl_loader_accepts_root_or_direct_smpl_directory(tmp_path: Path) -> None:
    direct = tmp_path / "body_models" / "smpl"
    direct.mkdir(parents=True)
    (direct / "SMPL_NEUTRAL.pkl").touch()
    assert _resolve_smplx_model_root(direct) == direct.parent
    assert _resolve_smplx_model_root(direct.parent) == direct.parent


def test_relative_smpl_offsets_preserve_root_and_parent_differences() -> None:
    rest = np.arange(66, dtype=np.float32).reshape(22, 3) / 10
    offsets = _relative_offsets(rest, np.asarray(SMPL22_PARENTS))
    np.testing.assert_array_equal(offsets[0], rest[0])
    for joint, parent in enumerate(SMPL22_PARENTS[1:], start=1):
        np.testing.assert_array_equal(offsets[joint], rest[joint] - rest[parent])


def test_explicit_6d_layouts_are_not_interchangeable():
    matrix = axis_angle_to_matrix(np.array([0.2, -0.3, 0.1], dtype=np.float64))
    motion135_6d = matrix_to_rotation_6d(matrix, convention="row")
    ms272_6d = matrix[:2, :].reshape(6)

    np.testing.assert_allclose(motion135_6d, matrix[:, :2].reshape(6), atol=1e-7)
    assert not np.allclose(motion135_6d, ms272_6d)
    np.testing.assert_allclose(
        rotation_6d_to_matrix(motion135_6d, convention="row"), matrix, atol=1e-6
    )


def test_hymotion201_fk_and_prefix_roundtrip():
    motion135 = _identity_motion135(6)
    motion135[:, 0] = np.linspace(0.0, 0.5, len(motion135))
    offsets = np.zeros((22, 3), dtype=np.float32)
    offsets[1:, 1] = 0.05

    motion201 = motion135_to_hymotion201(motion135, offsets)
    assert motion201.shape == (6, 201)
    np.testing.assert_allclose(motion201[:, 135:138], 0.0, atol=1e-7)
    np.testing.assert_allclose(hymotion201_to_motion135(motion201), motion135, atol=1e-7)

    fk_joints, _, _, _ = motion135_to_fk(
        torch.from_numpy(motion135), torch.from_numpy(offsets)
    )
    np.testing.assert_allclose(
        hymotion201_to_joints(motion201), fk_joints.numpy(), atol=1e-6
    )
    np.testing.assert_allclose(
        convert_motion(motion201, "hymotion201", "joints", rotation_space="local"),
        fk_joints.numpy(),
        atol=1e-6,
    )


def test_ms272_native_decode_and_motion135_repack():
    rng = np.random.default_rng(4)
    frames = 8
    local = axis_angle_to_matrix(rng.normal(scale=0.12, size=(frames, 22, 3)))
    root = np.zeros((frames, 3), dtype=np.float64)
    root[:, 0] = np.linspace(0.0, 0.4, frames)
    offsets = np.zeros((22, 3), dtype=np.float64)
    offsets[1:, 1] = np.linspace(0.02, 0.08, 21)

    world = np.zeros((frames, 22, 3), dtype=np.float64)
    world[:, 0] = root
    for joint in range(1, 22):
        world[:, joint] = world[:, 0] + offsets[joint]

    motion272 = encode_smpl_to_272(world, local)
    decoded_rot, decoded_root = recover_local_rotations_and_root(motion272)
    decoded_joints = recover_272_stored_positions(motion272)
    motion135 = motion272_to_motion135(motion272)

    assert motion272.shape == (frames, 272)
    assert decoded_rot.shape == (frames, 22, 3, 3)
    assert decoded_root.shape == (frames, 3)
    assert decoded_joints.shape == (frames, 22, 3)
    np.testing.assert_allclose(
        rotation_6d_to_matrix(
            motion135[:, 3:].reshape(frames, 22, 6), convention="row"
        ),
        decoded_rot,
        atol=1e-6,
    )


def test_hml263_to_joints_dispatch_shape():
    motion = np.zeros((7, 263), dtype=np.float32)
    joints = convert_motion(motion, "humanml3d-263", "joints")
    assert joints.shape == (7, 22, 3)


def test_native_hml263_encoder_matches_official_sample():
    fixture = Path(__file__).parent / "assets/humanml3d/004822_protocol.npz"
    with np.load(fixture) as sample:
        encoded = joints_to_hml263(sample["joints"])
        expected = sample["hml263"]
    np.testing.assert_allclose(encoded, expected, atol=1e-4, rtol=0)
    np.testing.assert_array_equal(encoded[:, 259:263], expected[:, 259:263])


def test_shape_aware_smpl_api_uses_betas_and_dispatches_to_hml263(tmp_path: Path):
    offsets = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.08, 0.0, 0.0],
            [-0.08, 0.0, 0.0],
            [0.0, 0.12, 0.0],
            [0.0, -0.40, 0.0],
            [0.0, -0.40, 0.0],
            [0.0, 0.12, 0.0],
            [0.0, -0.40, 0.0],
            [0.0, -0.40, 0.0],
            [0.0, 0.12, 0.0],
            [0.0, 0.0, 0.12],
            [0.0, 0.0, 0.12],
            [0.0, 0.18, 0.0],
            [0.12, 0.0, 0.0],
            [-0.12, 0.0, 0.0],
            [0.0, 0.0, 0.12],
            [0.0, -0.12, 0.0],
            [0.0, -0.12, 0.0],
            [0.0, -0.24, 0.0],
            [0.0, -0.24, 0.0],
            [0.0, -0.24, 0.0],
            [0.0, -0.24, 0.0],
        ],
        dtype=np.float64,
    )
    rest = np.zeros((22, 3), dtype=np.float64)
    for joint, parent in enumerate(SMPL22_PARENTS):
        rest[joint] = offsets[joint] if parent < 0 else rest[parent] + offsets[joint]
    shapedirs = np.zeros((22, 3, 2), dtype=np.float64)
    shapedirs[:, 0, 0] = np.arange(22) * 0.001
    regressor = np.eye(22, dtype=np.float64)
    tree = np.stack([np.asarray(SMPL22_PARENTS), np.arange(22)])
    model_path = tmp_path / "model.npz"
    np.savez(
        model_path,
        v_template=rest,
        shapedirs=shapedirs,
        J_regressor=regressor,
        kintree_table=tree,
    )

    frames = 4
    global_orient = np.zeros((frames, 3), dtype=np.float32)
    body_pose = np.zeros((frames, 63), dtype=np.float32)
    transl = np.zeros((frames, 3), dtype=np.float32)
    beta = np.asarray([1.5, 0.0], dtype=np.float32)
    joints = smpl_to_joints(
        global_orient,
        body_pose,
        transl,
        betas=beta,
        model_path=model_path,
    )
    expected_rest = rest + shapedirs[..., 0] * beta[0]
    np.testing.assert_allclose(
        joints, np.repeat(expected_rest[None], frames, axis=0), atol=1e-6
    )

    mapping = {
        "global_orient": global_orient,
        "body_pose": body_pose,
        "transl": transl,
        "betas": beta,
        "gender": np.asarray("neutral"),
    }
    direct = smpl_to_hml263(
        global_orient,
        body_pose,
        transl,
        betas=beta,
        model_path=model_path,
    )
    dispatched = convert_motion(
        mapping,
        "smpl",
        "hml263",
        model_path=model_path,
    )
    np.testing.assert_array_equal(direct, dispatched)


def test_dart276_smpl_roundtrip_and_motion135_layout():
    torch.manual_seed(7)
    frames = 7
    global_orient = torch.randn(frames, 3) * 0.2
    body_pose = torch.randn(frames, 21, 3) * 0.15
    transl = torch.cumsum(torch.randn(frames, 3) * 0.02, dim=0)
    joints = torch.cumsum(torch.randn(frames, 22, 3) * 0.01, dim=0)

    motion276 = smpl_params_and_joints_to_dart276(
        {"global_orient": global_orient, "body_pose": body_pose, "transl": transl},
        joints,
    )
    assert motion276.shape == (frames - 1, DART276_DIM)
    smpl, joints_rt = dart276_to_smpl_params(motion276, equal_length=True)
    motion276_rt = smpl_params_and_joints_to_dart276(smpl, joints_rt)
    torch.testing.assert_close(motion276, motion276_rt, atol=1e-4, rtol=1e-4)

    row = dart276_to_motion135(motion276, rotation_convention="row")
    column = dart276_to_motion135(motion276, rotation_convention="column")
    assert row.shape == column.shape == (frames, 135)
    assert not torch.allclose(row[:, 3:], column[:, 3:])


def test_convert_motion_cli_roundtrip(tmp_path: Path):
    motion135 = _identity_motion135(5)
    offsets = np.zeros((22, 3), dtype=np.float32)
    offsets[1:, 1] = 0.1
    source = tmp_path / "motion135.npy"
    offset_path = tmp_path / "offsets.npy"
    output = tmp_path / "motion201.npz"
    np.save(source, motion135)
    np.save(offset_path, offsets)

    subprocess.run(
        [
            sys.executable,
            "tools/convert_motion.py",
            str(source),
            str(output),
            "--src",
            "motion135",
            "--dst",
            "hymotion201",
            "--bone-offsets",
            str(offset_path),
        ],
        check=True,
        cwd=Path(__file__).resolve().parents[1],
    )
    with np.load(output) as result:
        assert result["motion"].shape == (5, 201)
        assert result["representation"].item() == "hymotion201"


def test_g1_output_helpers_and_motion135_rotation_decode(tmp_path: Path):
    from motius.motion.retarget.smpl_g1 import GMRSMPLToG1Retargeter

    retargeter = GMRSMPLToG1Retargeter(ground_align=False, smooth=False)
    captured = {}

    def fake_retarget(root_orient, pose_body, trans, **kwargs):
        captured["root_orient"] = root_orient
        captured["pose_body"] = pose_body
        frames = len(trans)
        return {
            "dof_pos": np.zeros((frames, 29), dtype=np.float32),
            "root_pos": trans,
            "root_orient_quat": np.tile([1, 0, 0, 0], (frames, 1)).astype(np.float32),
            "root_rot": np.tile([0, 0, 0, 1], (frames, 1)).astype(np.float32),
            "fps": 30.0,
            "joint_names": None,
            "dof": 29,
        }

    retargeter.retarget_smplx = fake_retarget
    result = retargeter.retarget_from_motion135(_identity_motion135(4))
    np.testing.assert_allclose(captured["root_orient"], 0.0, atol=1e-6)
    np.testing.assert_allclose(captured["pose_body"], 0.0, atol=1e-6)
    assert retargeter.to_mujoco_qpos(result).shape == (4, 36)

    output = tmp_path / "g1.pkl"
    retargeter.to_asap_pkl(result, str(output))
    assert output.is_file()


def test_gmr_z_up_axis_mapping_preserves_robot_forward_semantics():
    np.testing.assert_allclose(
        GMR_Y_UP_FROM_Z_UP @ GMR_Z_UP_FROM_Y_UP,
        np.eye(3),
        atol=1e-7,
    )
    smpl_forward = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    smpl_up = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(GMR_Z_UP_FROM_Y_UP @ smpl_forward, [1.0, 0.0, 0.0])
    np.testing.assert_allclose(GMR_Z_UP_FROM_Y_UP @ smpl_up, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(
        GMR_Y_UP_FROM_Z_UP @ (GMR_Z_UP_FROM_Y_UP @ smpl_forward),
        smpl_forward,
    )


def test_g1_38_qpos_roundtrip_without_canonicalization():
    frames = 5
    qpos = np.zeros((frames, 36), dtype=np.float32)
    qpos[:, 0] = np.linspace(0.0, 0.4, frames)
    qpos[:, 2] = 0.75
    qpos[:, 3] = 1.0
    qpos[:, 7:] = np.linspace(-0.2, 0.2, 29)

    motion = convert_motion(
        qpos,
        "g1_qpos",
        "g1_38",
        canonicalize=False,
        root_velocity=True,
    )
    decoded = convert_motion(motion, "g1_38", "g1_qpos", root_velocity=True)
    np.testing.assert_allclose(decoded, qpos, atol=1e-6)
