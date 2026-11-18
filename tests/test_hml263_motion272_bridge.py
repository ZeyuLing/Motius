"""Physical-scale HumanML regression: positions and outgoing-frame FK agree."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from motius.motion.representation import convert
from motius.motion.representation.motion272 import (
    _canonical_272_offsets,
    recover_272_stored_positions,
    recover_local_rotations_and_root,
)
from motius.motion.retarget._hml263_smpl_impl import recover_from_ric, resample_linear
from motius.motion.skeleton.names import SMPL22_PARENTS


def real_motion():
    with np.load(Path(__file__).parent / "assets/humanml3d/004822_protocol.npz") as data:
        return data["hml263"].copy()


@pytest.mark.parametrize("target_fps", [20.0, 30.0])
def test_real_humanml_positions_and_canonical_fk(target_fps):
    features = real_motion()
    result = convert.hml263_to_motion272(features, target_fps=target_fps)
    positions = recover_272_stored_positions(result)
    rotations, root = recover_local_rotations_and_root(result)
    offsets = _canonical_272_offsets()
    # Independent SMPL outgoing-frame FK, not the converter's implementation.
    fk = np.zeros_like(positions)
    world = np.zeros_like(rotations)
    fk[:, 0], world[:, 0] = root, rotations[:, 0]
    for joint in range(1, 22):
        parent = SMPL22_PARENTS[joint]
        world[:, joint] = world[:, parent] @ rotations[:, joint]
        fk[:, joint] = fk[:, parent] + (world[:, parent] @ offsets[joint, :, None])[..., 0]
    errors_mm = np.linalg.norm(fk - positions, axis=-1) * 1000
    # Different bone proportions prevent exact equality. Old bridge: 116.8 mm
    # overall and 333.8 / 473.5 mm at the wrists on the 20 fps fixture.
    assert errors_mm.mean() < 45
    assert errors_mm[:, 20:].mean(axis=0).max() < 80
    np.testing.assert_allclose(fk[:, 0], positions[:, 0], atol=1e-6)
    np.testing.assert_allclose(np.linalg.det(rotations), 1, atol=1e-5)

    # Preserve all source positions, including trajectory, modulo the one
    # initial heading rotation and floor/first-root origin used by MS272.
    source = resample_linear(recover_from_ric(features), 20, target_fps)
    origin = source[0, 0].copy()
    origin[1] = source[..., 1].min()
    source -= origin
    # Solve only one rigid rotation for the entire sequence (no per-frame fit).
    alignment = Rotation.align_vectors(source.reshape(-1, 3), positions.reshape(-1, 3))[0]
    np.testing.assert_allclose(alignment.apply(positions.reshape(-1, 3)), source.reshape(-1, 3), atol=2e-5)


def test_matching_bone_geometry_has_exact_position_fk_consistency():
    # A nontrivial articulated sequence on canon272 isolates rotation errors
    # from the real HumanML fixture's different bone lengths/branch geometry.
    frames = 12
    offsets = _canonical_272_offsets()
    poses = np.zeros((frames, 22, 3))
    poses[:, 4, 0] = np.linspace(0, 0.8, frames)
    poses[:, 9, 1] = np.linspace(0, 0.4, frames)
    poses[:, 16, 2] = np.linspace(0, -0.7, frames)
    poses[:, 19, 1] = np.linspace(0, 1.2, frames)
    local = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(frames, 22, 3, 3)
    world = np.zeros_like(local)
    positions = np.zeros((frames, 22, 3))
    positions[:, 0, 1] = 1.0
    world[:, 0] = local[:, 0]
    for joint in range(1, 22):
        parent = SMPL22_PARENTS[joint]
        world[:, joint] = world[:, parent] @ local[:, joint]
        positions[:, joint] = positions[:, parent] + (world[:, parent] @ offsets[joint, :, None])[..., 0]
    features = np.zeros((frames, 263), dtype=np.float32)
    features[:, 3] = 1.0
    features[:, 4:67] = positions[:, 1:].reshape(frames, -1)
    result = convert.hml263_to_motion272(features, source_fps=20, target_fps=20)
    from motius.motion.representation.motion272 import motion272_to_joints

    np.testing.assert_allclose(
        motion272_to_joints(result, bone_offsets=offsets),
        recover_272_stored_positions(result),
        atol=2e-6,
    )


def test_native_bridge_does_not_use_incoming_bone_rotation_channels():
    features = real_motion()
    expected = convert.hml263_to_motion272(features, target_len=37)
    # Position channels, rather than incompatible redundant HML rotations,
    # determine the outgoing-frame pose. This also covers generated features
    # whose position and rotation predictions disagree.
    features[:, 67:193] = 0
    actual = convert.hml263_to_motion272(features, target_len=37)
    assert actual.shape == (37, 272)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("options", [{"bridge": "native"}, {"bridge": "smpl"}, {"ik_kwargs": {}}])
def test_removed_bridge_selectors_are_rejected(options):
    with pytest.raises(TypeError):
        convert.hml263_to_motion272(real_motion(), **options)


@pytest.mark.parametrize("mode", ["auto", "position_ik", "hml263", "hml263_init", "hml263_end_effectors"])
def test_removed_rotation_selector_is_rejected(mode):
    from motius.motion.retarget.hml263_smpl import retarget_hml263_clip

    with pytest.raises(TypeError):
        retarget_hml263_clip(real_motion(), rotation_init=mode)


def test_landmark_constraints_use_shared_position_solver():
    from motius.motion.retarget.position_ik import estimate_local_rotations

    offsets = _canonical_272_offsets()
    rest = np.zeros((22, 3))
    for joint in range(1, 22):
        rest[joint] = rest[SMPL22_PARENTS[joint]] + offsets[joint]
    desired = Rotation.from_euler("xyz", [0.2, 0.3, -0.4]).as_matrix()

    def guides(frame, joint, rest_vectors, target_vectors, weights):
        if joint == 15:
            for axis in np.eye(3):
                rest_vectors.append(axis)
                target_vectors.append(desired @ axis)
                weights.append(1.0)

    rotations = estimate_local_rotations(
        rest[None] + [0, 1, 0], rest, np.asarray(SMPL22_PARENTS),
        orientation_guides=guides,
    )
    np.testing.assert_allclose(rotations[0, 15], desired, atol=1e-6)
    others = np.delete(rotations[0], 15, axis=0)
    np.testing.assert_allclose(others, np.broadcast_to(np.eye(3), others.shape), atol=1e-6)


@pytest.mark.parametrize("target_len,expected_frames", [(None, 20), (13, 13)])
def test_shared_ik_timing_and_rotation_channel_independence(tmp_path, target_len, expected_frames):
    offsets = _canonical_272_offsets()
    rest = np.zeros((22, 3))
    for joint in range(1, 22):
        rest[joint] = rest[SMPL22_PARENTS[joint]] + offsets[joint]
    model_path = tmp_path / "model.npz"
    # Minimal body-model data exercises the actual SMPL-H skeleton adapter,
    # resampling, retarget API and 272 encoder without licensed mesh assets.
    np.savez(model_path, v_template=rest, J_regressor=np.eye(22),
             shapedirs=np.zeros((22, 3, 10)),
             kintree_table=np.stack([SMPL22_PARENTS, np.arange(22)]))
    features = real_motion()[:8]
    options = dict(source_fps=10, target_fps=25, target_len=target_len)
    result = convert.hml263_to_motion272(features, **options)
    assert result.shape == (expected_frames, 272)
    smpl = convert.hml263_to_motion135(features, model_dir=model_path, device="cpu", **options)
    assert smpl.shape == (expected_frames, 135)
    features[:, 67:193] = 0
    np.testing.assert_array_equal(
        convert.hml263_to_motion135(features, model_dir=model_path, device="cpu", **options), smpl,
    )
    np.testing.assert_array_equal(convert.hml263_to_motion272(features, **options), result)
