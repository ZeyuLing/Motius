import numpy as np
import pytest

from motius.motion.representation.monocular_joints import select_common_hmr15
from tools.run_3dpw_hymotion_v2m_shard import _convert


def test_hymotion_v2m_conversion_is_named_joint_only_and_unranked():
    world = np.zeros((1, 5, 22, 3), dtype=np.float32)
    world[..., 0] = 1.0
    world[..., 1] = 2.0
    world[..., 2] = 3.0
    output = {
        "keypoints3d": world,
        "bbox_xyxy": np.tile([10.0, 20.0, 30.0, 50.0], (5, 1)),
        "shapes": np.zeros((1, 1, 16), dtype=np.float32),
        "camera_K": np.tile(np.eye(3, dtype=np.float32), (5, 1, 1)),
    }

    result = _convert(
        output,
        sequence="synthetic",
        checkpoint_sha256="a" * 64,
        shard_id=0,
        num_shards=8,
    )
    track = result.tracks[0]

    np.testing.assert_allclose(track.joints_world[..., 1], 2.0)
    np.testing.assert_allclose(track.joints_camera[..., 1], -2.0)
    assert track.body_model == "SMPL-H neutral"
    assert select_common_hmr15(
        track.joints_camera,
        track.joint_names,
        body_model=track.body_model,
    ).shape == (5, 15, 3)
    assert result.metadata["ranking_eligible"] is False
    assert result.metadata["camera_motion"] == "static_identity_fallback"
    assert "single-person" not in result.metadata["ranking_exclusion"]
    assert track.metadata["crop_protocol"] == "caller_supplied_dense_target_bbox"


def test_hymotion_v2m_conversion_rejects_nonfinite_suffix():
    world = np.zeros((1, 5, 22, 3), dtype=np.float32)
    world[:, 3:] = np.nan
    output = {
        "keypoints3d": world,
        "rot6d": np.zeros((1, 5, 22, 6), dtype=np.float32),
        "transl": np.zeros((1, 5, 3), dtype=np.float32),
        "bbox_xyxy": np.tile([10.0, 20.0, 30.0, 50.0], (5, 1)),
        "shapes": np.zeros((1, 1, 16), dtype=np.float32),
        "camera_K": np.tile(np.eye(3, dtype=np.float32), (5, 1, 1)),
    }

    with pytest.raises(ValueError, match="first_bad_frame=3"):
        _convert(
            output,
            sequence="synthetic",
            checkpoint_sha256="a" * 64,
            shard_id=0,
            num_shards=8,
        )
