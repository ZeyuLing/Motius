import numpy as np
import pytest
import torch
from torch import nn
import json

from motius.models.edge.audio import (
    edge_audio_window_count,
    validate_edge_music_features,
)
from motius.models.edge.network.motion import (
    EDGE_REPR_DIM,
    EDGE_SMPL24_OFFSETS,
    edge_forward_kinematics,
    edge_motion_to_motion135,
    edge_zup_to_aistpp_yup,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from motius.models.edge.network.sampler import edge_ddim_sample, stitch_edge_windows
from motius.motion.representation.rotation import rotation_6d_to_matrix as motius_rot6d_to_matrix
from tools.infer_edge_aistpp import _load_cases


def test_edge_rotation_6d_uses_pytorch3d_row_convention():
    matrix = torch.tensor(
        [[[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    value = matrix_to_rotation_6d(matrix)

    assert value.tolist() == [[0.0, -1.0, 0.0, 1.0, 0.0, 0.0]]
    torch.testing.assert_close(rotation_6d_to_matrix(value), matrix)


def test_edge_identity_fk_matches_released_rest_offsets():
    rotations = torch.eye(3).reshape(1, 1, 1, 3, 3).expand(1, 1, 24, 3, 3)
    root = torch.tensor([[[1.0, 2.0, 3.0]]])
    joints = edge_forward_kinematics(rotations, root)

    torch.testing.assert_close(joints[0, 0, 0], root[0, 0])
    torch.testing.assert_close(
        joints[0, 0, 1], root[0, 0] + torch.tensor(EDGE_SMPL24_OFFSETS[1])
    )


def test_edge_coordinate_conversion_is_zup_to_yup():
    source = torch.tensor([[[1.0, 2.0, 3.0]]])
    expected = torch.tensor([[[1.0, 3.0, -2.0]]])
    torch.testing.assert_close(edge_zup_to_aistpp_yup(source), expected)


def test_edge_motion135_conversion_is_lossless_rigid_basis_change():
    generator = torch.Generator().manual_seed(19)
    matrices, _ = torch.linalg.qr(torch.randn((2, 7, 24, 3, 3), generator=generator))
    determinant = torch.det(matrices)
    matrices[..., :, 2] *= torch.where(determinant < 0, -1.0, 1.0)[..., None]
    root = torch.randn((2, 7, 3), generator=generator)
    edge = torch.zeros((2, 7, EDGE_REPR_DIM))
    edge[..., 4:7] = root
    edge[..., 7:] = matrix_to_rotation_6d(matrices).reshape(2, 7, -1)

    expected = edge_zup_to_aistpp_yup(edge_forward_kinematics(matrices, root))
    motion135 = edge_motion_to_motion135(edge)
    converted_rotations = motius_rot6d_to_matrix(
        motion135[..., 3:].reshape(2, 7, 22, 6), convention="row"
    )
    converted_joints = edge_forward_kinematics(
        torch.cat((converted_rotations, matrices[..., 22:, :, :]), dim=2),
        motion135[..., :3],
    )

    torch.testing.assert_close(converted_joints, expected, atol=2e-5, rtol=2e-5)


def test_edge_feature_contract():
    value = validate_edge_music_features(np.zeros((150, 4800), dtype=np.float32))
    assert value.shape == (1, 150, 4800)
    with pytest.raises(ValueError, match="4800"):
        validate_edge_music_features(np.zeros((150, 438), dtype=np.float32))


@pytest.mark.parametrize(
    ("duration", "windows"),
    [(5.0, 1), (7.1, 2), (7.5, 2), (9.6, 3), (12.0, 4)],
)
def test_edge_audio_windows_cover_full_clip(duration, windows):
    assert edge_audio_window_count(duration) == windows
    covered_seconds = 5.0 + (windows - 1) * 2.5
    assert covered_seconds + 1e-6 >= duration


def test_edge_case_manifest_accepts_public_gallery_descriptors(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "gBR_test_mBR0_ch01",
                        "audio": "audio/mBR0.mp3",
                        "audio_end_seconds": 12.0,
                        "motions": {"gt": {"display_frames": 360, "fps": 30.0}},
                    }
                ]
            }
        )
    )

    records = _load_cases(path)

    assert records[0]["case_id"] == "gBR_test_mBR0_ch01"
    assert records[0]["music_id"] == "mBR0"
    assert records[0]["frames"] == 360


def _identity_edge_windows(count: int) -> torch.Tensor:
    windows = torch.zeros((count, 150, EDGE_REPR_DIM))
    identity = matrix_to_rotation_6d(torch.eye(3)).repeat(24)
    windows[:, :, 7:] = identity
    return windows


def test_edge_stitching_uses_75_frame_overlap():
    windows = _identity_edge_windows(2)
    windows[0, :, 4] = 1.0
    windows[1, :, 4] = 3.0
    stitched = stitch_edge_windows(windows)

    assert stitched.shape == (225, EDGE_REPR_DIM)
    assert stitched[0, 4] == pytest.approx(1.0)
    assert stitched[-1, 4] == pytest.approx(3.0)
    assert torch.isfinite(stitched).all()


class _ZeroPrediction(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def guided_forward(self, sample, condition, timestep, weight):
        del condition, timestep, weight
        return sample * 0 + self.anchor


def test_edge_sampler_shape_and_seed_are_deterministic():
    model = _ZeroPrediction()
    condition = torch.zeros((1, 150, 4800))
    first = edge_ddim_sample(
        model,
        condition,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(7),
    )
    second = edge_ddim_sample(
        model,
        condition,
        sampling_steps=2,
        generator=torch.Generator().manual_seed(7),
    )

    assert first.shape == (1, 150, EDGE_REPR_DIM)
    torch.testing.assert_close(first, second)
