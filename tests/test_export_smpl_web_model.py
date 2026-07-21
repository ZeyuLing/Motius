import numpy as np
import pytest

from tools.export_smpl_web_model import low_rank_pose_correctives


def test_low_rank_pose_correctives_reconstruct_requested_rank():
    rng = np.random.default_rng(7)
    left = rng.normal(size=(18, 3)).astype(np.float32)
    right = rng.normal(size=(3, 9)).astype(np.float32)
    posedirs = left @ right

    basis, projection, energy = low_rank_pose_correctives(posedirs, rank=3)

    assert basis.shape == (18, 3)
    assert projection.shape == (3, 9)
    np.testing.assert_allclose(basis @ projection, posedirs, atol=2e-5)
    assert energy == pytest.approx(1.0, abs=1e-5)
