import numpy as np
import torch

from motius.models.condmdi.network.masks import build_observation_mask
from motius.models.condmdi.network.representation import (
    absolute_to_relative,
    relative_to_absolute,
)
from motius.pipelines.condmdi import CondMDIPipeline


class _RandomDiffusion:
    def p_sample_loop(self, _model, shape, noise, **_kwargs):
        return noise + torch.randn(shape, device=noise.device)


class _TestBundle:
    def __init__(self):
        self.device = torch.device("cpu")
        self.guidance_param = 1.0
        self.config = {"diffusion_steps": 1000}
        self.diffusion = _RandomDiffusion()
        self.net = torch.nn.Identity()

    def eval(self):
        return self

    def normalize_absolute(self, motion):
        return motion

    def denormalize_absolute(self, motion):
        return motion


def test_condmdi_absolute_root_round_trip():
    generator = torch.Generator().manual_seed(7)
    motion = torch.randn((2, 48, 263), generator=generator) * 0.01
    motion[..., 3] = 0.9
    recovered = absolute_to_relative(relative_to_absolute(motion))
    torch.testing.assert_close(recovered[..., :-1, :3], motion[..., :-1, :3], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(recovered[..., 3:], motion[..., 3:])


def test_condmdi_keyframe_mask_observes_requested_frames():
    mask = build_observation_mask(
        [48],
        48,
        keyframe_indices=[[0, 17, 47]],
        feature_mode="pos_rot_vel",
    )
    assert mask.shape == (1, 263, 1, 48)
    assert np.array_equal(torch.where(mask[0, :, 0].any(dim=0))[0].numpy(), [0, 17, 47])
    assert not mask[0, :, 0, 1].any()


def test_condmdi_joint_control_does_not_select_unrequested_joint_features():
    wrist_only = build_observation_mask(
        [40],
        40,
        mode="joints",
        joint_indices=[21],
        feature_mode="pos",
    )
    pelvis_only = build_observation_mask(
        [40],
        40,
        mode="joints",
        joint_indices=[0],
        feature_mode="pos",
    )
    assert wrist_only.any()
    assert pelvis_only.any()
    assert not torch.logical_and(wrist_only, pelvis_only).any()


def test_condmdi_seed_controls_the_complete_sampling_process():
    pipeline = CondMDIPipeline(_TestBundle())
    first = pipeline.infer_t2m(["walk"], [24], seed=19)[0]
    torch.manual_seed(999)
    second = pipeline.infer_t2m(["walk"], [24], seed=19)[0]
    third = pipeline.infer_t2m(["walk"], [24], seed=20)[0]
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, third)
