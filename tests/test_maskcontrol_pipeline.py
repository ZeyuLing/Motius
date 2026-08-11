from types import SimpleNamespace

import numpy as np
import pytest
import torch

from motius.models.maskcontrol.network import (
    BODY_PART_JOINTS,
    CONTROL_JOINT_IDS,
    relative_hml263_positions,
)
from motius.models.momask.network.vq.model import RVQVAE
from motius.pipelines.maskcontrol import MaskControlPipeline


class _DummyBundle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(()))
        self.length_estimator = None
        self.control_model = object()
        self.calls = []

    @property
    def device(self):
        return self.anchor.device

    def to_device(self, device):
        self.to(device)
        return self

    def generate(self, captions, frame_lengths, targets, target_mask, **kwargs):
        self.calls.append(
            {
                "captions": captions,
                "lengths": frame_lengths.detach().cpu().tolist(),
                "targets": targets.detach().cpu().clone(),
                "mask": target_mask.detach().cpu().clone(),
                "kwargs": dict(kwargs),
            }
        )
        batch, frames = targets.shape[:2]
        value = torch.zeros((batch, frames, 263), device=targets.device)
        return value, value


@pytest.fixture()
def pipeline():
    return MaskControlPipeline(_DummyBundle())


def test_t2m_rounds_to_checkpoint_length_and_disables_control(pipeline):
    result = pipeline.infer_t2m(["walk"], [41], seed=7)
    assert result[0].shape == (44, 263)
    call = pipeline.bundle.calls[-1]
    assert call["lengths"] == [44]
    assert call["kwargs"]["use_control"] is False
    assert not call["mask"].any()


def test_explicit_zero_control_is_preserved(pipeline):
    targets = np.zeros((1, 80, 22, 3), dtype=np.float32)
    mask = np.zeros((1, 80, 22), dtype=bool)
    mask[0, 0, 20] = True
    pipeline.infer_control(
        ["wave"],
        [80],
        targets,
        mask,
        each_iterations=0,
        final_iterations=0,
    )
    call = pipeline.bundle.calls[-1]
    assert call["mask"][0, 0, 20]
    assert torch.equal(call["targets"][0, 0, 20], torch.zeros(3))


def test_unsupported_control_joint_is_rejected(pipeline):
    targets = np.zeros((1, 80, 22, 3), dtype=np.float32)
    mask = np.zeros((1, 80, 22), dtype=bool)
    mask[0, 4, 3] = True
    with pytest.raises(ValueError, match="only support joints"):
        pipeline.infer_control(["walk"], [80], targets, mask)


def test_temporal_prefix_uses_all_released_anchor_joints(pipeline):
    motion = np.zeros((80, 263), dtype=np.float32)
    pipeline.infer_temporal(
        ["walk"],
        [motion],
        mode="prefix",
        prefix_ratio=0.25,
        each_iterations=0,
        final_iterations=0,
    )
    mask = pipeline.bundle.calls[-1]["mask"][0]
    assert mask[:20, list(CONTROL_JOINT_IDS)].all()
    assert not mask[20:].any()
    unsupported = [value for value in range(22) if value not in CONTROL_JOINT_IDS]
    assert not mask[:, unsupported].any()


def test_temporal_blank_mode_uses_true_unconditional_text(pipeline):
    motion = np.zeros((80, 263), dtype=np.float32)
    pipeline.infer_temporal(
        None,
        [motion],
        mode="first_frame",
        each_iterations=0,
        final_iterations=0,
    )
    assert pipeline.bundle.calls[-1]["captions"] is None


def test_body_part_and_sequential_paths_share_control_contract(pipeline):
    body = pipeline.infer_body_part(
        [
            ("lower", "jump forward", (0, 80)),
            ("upper", "raise both arms", (0, 80)),
        ],
        length=80,
        each_iterations=0,
        final_iterations=0,
    )
    assert body.shape == (80, 263)
    first_body_call = pipeline.bundle.calls[-2]
    assert first_body_call["lengths"] == [80]
    assert first_body_call["kwargs"]["use_control"] is False
    assert first_body_call["kwargs"]["use_residual"] is False
    body_call = pipeline.bundle.calls[-1]
    assert body_call["mask"][:, :80, [0, 10, 11]].all()
    assert body_call["kwargs"]["use_residual"] is False

    sequence = pipeline.infer_sequential(
        ["walk", "sit"],
        [40, 40],
        transition_padding=5,
        each_iterations=0,
        final_iterations=0,
    )
    assert sequence.shape == (80, 263)
    sequence_call = pipeline.bundle.calls[-1]
    assert sequence_call["captions"] is None
    assert sequence_call["kwargs"]["relative_control"] is True
    assert not sequence_call["mask"][0, 35:45].any()


def test_body_part_timeline_preserves_official_elbow_and_knee_groups(pipeline):
    pipeline.infer_body_part(
        [
            ("left_arm", "wave with the left hand", (0, 40)),
            ("legs", "walk forward", (40, 80)),
        ],
        length=80,
        each_iterations=0,
        final_iterations=0,
    )

    first, edited = pipeline.bundle.calls[-2:]
    assert first["lengths"] == [40]
    assert first["kwargs"]["use_residual"] is False
    assert edited["mask"][0, :40, [18, 20]].all()
    assert not edited["mask"][0, :40, [4, 5, 10, 11]].any()
    assert BODY_PART_JOINTS["right_arm"] == (21, 19)
    assert BODY_PART_JOINTS["legs"] == (10, 4, 11, 5)


def test_relative_proxy_has_expected_shape_and_root_gradient_channel():
    motion = torch.zeros((2, 12, 263))
    motion[..., 0] = 2.0
    relative = relative_hml263_positions(motion)
    assert relative.shape == (2, 12, 22, 3)
    assert torch.equal(relative[..., 0, :], torch.full((2, 12, 3), 2.0))


def test_rvq_decoder_accepts_indices_and_continuous_expectations():
    args = SimpleNamespace(
        num_quantizers=2,
        shared_codebook=False,
        quantize_dropout_prob=0.0,
        mu=0.99,
    )
    model = RVQVAE(
        args,
        input_width=4,
        nb_code=8,
        code_dim=8,
        output_emb_width=8,
        down_t=1,
        stride_t=2,
        width=8,
        depth=1,
        dilation_growth_rate=2,
        activation="relu",
        norm=None,
    ).eval()
    indices = torch.randint(0, 8, (1, 5, 2))
    codes = model.quantizer.get_codes_from_indices(indices).sum(dim=0)
    with torch.no_grad():
        from_indices = model.forward_decoder(indices)
        from_embeddings = model.forward_decoder(codes)
    torch.testing.assert_close(from_indices, from_embeddings, rtol=0, atol=0)
