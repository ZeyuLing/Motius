"""HYMotion-G1 artifact and native decoding contracts."""

from __future__ import annotations

import torch
from safetensors.torch import save_file

from motius.models.hymotion_t2m import HyMotionT2MBundle


def _minimal_bundle() -> HyMotionT2MBundle:
    bundle = HyMotionT2MBundle.__new__(HyMotionT2MBundle)
    torch.nn.Module.__init__(bundle)
    bundle.motion_transformer = torch.nn.Linear(3, 2)
    bundle.null_vtxt_feat = torch.nn.Parameter(torch.zeros(1, 1, 4))
    bundle.null_ctxt_input = torch.nn.Parameter(torch.zeros(1, 1, 5))
    bundle.special_game_vtxt_feat = torch.nn.Parameter(torch.zeros(1, 1, 4))
    bundle.special_game_ctxt_feat = torch.nn.Parameter(torch.zeros(1, 1, 5))
    return bundle


def test_split_artifact_loads_unprefixed_transformer_and_bundle_params(tmp_path):
    source = torch.nn.Linear(3, 2)
    transformer_path = tmp_path / "motion_transformer.safetensors"
    bundle_path = tmp_path / "bundle_params.safetensors"
    save_file(source.state_dict(), str(transformer_path))
    save_file(
        {
            "null_vtxt_feat": torch.full((1, 1, 4), 2.0),
            "null_ctxt_input": torch.full((1, 1, 5), 3.0),
        },
        str(bundle_path),
    )

    bundle = _minimal_bundle()
    bundle._load_artifact_weights(
        str(transformer_path),
        bundle_weights_path=str(bundle_path),
    )

    for actual, expected in zip(
        bundle.motion_transformer.parameters(), source.parameters()
    ):
        assert torch.equal(actual, expected)
    assert torch.equal(bundle.null_vtxt_feat, torch.full((1, 1, 4), 2.0))
    assert torch.equal(bundle.null_ctxt_input, torch.full((1, 1, 5), 3.0))


def test_g1_decode_returns_exact_denormalized_motion_and_qpos():
    bundle = _minimal_bundle()
    bundle.motion_type = "g1_29dof"
    bundle.register_buffer("mean", torch.linspace(-0.1, 0.1, 38))
    bundle.register_buffer("std", torch.linspace(0.2, 0.5, 38))
    latent = torch.zeros(2, 4, 38)

    result = bundle.decode_motion_from_latent(
        latent,
        should_apply_smoothing=True,
    )

    expected = bundle.mean.expand_as(latent)
    assert result["representation"] == "g1_38"
    assert result["fps"] == 30.0
    assert torch.equal(result["motion"], expected)
    assert torch.equal(result["g1_38"], expected)
    assert result["g1_qpos"].shape == (2, 4, 36)
    assert torch.equal(result["qpos"], result["g1_qpos"])
    assert torch.equal(result["transl"], result["g1_qpos"][..., :3])
    assert result["keypoints3d"] is None
