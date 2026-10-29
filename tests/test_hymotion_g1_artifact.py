"""HYMotion-G1 artifact and native decoding contracts."""

from __future__ import annotations

import torch
import pytest
from safetensors.torch import save_file

from motius.models.hymotion_t2m import HyMotionT2MBundle
from motius.motion.representation.hymotion_runtime.codecs import HyMotionCodec


def _minimal_bundle() -> HyMotionT2MBundle:
    bundle = HyMotionT2MBundle.__new__(HyMotionT2MBundle)
    torch.nn.Module.__init__(bundle)
    bundle.motion_transformer = torch.nn.Linear(3, 2)
    bundle.null_vtxt_feat = torch.nn.Parameter(torch.zeros(1, 1, 4))
    bundle.null_ctxt_input = torch.nn.Parameter(torch.zeros(1, 1, 5))
    bundle.special_game_vtxt_feat = torch.nn.Parameter(torch.zeros(1, 1, 4))
    bundle.special_game_ctxt_feat = torch.nn.Parameter(torch.zeros(1, 1, 5))
    bundle.enable_special_game_feat = False
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

    # G1 artifacts disable these conditions. A human artifact that enables
    # them must never silently replace omitted pretrained values with zeros.
    bundle.enable_special_game_feat = True
    with pytest.raises(ValueError, match="missing special_game_vtxt_feat"):
        bundle._load_artifact_weights(str(transformer_path), str(bundle_path))


def test_g1_decode_returns_exact_denormalized_motion_and_qpos():
    bundle = _minimal_bundle()
    bundle.motion_type = "g1_29dof"
    bundle.register_buffer("mean", torch.linspace(-0.1, 0.1, 38))
    bundle.register_buffer("std", torch.linspace(0.2, 0.5, 38))
    latent = torch.zeros(2, 4, 38)

    result = HyMotionCodec(bundle).decode_motion_from_latent(
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


def test_packaged_wooden_geometry_loads_and_deforms():
    from motius.motion.body_models.hymotion_wooden import HYMotionWoodenBody
    body = HYMotionWoodenBody()
    rot6d = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(1, 22, 1)
    vertices, joints = body(rot6d, torch.zeros(1, 3))
    assert vertices.shape == (1, body.v_template.shape[0], 3)
    assert joints.shape == (1, body.j_template.shape[0], 3)
    assert torch.isfinite(vertices).all()
    assert torch.isfinite(joints).all()
