from __future__ import annotations

import torch

from motius.models.unimumo import UniMuMoBundle, UniMuMoGenerator
from motius.models.unimumo.generator import DelayedPattern, generate_parallel
from motius.models.unimumo.motion_codec import UniMuMoMotionCodec
from motius.pipelines.unimumo import UniMuMoPipeline
from motius.registry import MODEL_BUNDLES, PIPELINES


def _generator_config():
    return {
        "num_codebooks": 2,
        "codebook_size": 8,
        "dimension": 8,
        "hidden_dimension": 16,
        "num_heads": 2,
        "num_layers": 1,
        "dropout": 0.0,
        "attention_dropout": 0.0,
        "bias_attention": False,
        "bias_ffn": False,
        "output_bias": False,
        "norm_first": True,
    }


def test_unimumo_registration():
    assert MODEL_BUNDLES.get("UniMuMoBundle") is UniMuMoBundle
    assert PIPELINES.get("UniMuMoPipeline") is UniMuMoPipeline


def test_unimumo_motion_codec_preserves_duration_contract():
    codec = UniMuMoMotionCodec(
        {
            "motion_dim": 263,
            "latent_dim": 8,
            "encoder_channels": [16, 8],
            "decoder_channels": [8, 16],
            "motion_fps": 60.0,
            "code_fps": 50.0,
            "depth_per_block": 1,
            "pre_quant_multiplier": 2,
            "post_quant_multiplier": 2,
        }
    )
    motion = torch.randn(2, 12, 263)
    music_embeddings = torch.randn(2, 8, 10)
    motion_embeddings = codec.encode_embeddings(motion, music_embeddings)
    assert motion_embeddings.shape == (2, 8, 10)
    reconstructed = codec.decode_embeddings(music_embeddings, motion_embeddings)
    assert reconstructed.shape == (2, 12, 263)


def test_unimumo_delayed_pattern_matches_valid_and_generation_layouts():
    codes = torch.tensor([[[0, 1, 2], [3, 4, 5]]])
    pattern = DelayedPattern(3, (0, 1))
    full, full_mask = pattern.build(codes, special_token=8)
    assert full.tolist() == [[[8, 0, 1, 2, 8], [8, 8, 3, 4, 5]]]
    assert full_mask.tolist() == [
        [False, True, True, True, False],
        [False, False, True, True, True],
    ]
    torch.testing.assert_close(pattern.revert(full), codes)

    valid, valid_mask = pattern.build(codes, special_token=8, valid_only=True)
    assert valid.tolist() == [[[8, 0, 1, 2], [8, 8, 3, 4]]]
    assert valid_mask.shape == (2, 4)


def test_unimumo_generator_forward_and_cfg_generation():
    torch.manual_seed(4)
    model = UniMuMoGenerator(_generator_config()).eval()
    music = torch.tensor([[[8, 1, 2], [8, 8, 3]]])
    motion = torch.tensor([[[8, 4, 5], [8, 8, 6]]])
    condition = torch.randn(1, 4, 8)
    condition_mask = torch.tensor([[[1, 1, 0, 0], [0, 0, 1, 1]]]).bool()
    music_logits, motion_logits = model(
        music,
        motion,
        condition=condition,
        condition_mask=condition_mask,
    )
    assert music_logits.shape == (1, 2, 3, 8)
    assert motion_logits.shape == (1, 2, 3, 8)

    generated_music, generated_motion = generate_parallel(
        model,
        condition=condition,
        condition_mask=condition_mask,
        unconditional_condition=torch.zeros_like(condition),
        unconditional_mask=condition_mask,
        timesteps=3,
        guidance_scale=1.0,
        temperature=0.0,
    )
    assert generated_music.shape == (1, 2, 3)
    assert generated_motion.shape == (1, 2, 3)
    assert generated_music.min() >= 0 and generated_music.max() < 8
    assert generated_motion.min() >= 0 and generated_motion.max() < 8


def test_unimumo_description_modes():
    descriptions = ["piano solo <separation> a person spins"]
    assert UniMuMoBundle._split_descriptions(descriptions, "music_motion") == (
        ["piano solo"],
        ["a person spins"],
    )
    assert UniMuMoBundle._split_descriptions(descriptions, "music2motion") == (
        [""],
        ["a person spins"],
    )
    assert UniMuMoBundle._split_descriptions(descriptions, "motion2music") == (
        ["piano solo"],
        [""],
    )
