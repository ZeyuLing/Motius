from __future__ import annotations

from types import SimpleNamespace

import torch

from motius.models.unimumo import UniMuMoBundle, UniMuMoGenerator
from motius.models.unimumo.generator import DelayedPattern, generate_parallel
from motius.models.unimumo.motion_codec import UniMuMoMotionCodec
from motius.pipelines.unimumo import UniMuMoPipeline
from motius.registry import MODEL_BUNDLES, PIPELINES
from tools.infer_unimumo_aistpp import smpl22_to_smpl24
from tools.run_m2t_humanml3d import _unimumo_caption_batch


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


def test_unimumo_cfg_conditions_share_padding_layout():
    class FakeBundle:
        device = torch.device("cpu")
        _split_descriptions = staticmethod(UniMuMoBundle._split_descriptions)
        _condition_from_streams = UniMuMoBundle._condition_from_streams
        cfg_text_condition = UniMuMoBundle.cfg_text_condition

        def _encode_descriptions(self, descriptions):
            lengths = [max(1, len(value.split())) for value in descriptions]
            width = max(lengths)
            hidden = torch.zeros(len(descriptions), width, 3)
            mask = torch.zeros(len(descriptions), width, dtype=torch.bool)
            for index, length in enumerate(lengths):
                mask[index, :length] = True
            return hidden, mask

    condition, condition_mask, null, null_mask = FakeBundle().cfg_text_condition(
        ["long orchestral score <separation> a person spins quickly"]
    )
    assert condition.shape == null.shape
    assert condition_mask.shape == null_mask.shape
    assert condition.shape[1] == 7


def test_unimumo_pipeline_resamples_motion_to_native_fps():
    pipeline = UniMuMoPipeline.__new__(UniMuMoPipeline)
    pipeline.bundle = SimpleNamespace(motion_fps=60.0)
    motion = torch.arange(12 * 263, dtype=torch.float32).reshape(12, 263)

    native = pipeline._prepare_motion(motion, input_fps=60)
    upsampled = pipeline._prepare_motion(motion, input_fps=20)
    batched = pipeline._prepare_motion(motion[None], input_fps=20)

    assert native.shape == (12, 263)
    assert upsampled.shape == (36, 263)
    assert batched.shape == (1, 36, 263)
    torch.testing.assert_close(upsampled, batched[0])


def test_unimumo_aistpp_bridge_preserves_body_and_extends_hands():
    joints = torch.zeros(3, 22, 3).numpy()
    joints[:, 18, 0] = 1
    joints[:, 19, 0] = -1
    joints[:, 20, 0] = 2
    joints[:, 21, 0] = -2
    bridged = smpl22_to_smpl24(joints)
    assert bridged.shape == (3, 24, 3)
    torch.testing.assert_close(torch.from_numpy(bridged[:, :22]), torch.from_numpy(joints))
    torch.testing.assert_close(torch.from_numpy(bridged[:, 22, 0]), torch.full((3,), 2.35))
    torch.testing.assert_close(torch.from_numpy(bridged[:, 23, 0]), torch.full((3,), -2.35))


def test_unimumo_caption_batch_matches_official_padding_and_cleanup():
    class FakePipeline:
        def infer_motion_to_text(self, motion, *, input_fps):
            assert motion.shape == (2, 200, 263)
            assert input_fps == 20.0
            assert torch.count_nonzero(torch.from_numpy(motion[0, 2:])) == 0
            return SimpleNamespace(
                captions=(
                    "The motion is that a person spins",
                    "The dance is that someone jumps",
                )
            )

    captions = _unimumo_caption_batch(
        FakePipeline(),
        [torch.ones(2, 263).numpy(), torch.ones(4, 263).numpy()],
    )
    assert captions == ("A person spins", "Someone jumps")
