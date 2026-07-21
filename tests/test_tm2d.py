from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import torch

from motius.models.tm2d import TM2DBundle, TM2DTokenizer
from motius.models.tm2d.bundle import DEFAULT_TM2D_CONFIG
from motius.pipelines.tm2d import TM2DPipeline
from motius.registry import MODEL_BUNDLES, PIPELINES


def _tiny_config():
    config = copy.deepcopy(DEFAULT_TM2D_CONFIG)
    config.update(
        {
            "codebook_size": 8,
            "latent_dim": 16,
            "motion_vocabulary_size": 11,
            "motion_start_id": 8,
            "motion_end_id": 9,
            "motion_pad_id": 10,
            "text_vocabulary_size": 8,
            "text_length_id": 6,
            "text_pad_id": 7,
            "max_text_tokens": 4,
            "default_music_seed_token": 1,
            "d_model": 16,
            "d_inner": 32,
            "n_encoder_layers": 1,
            "n_decoder_layers": 1,
            "n_head": 2,
            "d_k": 8,
            "d_v": 8,
            "dropout": 0.0,
            "max_source_length": 20,
            "max_target_length": 20,
            "audio_chunk_length": 5,
            "audio_chunk_overlap": 1,
        }
    )
    return config


def _vocabulary():
    return {"sos": 0, "eos": 1, "unk": 2, "walk": 3, "forward": 4}


def test_tm2d_registration_and_representation_contract():
    assert MODEL_BUNDLES.get("TM2DBundle") is TM2DBundle
    assert PIPELINES.get("TM2DPipeline") is TM2DPipeline
    assert DEFAULT_TM2D_CONFIG["vq_encoder_dim"] == 283
    assert DEFAULT_TM2D_CONFIG["normalized_motion_dim"] == 287
    assert DEFAULT_TM2D_CONFIG["code_stride"] == 8


def test_tm2d_tokenizer_encodes_duration_indicator():
    tokenizer = TM2DTokenizer(
        _vocabulary(), max_text_tokens=4, length_id=6, pad_id=7
    )
    encoded = tokenizer.encode(
        "unused", 5, pretokenized=["walk", "forward"]
    )
    np.testing.assert_array_equal(encoded, [[0, 3, 4, 1, 6, 7]])

    empty = tokenizer.encode("unused", 4, pretokenized=[])
    np.testing.assert_array_equal(empty, [[0, 1, 6, 6, 7, 7]])


def test_tm2d_tokenizer_validates_pretokenized_batch_size():
    tokenizer = TM2DTokenizer(
        _vocabulary(), max_text_tokens=4, length_id=6, pad_id=7
    )
    with np.testing.assert_raises_regex(ValueError, "equal length"):
        tokenizer.encode(
            ["first", "second"],
            [3, 3],
            pretokenized=[["walk"]],
        )


def test_tm2d_text_and_music_pipeline_run_on_cpu():
    torch.manual_seed(3)
    bundle = TM2DBundle(_tiny_config(), vocabulary=_vocabulary())
    pipeline = TM2DPipeline(bundle, device="cpu")

    text = pipeline.infer_text_to_motion(
        "unused",
        num_frames=24,
        output_fps=30.0,
        pretokenized=["walk", "forward"],
        sample=False,
    )
    assert text.joints.shape == (24, 24, 3)
    assert text.native_joints.shape == (48, 24, 3)
    assert text.model_motion.shape == (48, 287)
    assert text.motion_tokens.shape == (6,)

    features = np.random.default_rng(4).normal(size=(8, 438)).astype(np.float32)
    dance = pipeline.infer_music_to_dance(
        music_features=features,
        sample=False,
    )
    assert dance.joints.shape == (64, 24, 3)
    assert dance.model_motion.shape == (64, 287)
    assert dance.motion_tokens.shape == (8,)
    assert dance.music_features.shape == (8, 438)


def test_tm2d_vq_encoder_uses_283_channels():
    bundle = TM2DBundle(_tiny_config(), vocabulary=_vocabulary())
    motion = torch.randn(2, 24, 287)
    tokens = bundle.encode_motion(motion)
    assert tokens.shape == (2, 3)
    decoded = bundle.decode_tokens(tokens)
    assert decoded.shape == (2, 24, 287)


def test_tm2d_artifact_round_trip(tmp_path: Path):
    torch.manual_seed(7)
    bundle = TM2DBundle(_tiny_config(), vocabulary=_vocabulary())
    artifact = tmp_path / "tm2d"
    bundle.save_pretrained(artifact)
    restored = TM2DBundle.from_pretrained(artifact, local_files_only=True)
    assert restored.config == bundle.config
    assert restored.tokenizer.vocabulary == bundle.tokenizer.vocabulary
    for key, value in bundle.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)

    pipeline = TM2DPipeline.from_pretrained(
        artifact,
        bundle_kwargs={"local_files_only": True},
        device="cpu",
    )
    assert isinstance(pipeline.bundle, TM2DBundle)
