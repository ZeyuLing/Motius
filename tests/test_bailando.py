from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from motius.datasets.aistpp_music_to_dance import (
    AISTPPMusicDanceDataset,
    aistpp_music_id,
)
from motius.evaluation.music_to_dance import (
    AISTPPMusicDanceEvaluator,
    beat_alignment_score,
    motion_beat_frames,
)
from motius.models.bailando.audio import extract_bailando_audio_features
from motius.motion import aistpp_smpl24_fk, convert_motion, get_spec
from motius.motion.representation.aistpp import AISTPP_SMPL24_PARENTS
from motius.motion.representation.humanml import linear_resample_joints
from motius.models.bailando.bundle import BailandoBundle, DEFAULT_BAILANDO_CONFIG
from motius.pipelines.bailando import BailandoPipeline
from motius.registry import EVALUATORS, MODEL_BUNDLES, PIPELINES
from tools.smpl_gallery_assets import (
    encode_joint_positions,
    resample_joint_positions,
)


def _tiny_half():
    return {
        "levels": 1,
        "downs_t": [1],
        "strides_t": [2],
        "emb_width": 8,
        "l_bins": 8,
        "l_mu": 0.99,
        "commit": 0.02,
        "hvqvae_multipliers": [1],
        "width": 8,
        "depth": 1,
        "m_conv": 1.0,
        "dilation_growth_rate": 1,
        "sample_length": 8,
        "use_bottleneck": True,
        "joint_channel": 3,
        "vqvae_reverse_decoder_dilation": True,
    }


def _tiny_config():
    half = _tiny_half()
    return {
        "fps": 60.0,
        "code_downsample": 2,
        "motion_representation": "aistpp_smpl24_joints",
        "default_initial_codes": [1, 2],
        "vqvae": {
            "up_half": copy.deepcopy(half),
            "down_half": {**copy.deepcopy(half), "acc": 1.0},
            "use_bottleneck": True,
            "joint_channel": 3,
        },
        "gpt": {
            "block_size": 4,
            "base": {
                "embd_pdrop": 0.0,
                "resid_pdrop": 0.0,
                "attn_pdrop": 0.0,
                "vocab_size_up": 8,
                "vocab_size_down": 8,
                "block_size": 4,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 8,
                "n_music": 438,
                "n_music_emb": 8,
            },
            "head": {
                "embd_pdrop": 0.0,
                "resid_pdrop": 0.0,
                "attn_pdrop": 0.0,
                "vocab_size": 8,
                "block_size": 4,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 8,
                "vocab_size_up": 8,
                "vocab_size_down": 8,
            },
            "critic_net": {
                "embd_pdrop": 0.0,
                "resid_pdrop": 0.0,
                "attn_pdrop": 0.0,
                "block_size": 4,
                "n_layer": 1,
                "n_head": 2,
                "n_embd": 8,
                "vocab_size_up": 1,
                "vocab_size_down": 1,
            },
            "n_music": 438,
            "n_music_emb": 8,
        },
    }


def test_bailando_registration_and_default_contract():
    assert MODEL_BUNDLES.get("BailandoBundle") is BailandoBundle
    assert PIPELINES.get("BailandoPipeline") is BailandoPipeline
    assert EVALUATORS.get("AISTPPMusicDanceEvaluator") is AISTPPMusicDanceEvaluator
    assert DEFAULT_BAILANDO_CONFIG["motion_representation"] == "aistpp_smpl24_joints"
    assert DEFAULT_BAILANDO_CONFIG["gpt"]["base"]["n_music"] == 438


def test_bailando_pipeline_runs_on_cpu_with_precomputed_features():
    torch.manual_seed(4)
    bundle = BailandoBundle(config=_tiny_config())
    pipeline = BailandoPipeline(bundle, device="cpu")
    features = np.random.default_rng(4).normal(size=(3, 438)).astype(np.float32)
    output = pipeline(music_features=features)
    assert output.joints.shape == (1, 6, 24, 3)
    assert output.model_motion.shape == (1, 6, 72)
    assert output.codes_up.shape == (1, 3)
    np.testing.assert_allclose(output.joints[:, 0, 0], 0.0, atol=1e-7)


def test_bailando_bundle_artifact_round_trip(tmp_path: Path):
    torch.manual_seed(5)
    bundle = BailandoBundle(config=_tiny_config())
    artifact = tmp_path / "bailando"
    bundle.save_pretrained(artifact)
    restored = BailandoBundle.from_pretrained(artifact, local_files_only=True)
    assert restored.config == bundle.config
    for key, value in bundle.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)

    pipeline = BailandoPipeline.from_pretrained(
        artifact,
        bundle_kwargs={"local_files_only": True},
        device="cpu",
    )
    assert isinstance(pipeline.bundle, BailandoBundle)


def _dance(seed: int, phase: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.2, size=(24, 3)).astype(np.float32)
    base[:, 1] += np.linspace(0.0, 1.5, 24)
    time = np.linspace(0.0, 4.0 * np.pi, 90, dtype=np.float32)
    joints = np.repeat(base[None], len(time), axis=0)
    joints[:, :, 0] += 0.03 * np.sin(time[:, None] + phase)
    joints[:, :, 2] += 0.02 * np.cos(time[:, None] * 0.7 + phase)
    joints[:, 0, 0] += np.linspace(0.0, 0.8, len(time))
    return joints


def test_music_to_dance_gt_identity_metrics_are_consistent():
    evaluator = AISTPPMusicDanceEvaluator(physical=True)
    for index in range(3):
        motion = _dance(index + 1, index * 0.2)
        beats = np.zeros(90, dtype=bool)
        beats[10::15] = True
        evaluator.process(
            {
                "pred_joints": motion,
                "gt_joints": motion.copy(),
                "music_beats": beats,
                "music_fps": 60.0,
                "motion_fps": 60.0,
            }
        )
    metrics = evaluator.compute()
    assert metrics["FID_k"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["FID_g"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["Diversity_k"] == pytest.approx(metrics["GT_Diversity_k"])
    assert metrics["Diversity_g"] == pytest.approx(metrics["GT_Diversity_g"])
    assert metrics["num_samples"] == 3
    assert "Physical/Slide" in metrics


def test_music_to_dance_evaluator_resamples_30fps_predictions_for_official_features():
    evaluator = AISTPPMusicDanceEvaluator(physical=False)
    for index in range(3):
        motion_30 = _dance(index + 50, index * 0.2)[::3]
        motion_60 = linear_resample_joints(motion_30, 30.0, 60.0)
        evaluator.process(
            {
                "pred_joints": motion_30,
                "gt_joints": motion_60,
                "music_beats": np.zeros(len(motion_60), dtype=bool),
                "pred_motion_fps": 30.0,
                "gt_motion_fps": 60.0,
            }
        )
    metrics = evaluator.compute()
    assert metrics["FID_k"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["FID_g"] == pytest.approx(0.0, abs=1e-6)


def test_music_to_dance_evaluator_accepts_full_reference_pool():
    motions = [_dance(index + 10, index * 0.1) for index in range(5)]
    from motius.evaluation.metrics.dance_features import (
        extract_geometric_features,
        extract_kinetic_features,
    )
    from motius.evaluation.music_to_dance import root_anchor_motion

    reference = {
        "kinetic": np.stack(
            [extract_kinetic_features(root_anchor_motion(motion)) for motion in motions]
        ),
        "geometric": np.stack(
            [extract_geometric_features(root_anchor_motion(motion)) for motion in motions]
        ),
    }
    evaluator = AISTPPMusicDanceEvaluator(
        physical=False,
        reference_features=reference,
    )
    for motion in motions[:3]:
        evaluator.process(
            {
                "pred_joints": motion,
                "gt_joints": motion,
                "music_beats": np.ones(len(motion), dtype=bool),
            }
        )
    metrics = evaluator.compute()
    assert metrics["num_samples"] == 3
    assert metrics["num_reference_samples"] == 5
    assert metrics["reference_source"] == "aistpp_reference_feature_pool"


def test_music_to_dance_evaluator_artifact_round_trip(tmp_path: Path):
    motions = [_dance(index + 20, index * 0.1) for index in range(4)]
    from motius.evaluation.metrics.dance_features import (
        extract_geometric_features,
        extract_kinetic_features,
    )
    from motius.evaluation.music_to_dance import root_anchor_motion

    reference = {
        "kinetic": np.stack(
            [extract_kinetic_features(root_anchor_motion(motion)) for motion in motions]
        ),
        "geometric": np.stack(
            [extract_geometric_features(root_anchor_motion(motion)) for motion in motions]
        ),
    }
    evaluator = AISTPPMusicDanceEvaluator(
        physical=False,
        reference_features=reference,
        joint_reference_embeddings=np.eye(4, dtype=np.float32),
    )
    artifact = tmp_path / "evaluator"
    evaluator.save_pretrained(artifact)
    restored = AISTPPMusicDanceEvaluator.from_pretrained(
        artifact,
        physical=False,
    )
    np.testing.assert_array_equal(
        restored.reference_features["kinetic"], reference["kinetic"]
    )
    np.testing.assert_array_equal(
        restored.reference_features["geometric"], reference["geometric"]
    )
    np.testing.assert_array_equal(
        np.load(artifact / "aistpp_reference_utmr_embeddings.npy"),
        np.eye(4, dtype=np.float32),
    )


def test_beat_alignment_rescales_7p5_fps_music_frames():
    music = np.array([1, 3, 5])
    motion = music * 8
    assert beat_alignment_score(
        music, motion, music_fps=7.5, motion_fps=60.0
    ) == pytest.approx(1.0)


def test_beat_alignment_tolerance_is_frame_rate_invariant():
    music_60 = np.array([60, 120, 180])
    motion_60 = np.array([64, 116, 186])
    music_30 = music_60 // 2
    motion_30 = motion_60 // 2
    score_60 = beat_alignment_score(
        music_60, motion_60, music_fps=60.0, motion_fps=60.0
    )
    score_30 = beat_alignment_score(
        music_30, motion_30, music_fps=30.0, motion_fps=30.0
    )
    assert score_30 == pytest.approx(score_60)


def test_beat_alignment_60fps_is_exact_official_formula():
    music = np.array([4, 12, 31], dtype=np.float64)
    motion = np.array([7, 10, 35], dtype=np.float64)
    nearest = np.min((music[:, None] - motion[None]) ** 2, axis=1)
    expected = np.exp(-nearest / 18.0).mean()
    assert beat_alignment_score(music, motion) == pytest.approx(expected, abs=1e-15)


def test_evaluator_uses_full_motion_and_truncates_music_for_beat_alignment():
    motions = [_dance(41, 0.0), _dance(42, 0.3)]
    evaluator = AISTPPMusicDanceEvaluator(max_frames=30, physical=False)
    expected = []
    for motion in motions:
        motion_beats = motion_beat_frames(motion)
        beats = np.zeros(140, dtype=bool)
        if len(motion_beats):
            beats[motion_beats] = True
        beats[120] = True  # Outside the generated velocity stream; official code drops it.
        expected.append(beat_alignment_score(beats[: len(motion) - 1], motion_beats))
        evaluator.process(
            {
                "pred_joints": motion,
                "gt_joints": motion,
                "music_beats": beats,
            }
        )
    assert evaluator.compute()["BeatAlign"] == pytest.approx(np.mean(expected))


def test_bailando_audio_feature_shape():
    sample_rate = 3_840
    time = np.arange(sample_rate * 2, dtype=np.float32) / sample_rate
    waveform = np.sin(2.0 * np.pi * 220.0 * time).astype(np.float32)
    features = extract_bailando_audio_features(waveform, sample_rate=sample_rate)
    assert features.ndim == 2
    assert features.shape[1] == 438
    assert np.isfinite(features).all()


def test_aistpp_music_to_dance_dataset_keeps_both_music_rates(tmp_path: Path):
    import json

    test_root = tmp_path / "test"
    feature_root = tmp_path / "features"
    test_root.mkdir()
    feature_root.mkdir()
    name = "gBR_sBM_cAll_d04_mBR0_ch01"
    dance = np.arange(6 * 72, dtype=np.float32).reshape(6, 72)
    full_music = np.zeros((48, 438), dtype=np.float32)
    full_music[[8, 24, 40], 53] = 1.0
    model_music = np.ones((6, 438), dtype=np.float32)
    (test_root / f"{name}.json").write_text(
        json.dumps(
            {
                "id": name,
                "dance_array": dance.tolist(),
                "music_array": full_music.tolist(),
            }
        )
    )
    (feature_root / "mBR0.json").write_text(
        json.dumps({"id": "mBR0", "music_array": model_music.tolist()})
    )

    dataset = AISTPPMusicDanceDataset(test_root, feature_root)
    sample = dataset[0]
    assert aistpp_music_id(name) == "mBR0"
    assert sample["gt_joints"].shape == (6, 24, 3)
    assert sample["music_features"].shape == (6, 438)
    np.testing.assert_array_equal(np.flatnonzero(sample["music_beats"]), [8, 24, 40])


def test_aistpp_smpl24_representation_exact_smpl22_subset():
    joints = np.arange(5 * 24 * 3, dtype=np.float32).reshape(5, 24, 3)
    assert get_spec("AIST++ SMPL-24").dim == 72
    native = convert_motion(joints.reshape(5, 72), "aistpp_joints72", "joints")
    smpl22 = convert_motion(joints, "aistpp_smpl24_joints", "smpl22_joints")
    np.testing.assert_array_equal(native, joints)
    np.testing.assert_array_equal(smpl22, joints[:, :22])


def test_aistpp_smpl24_fk_uses_scaled_root_and_parent_offsets():
    poses = np.zeros((2, 24, 3), dtype=np.float32)
    translation = np.asarray([[2.0, 4.0, 6.0], [4.0, 6.0, 8.0]], dtype=np.float32)
    offsets = np.zeros((24, 3), dtype=np.float32)
    offsets[1:, 1] = 0.1
    joints = aistpp_smpl24_fk(poses, translation, 2.0, offsets)
    np.testing.assert_allclose(joints[:, 0], translation / 2.0)
    for joint in range(1, 24):
        parent = int(AISTPP_SMPL24_PARENTS[joint])
        difference = joints[:, joint] - joints[:, parent]
        np.testing.assert_allclose(
            difference,
            np.broadcast_to(offsets[joint], difference.shape),
            atol=1e-6,
        )


def test_native_skeleton_web_asset_preserves_30fps_joint_positions():
    joints = np.arange(6 * 24 * 3, dtype=np.float32).reshape(6, 24, 3) / 100.0
    sampled = resample_joint_positions(
        joints, source_fps=60.0, target_fps=30.0, target_frames=3
    )
    np.testing.assert_array_equal(sampled, joints[[0, 2, 4]])

    encoded, descriptor = encode_joint_positions(sampled)
    quantized = np.frombuffer(encoded, dtype="<u2").reshape(3, 24, 3)
    restored = (
        np.asarray(descriptor["position_minimum"], dtype=np.float32)
        + quantized * np.asarray(descriptor["position_scale"], dtype=np.float32)
    )
    tolerance = max(descriptor["position_scale"]) + 1e-7
    np.testing.assert_allclose(restored, sampled, atol=tolerance)
    assert descriptor["display_frames"] == 3
    assert descriptor["joint_count"] == 24
