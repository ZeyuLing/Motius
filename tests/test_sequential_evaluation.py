import numpy as np
import pytest

from motius.evaluation.sequential import (
    SequentialCase,
    SequentialSegment,
    _frechet_distance,
    evaluate_sequential_cases,
    load_joints66,
)


class _Evaluator:
    def encode_motions(self, motions):
        return np.stack([motion.mean(axis=0)[:4] for motion in motions])

    def evaluate(self, captions, predicted, reference, **_kwargs):
        assert len(captions) == len(predicted)
        assert reference is None
        assert len(_kwargs["positive_group_ids"]) == len(captions)
        return {
            "n_samples_used": len(captions),
            "r_precision": [0.25, 0.5, 0.75],
            "matching_score": 1.25,
            "fid": 0.5,
            "diversity_reference": 0.9,
            "diversity_predicted": 0.8,
        }


class _RecordingEvaluator(_Evaluator):
    def __init__(self):
        self.evaluated_motions = []
        self.encoded_motions = []

    def encode_motions(self, motions):
        self.encoded_motions.append([np.asarray(item).copy() for item in motions])
        return super().encode_motions(motions)

    def evaluate(self, captions, predicted, reference, **kwargs):
        self.evaluated_motions.append(
            [np.asarray(item).copy() for item in predicted]
        )
        return super().evaluate(captions, predicted, reference, **kwargs)


def _motion(frames, offset=0.0):
    time = np.arange(frames, dtype=np.float32)[:, None, None]
    joints = np.arange(22, dtype=np.float32)[None, :, None]
    xyz = np.arange(3, dtype=np.float32)[None, None, :]
    return (time * 0.001 + joints * 0.01 + xyz * 0.1 + offset).astype(np.float32)


def _segmented_motion():
    base = np.zeros((22, 3), dtype=np.float32)
    base[:, 1] = np.linspace(0.9, 1.8, 22)
    base[1, 0], base[2, 0] = -0.2, 0.2
    base[16, 0], base[17, 0] = -0.35, 0.35
    base[[7, 8, 10, 11], 1] = 0.0
    motion = []
    for frame in range(90):
        segment = frame // 30
        local_frame = frame % 30
        yaw = (0.15, 0.85, -1.1)[segment] + local_frame * 0.005
        rotation = np.asarray(
            [
                [np.cos(yaw), 0.0, np.sin(yaw)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw), 0.0, np.cos(yaw)],
            ],
            dtype=np.float32,
        )
        translation = np.asarray(
            [
                local_frame * 0.015 + (0.0, 2.0, -1.5)[segment],
                0.0,
                local_frame * 0.01 + (0.0, -1.0, 2.5)[segment],
            ],
            dtype=np.float32,
        )
        motion.append(base @ rotation.T + translation)
    return np.asarray(motion, dtype=np.float32)


def test_load_joints66_accepts_joint_tensor(tmp_path):
    path = tmp_path / "motion.npy"
    np.save(path, _motion(12))
    loaded = load_joints66(path)
    assert loaded.shape == (12, 66)
    assert loaded.dtype == np.float32


def test_sequential_case_rejects_overlapping_segments(tmp_path):
    with pytest.raises(ValueError, match="overlapping"):
        SequentialCase.from_mapping(
            {
                "case_id": "bad",
                "reference_path": "gt.npy",
                "prediction_path": "pred.npy",
                "segments": [
                    {"caption": "walk", "start_frame": 0, "end_frame": 20},
                    {"caption": "turn", "start_frame": 19, "end_frame": 40},
                ],
            },
            base_dir=tmp_path,
        )


def test_evaluate_sequential_cases_reports_semantic_and_transition_metrics(tmp_path):
    cases = []
    for index in range(2):
        reference = tmp_path / f"gt_{index}.npy"
        prediction = tmp_path / f"pred_{index}.npy"
        np.save(reference, _motion(90, offset=index * 0.01))
        np.save(prediction, _motion(90, offset=index * 0.01 + 0.002))
        cases.append(
            SequentialCase(
                f"case-{index}",
                reference,
                prediction,
                (
                    SequentialSegment("walk", 0, 30),
                    SequentialSegment("turn", 30, 60),
                    SequentialSegment("sit", 60, 90),
                ),
            )
        )
    summary = evaluate_sequential_cases(
        cases,
        _Evaluator(),
        transition_frames=20,
        protocol="babel-official-val-shortmerge30-llm-joints66-v1",
    )
    assert summary["protocol"] == "babel-official-val-shortmerge30-llm-joints66-v1"
    assert summary["fid_embedding_space"] == "l2_normalized"
    assert summary["n_cases"] == 2
    assert summary["n_segments"] == 6
    assert summary["n_transitions"] == 4
    assert summary["caption_groups"]["unique"] == 3
    assert summary["caption_groups"]["duplicate_groups"] == 3
    assert summary["subsequence"]["mm_dist"] == 1.25
    assert "matching_score" not in summary["subsequence"]
    assert summary["reference_subsequence"]["mm_dist"] == 1.25
    assert summary["reference_subsequence"]["fid"] == 0.0
    assert summary["reference_transition"]["fid"] == 0.0
    assert summary["reference_transition"]["auj_gap"] == 0.0
    assert np.isfinite(summary["transition"]["fid"])
    assert summary["transition"]["auj_gap"] >= 0


def test_official_style_cases_use_independent_reference_pools(tmp_path):
    prediction = tmp_path / "pred.npy"
    np.save(prediction, _motion(90, offset=0.004))
    case = SequentialCase(
        "official-000",
        None,
        prediction,
        (
            SequentialSegment("walk", 0, 30),
            SequentialSegment("turn", 30, 60),
            SequentialSegment("sit", 60, 90),
        ),
    )
    references = [_motion(30, offset=index * 0.01) for index in range(5)]
    transitions = [_motion(20, offset=index * 0.02) for index in range(4)]
    summary = evaluate_sequential_cases(
        [case],
        _Evaluator(),
        reference_segment_pool=references,
        reference_transition_pool=transitions,
        transition_frames=20,
    )
    assert summary["n_reference_segments"] == 5
    assert summary["n_reference_transitions"] == 4
    assert summary["reference_subsequence"] is None
    assert np.isfinite(summary["subsequence"]["fid"])


def test_tmr_semantics_recanonicalize_each_subclip_and_preserve_transition_gap(
    tmp_path,
):
    motion = _segmented_motion()
    prediction = tmp_path / "pred.npy"
    np.save(prediction, motion)
    case = SequentialCase(
        "canonical-audit",
        None,
        prediction,
        (
            SequentialSegment("walk", 0, 30),
            SequentialSegment("turn", 30, 60),
            SequentialSegment("sit", 60, 90),
        ),
    )
    evaluator = _RecordingEvaluator()
    summary = evaluate_sequential_cases(
        [case],
        evaluator,
        reference_segment_pool=[motion[0:30], motion[30:60], motion[60:90]],
        reference_transition_pool=[motion[20:40], motion[50:70]],
        transition_frames=20,
    )

    semantic_clips = evaluator.evaluated_motions[0]
    assert len(semantic_clips) == 3
    for clip in semantic_clips:
        shaped = clip.reshape(-1, 22, 3)
        np.testing.assert_allclose(shaped[0, 0, (0, 2)], 0.0, atol=1e-6)
        right = (shaped[0, 2] - shaped[0, 1]) + (
            shaped[0, 17] - shaped[0, 16]
        )
        right[1] = 0.0
        right /= np.linalg.norm(right)
        forward = np.cross(right, np.asarray([0.0, 1.0, 0.0]))
        np.testing.assert_allclose(forward, [0.0, 0.0, 1.0], atol=1e-6)

    predicted_transition_windows = evaluator.encoded_motions[-1]
    assert len(predicted_transition_windows) == 2
    for window in predicted_transition_windows:
        roots = window.reshape(-1, 22, 3)[:, 0]
        assert np.linalg.norm(roots[10, (0, 2)] - roots[9, (0, 2)]) > 1.0
    assert summary["canonicalization"] == {
        "semantic": "independent_per_subclip_first_frame",
        "transition": "independent_per_boundary_window_first_frame",
        "transition_gap_policy": "preserved_within_window",
    }


def test_sequential_fid_is_invariant_to_per_sample_embedding_scale():
    embeddings = np.asarray(
        [[1.0, 0.1], [0.1, 1.0], [0.8, 0.6], [0.3, 0.9]],
        dtype=np.float32,
    )
    scaled = embeddings * np.asarray([[2.0], [4.0], [8.0], [16.0]], dtype=np.float32)
    assert abs(_frechet_distance(embeddings, scaled)) < 1e-8
