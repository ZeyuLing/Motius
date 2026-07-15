import numpy as np
import pytest

from motius.evaluation.sequential import (
    SequentialCase,
    SequentialSegment,
    evaluate_sequential_cases,
    load_joints66,
)


class _Evaluator:
    def encode_motions(self, motions):
        return np.stack([motion.mean(axis=0)[:4] for motion in motions])

    def evaluate(self, captions, predicted, reference, **_kwargs):
        assert len(captions) == len(predicted)
        assert reference is None
        return {
            "n_samples_used": len(captions),
            "r_precision": [0.25, 0.5, 0.75],
            "matching_score": 1.25,
            "fid": 0.5,
            "diversity_reference": 0.9,
            "diversity_predicted": 0.8,
        }


def _motion(frames, offset=0.0):
    time = np.arange(frames, dtype=np.float32)[:, None, None]
    joints = np.arange(22, dtype=np.float32)[None, :, None]
    xyz = np.arange(3, dtype=np.float32)[None, None, :]
    return (time * 0.001 + joints * 0.01 + xyz * 0.1 + offset).astype(np.float32)


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
    assert summary["n_cases"] == 2
    assert summary["n_segments"] == 6
    assert summary["n_transitions"] == 4
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
