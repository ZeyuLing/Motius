from __future__ import annotations

import numpy as np
import pytest

from motius.evaluation.dance_to_music import (
    D2MGANBeatScore,
    aggregate_d2mgan_beat_scores,
    d2mgan_beat_score,
)
from motius.evaluation.protocols import d2mgan_aistpp_test_segments
from tools.build_dance_to_music_leaderboard import build_results
from tools.infer_unimumo_d2mgan_aistpp import load_segment
from tools.prepare_unimumo_d2mgan_aistpp_motion import required_frames_by_motion


def test_d2mgan_beat_score_matches_upstream_formula() -> None:
    result = d2mgan_beat_score(
        np.asarray([1, 0, 1, 0]),
        np.asarray([1, 1, 0, 1]),
    )

    assert result.beat_count_ratio == pytest.approx(1.5)
    assert result.beat_hit_rate == pytest.approx(0.5)
    assert result.reference_beat_bins == 2
    assert result.generated_beat_bins == 3
    assert result.hit_beat_bins == 1


def test_d2mgan_coverage_is_an_unbounded_beat_count_ratio() -> None:
    result = d2mgan_beat_score(
        np.asarray([1, 0, 0]),
        np.asarray([1, 1, 1]),
    )

    assert result.beat_count_ratio == 3.0
    assert result.beat_hit_rate == 1.0


def test_d2mgan_aggregate_is_per_clip_macro_average() -> None:
    result = aggregate_d2mgan_beat_scores(
        [
            D2MGANBeatScore(2.0, 1.0, 1, 2, 1),
            D2MGANBeatScore(0.5, 0.25, 4, 2, 1),
        ]
    )

    assert result == {
        "n_samples": 2,
        "beat_count_ratio": 1.25,
        "beat_hit_rate": 0.625,
    }


def test_d2mgan_aistpp_protocol_vendors_the_official_split() -> None:
    segments = d2mgan_aistpp_test_segments()

    assert len(segments) == 86
    assert len({segment.case_id for segment in segments}) == 86
    assert segments[0].source_motion_id == "gPO_sBM_cAll_d11_mPO1_ch02"
    assert segments[0].music_id == "mPO1"
    assert segments[0].start_seconds == 0.0
    assert segments[1].start_seconds == 2.0


def test_d2m_leaderboard_labels_coverage_as_a_target_ratio() -> None:
    result = build_results(
        {
            "dataset": "official split",
            "n_samples": 86,
            "aggregation": "macro",
            "protocol": "official detector",
            "coverage_note": "unbounded",
            "beat_count_ratio": 1.08,
            "beat_hit_rate": 0.88,
        }
    )

    reproduction = next(row for row in result["rows"] if row["source"] == "motius")
    assert reproduction["beat_count_ratio"] == 1.08
    assert "beats_coverage" not in reproduction
    assert result["protocol"]["coverage_note"] == "unbounded"


def test_d2m_inference_preserves_the_official_119_frame_tail(tmp_path) -> None:
    segment = next(
        item
        for item in d2mgan_aistpp_test_segments()
        if item.segment_index == 6
    )
    path = (
        tmp_path
        / "train"
        / "joint_vecs"
        / f"{segment.source_motion_id}.npy"
    )
    path.parent.mkdir(parents=True)
    np.save(path, np.zeros((719, 263), dtype=np.float32))

    _, clip = load_segment(tmp_path, segment)

    assert clip.shape == (119, 263)


def test_d2m_motion_preparation_requires_a_valid_tail_clip() -> None:
    required = required_frames_by_motion()

    assert required["gBR_sBM_cAll_d04_mBR0_ch02"] == 714
