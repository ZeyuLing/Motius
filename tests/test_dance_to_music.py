from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from motius.evaluation.dance_to_music import (
    D2MGANBeatScore,
    aggregate_d2mgan_beat_scores,
    d2mgan_beat_score,
)
from motius.evaluation.protocols import d2mgan_aistpp_test_segments
from tools.build_dance_to_music_leaderboard import build_results, sync_result_page

REPO_ROOT = Path(__file__).resolve().parents[1]
D2M_SPACE = REPO_ROOT / "docs" / "leaderboards" / "hf_space_dance_to_music"


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

    assert result["schema_version"] == 3
    assert result["results_scope"] == "motius_measured_only"
    assert "paper_rows" not in result
    assert [row["method"] for row in result["rows"]] == ["UniMuMo"]
    reproduction = result["rows"][0]
    assert reproduction["beat_count_ratio"] == 1.08
    assert "beats_coverage" not in reproduction
    coverage_note = result["protocol"]["coverage_note"]
    assert "target of 100%" in coverage_note
    assert "no upper bound" in coverage_note
    assert "paper" not in coverage_note.lower()


def test_d2m_public_surface_omits_reference_and_paper_result_rows() -> None:
    payload = json.loads(
        (D2M_SPACE / "dance_to_music_results.json").read_text(encoding="utf-8")
    )
    page = (D2M_SPACE / "index.html").read_text(encoding="utf-8")

    assert payload["results_scope"] == "motius_measured_only"
    assert [row["method"] for row in payload["rows"]] == ["UniMuMo"]
    assert all(row["source"] == "motius" for row in payload["rows"])
    assert "paper_rows" not in payload
    result = payload["rows"][0]
    assert f"{100 * result['beat_count_ratio']:.2f}%" in page
    assert f"{100 * result['beat_hit_rate']:.2f}%" in page
    assert "Paper Results" not in page
    assert "data.paper_rows" not in page


def test_d2m_builder_synchronizes_static_result_card(tmp_path: Path) -> None:
    result_page = tmp_path / "index.html"
    result_page.write_text(
        '<dd class="metric-value" data-result-field="beat_count_ratio">0%</dd>'
        '<dd class="metric-value" data-result-field="beat_hit_rate">0%</dd>',
        encoding="utf-8",
    )
    results = build_results(
        {
            "dataset": "official split",
            "n_samples": 86,
            "aggregation": "macro",
            "protocol": "official detector",
            "beat_count_ratio": 1.08,
            "beat_hit_rate": 0.88,
        }
    )

    sync_result_page(result_page, results)

    page = result_page.read_text(encoding="utf-8")
    assert "108.00%" in page
    assert "88.00%" in page


def test_d2m_inference_preserves_the_official_119_frame_tail(tmp_path) -> None:
    from tools.infer_unimumo_d2mgan_aistpp import load_segment

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
    from tools.prepare_unimumo_d2mgan_aistpp_motion import required_frames_by_motion

    required = required_frames_by_motion()

    assert required["gBR_sBM_cAll_d04_mBR0_ch02"] == 714
