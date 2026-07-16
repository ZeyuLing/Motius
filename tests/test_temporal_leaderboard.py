import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "docs" / "leaderboards" / "hf_space_temporal_condition"
T2M_PAGE = ROOT / "docs" / "leaderboards" / "hf_space_t2m_humanml3d" / "index.html"


def _temporal_data():
    return json.loads((SPACE_DIR / "temporal_control_results.json").read_text())


def _t2m_gt_row():
    page = T2M_PAGE.read_text()
    match = re.search(r'\{method: "GT", version: "0 beta"[^\n]+\}', page)
    assert match, "T2M GT row is missing"
    fields = {}
    for key, value in re.findall(r"(msN|msR[123]|msFID|msMM|msDiv|utmrN|utmrR[123]|utmrFID|utmrMM|utmrDiv): ([0-9.]+)", match.group(0)):
        fields[key] = float(value)
    return fields


def test_temporal_control_gt_reuses_t2m_semantic_metrics():
    gt = _temporal_data()["gt_reference"]
    t2m_gt = _t2m_gt_row()

    assert gt["samples"] == int(t2m_gt["utmrN"])
    assert gt["metrics"]["r_precision_top1"] == t2m_gt["utmrR1"]
    assert gt["metrics"]["r_precision_top2"] == t2m_gt["utmrR2"]
    assert gt["metrics"]["r_precision_top3"] == t2m_gt["utmrR3"]
    assert gt["metrics"]["fid"] == t2m_gt["utmrFID"]
    assert gt["metrics"]["mm_dist"] == t2m_gt["utmrMM"]
    assert gt["metrics"]["diversity"] == t2m_gt["utmrDiv"]


def test_temporal_control_gt_only_supplements_temporal_metrics():
    gt = _temporal_data()["gt_reference"]

    assert gt["is_reference"] is True
    assert gt["rank_excluded"] is True
    assert gt["temporal_samples"] == 4012
    assert gt["metrics"]["constraint_error_cm"] == 0.0
    assert gt["metrics"]["fail_20"] == 0.0
    assert gt["metrics"]["fail_50"] == 0.0
    assert gt["metrics"]["foot_skating"] == pytest.approx(0.08914819392293807)
    assert "one 4012-clip pass" in gt["sources"]["temporal"]


def test_temporal_control_snapshot_covers_all_official_settings():
    data = _temporal_data()
    settings = data["settings"]

    assert data["num_cases"] == 4012
    assert len(settings) == 8
    assert {setting["task"] for setting in settings} == {"Prediction", "MIB", "Keyframe"}
    assert {setting["id"] for setting in settings} == {
        "temporal_start_1f",
        "temporal_pre20",
        "temporal_pre20_uncond",
        "temporal_both_1f",
        "temporal_mid80",
        "temporal_mid80_uncond",
        "temporal_adaptive_keyframes",
        "temporal_adaptive_keyframes_uncond",
    }
    assert all(setting["review_status"] == "complete" for setting in settings)
    assert all(setting["methods"] for setting in settings)


def test_frontend_keeps_gt_out_of_rankings_and_exposes_both_protocols():
    page = (SPACE_DIR / "index.html").read_text()
    script = (SPACE_DIR / "leaderboard.js").read_text()

    assert 'data-protocol="control"' in page
    assert 'data-protocol="tp2m"' in page
    assert 'src="leaderboard.js"' in page
    assert 'fetch("temporal_control_results.json")' in script
    assert "!row.isReference" in script
    assert "activeRows().filter(isRankable)" in script
    assert "GT is visible but excluded from all ranks and charts" in script
    assert script.count('method: "GT", settingId: "c') == 3
