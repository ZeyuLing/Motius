import json
from pathlib import Path


ROOT = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "leaderboards"
    / "hf_space_monocular_capture"
)


def test_monocular_capture_leaderboard_is_verified_only():
    data = json.loads((ROOT / "monocular_capture_results.json").read_text())

    assert set(data["protocols"]) == {
        "3dpw_test_camera_v1",
        "emdb_1_camera_v1",
        "emdb_2_global_v1",
    }
    assert all(row.get("verified") is True for row in data["rows"])
    assert len(data["methods"]) == 5
    assert {method["method"] for method in data["methods"]} == {
        "GVHMR",
        "PromptHMR-Video",
        "GEM-SMPL",
        "GEM-X",
        "HYMotion-V2M",
    }


def test_monocular_capture_page_separates_protocols_and_body_models():
    page = (ROOT / "index.html").read_text()
    script = (ROOT / "leaderboard.js").read_text()

    assert "3DPW Test · Camera" in page
    assert "EMDB-1 · Camera" in page
    assert "EMDB-2 · World" in page
    assert "SOMA-77" in page
    assert "never through fabricated SMPL vertices" in page
    assert "row.verified===true" in script
    assert "No verified rows yet." in script
