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
    assert len(data["methods"]) == 4
    method_names = {method["method"] for method in data["methods"]}
    assert method_names == {
        "GVHMR",
        "PromptHMR-Video",
        "GEM-SMPL",
        "GEM-X",
    }
    assert data["rows"][0]["method"] == "GT"
    assert data["rows"][0]["reference"] is True
    assert {row["method"] for row in data["rows"][1:]} == method_names
    assert all(
        method["demo"]["video"].startswith("https://")
        for method in data["methods"]
    )


def test_monocular_capture_page_publishes_demos_and_body_model_contract():
    page = (ROOT / "index.html").read_text()
    script = (ROOT / "leaderboard.js").read_text()

    assert "Monocular Motion Capture · 3DPW Test" in page
    assert "All-method visual comparison" in page
    assert "SMPL / SMPL-X / SOMA" in page
    assert "Missing compatible surfaces remain unavailable" in page
    assert 'document.getElementById("demo-grid")' in script
    assert "method.demo.video" in script
    assert "renderTable()" in script
