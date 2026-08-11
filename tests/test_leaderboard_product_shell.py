import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LEADERBOARD_ROOT = ROOT / "docs" / "leaderboards"
PRODUCT_PAGES = {
    "part_level": LEADERBOARD_ROOT / "hf_space_body_part_condition_humanml3d",
    "reconstruction": LEADERBOARD_ROOT / "hf_space_motion_reconstruction",
    "monocular": LEADERBOARD_ROOT / "hf_space_monocular_capture",
}
CASE_EXPLORERS = {
    "part_level": {"population": 4012, "methods": 6, "chunks": 16},
    "reconstruction": {"population": 4042, "methods": 10, "chunks": 16},
}
SHELL_TOKENS = (
    'class="score-strip"',
    'class="panel comparison-studio"',
    'id="bar-chart"',
    'id="radar-chart"',
    'id="table-body"',
    'class="panel protocol-details"',
    'class="site-footer"',
)


def test_new_leaderboards_share_one_product_shell() -> None:
    canonical_style = (ROOT / "tools" / "leaderboard_product.css").read_text()

    for leaderboard in PRODUCT_PAGES.values():
        page = (leaderboard / "index.html").read_text()
        assert all(token in page for token in SHELL_TOKENS)
        assert (leaderboard / "leaderboard.css").read_text() == canonical_style


def test_part_level_page_exposes_all_measured_settings_without_a_table_wall() -> None:
    root = PRODUCT_PAGES["part_level"]
    payload = json.loads((root / "body_part_condition_results.json").read_text())
    page = (root / "index.html").read_text()
    script = (root / "leaderboard.js").read_text()

    assert len(payload["settings"]) == payload["num_settings"] == 84
    assert 'id="setting-select"' in page
    assert "syncSettingSelect()" in script
    assert "unsupported" not in json.dumps(payload).lower()
    assert page.count("<table") == 1


def test_task_specific_content_remains_part_of_the_shared_shell() -> None:
    reconstruction = (PRODUCT_PAGES["reconstruction"] / "index.html").read_text()
    monocular = (PRODUCT_PAGES["monocular"] / "index.html").read_text()

    assert "Geometry reconstruction" in reconstruction
    assert "Physical diagnostics" in reconstruction
    assert 'id="demo-grid"' in monocular
    assert "All-method visual comparison" in monocular


def test_reconstruction_publishes_every_completed_tokenizer() -> None:
    root = PRODUCT_PAGES["reconstruction"]
    payload = json.loads((root / "reconstruction_results.json").read_text())

    assert [row["method"] for row in payload["rows"]] == [
        "GT",
        "T2M-GPT / MotionGPT",
        "MoMask",
        "MLD / MotionLCM",
        "MoGenTS",
        "MotionGPT3",
        "MotionStreamer",
        "GoToZero / MotionMillion",
        "PRISM",
        "VerMo",
    ]
    assert all(row["geometry_samples"] == 4042 for row in payload["rows"])


def test_reconstruction_radar_assigns_a_defined_color_to_every_method() -> None:
    script = (PRODUCT_PAGES["reconstruction"] / "leaderboard.js").read_text()

    assert "COLORS[index % COLORS.length]" in script
    assert "backgroundColor: `${color}18`" in script


def test_motion_benchmarks_embed_complete_case_explorers() -> None:
    for key, expected in CASE_EXPLORERS.items():
        root = PRODUCT_PAGES[key]
        page = (root / "index.html").read_text()
        manifest = json.loads((root / "cases" / "manifest.json").read_text())

        assert 'class="panel case-explorer"' in page
        assert 'src="cases/index.html"' in page
        assert manifest["population"] == expected["population"]
        assert len(manifest["motion_methods"]) == expected["methods"]
        chunk_spec = manifest["case_descriptor_chunks"]
        chunk_count = (
            manifest["population"] + chunk_spec["size"] - 1
        ) // chunk_spec["size"]
        assert chunk_count == expected["chunks"]

    part_manifest = json.loads(
        (PRODUCT_PAGES["part_level"] / "cases" / "manifest.json").read_text()
    )
    assert part_manifest["cases"][0]["condition_intervals"]
    assert part_manifest["cases"][0]["condition"]["position_joint_indices"] == [20]
