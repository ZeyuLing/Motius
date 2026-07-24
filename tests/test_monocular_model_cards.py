import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_monocular_methods_have_linked_model_cards():
    cards = {
        "GVHMR": "gvhmr.md",
        "PromptHMR-Video": "prompthmr.md",
        "HYMotion-V2M": "hymotion_v2m.md",
        "GEM-SMPL": "gem_smpl.md",
        "GEM-X": "gem_x.md",
    }
    model_zoo = (ROOT / "docs/model_zoo/README.md").read_text()
    task = (
        "[Monocular Motion Capture]"
        "(https://huggingface.co/spaces/ZeyuLing/"
        "monocular-motion-capture-leaderboard)"
    )
    for method, filename in cards.items():
        card = ROOT / "docs/model_zoo" / filename
        assert card.is_file()
        text = card.read_text()
        assert "Official source" in text
        assert "checkpoint" in text.lower()
        assert "**Tasks:** Monocular Motion Capture" in text
        assert "## Evaluation Results" in text
        assert f"| [{method}]({filename}) | {task} |" in model_zoo


def test_monocular_leaderboard_publication_is_paused():
    path = (
        ROOT
        / "docs/leaderboards/hf_space_monocular_capture/"
        "monocular_capture_results.json"
    )
    results = json.loads(path.read_text())
    assert results["rows"] == []
    assert "publication is paused" in results["verification_policy"].lower()


def test_monocular_runtime_legal_files_are_packaged():
    pyproject = (ROOT / "pyproject.toml").read_text()
    expected = (
        '"motius.models.gvhmr" = ["ATTRIBUTIONS.md"]',
        '"motius.models.prompthmr" = ["ATTRIBUTIONS.md"]',
        '"motius.models.gem_smpl" = ["ATTRIBUTIONS.md", "setup_runtime.sh"]',
        '"motius.models.gem_x" = ["ATTRIBUTIONS.md", "setup_runtime.sh"]',
        (
            '"motius.models.hymotion_v2m" = '
            '["ATTRIBUTIONS.md", "License.txt", "NOTICE"]'
        ),
    )
    assert all(entry in pyproject for entry in expected)
