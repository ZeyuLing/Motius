import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_monocular_methods_have_linked_model_cards():
    cards = {
        "GVHMR": ("gvhmr.md", True),
        "GEM-SMPL": ("gem_smpl.md", True),
        "GEM-X": ("gem_x.md", True),
    }
    model_zoo = (ROOT / "docs/model_zoo/README.md").read_text()
    task = (
        "[Monocular Motion Capture]"
        "(https://huggingface.co/spaces/ZeyuLing/"
        "monocular-motion-capture-leaderboard)"
    )
    for method, (filename, links_leaderboard) in cards.items():
        card = ROOT / "docs/model_zoo" / filename
        assert card.is_file()
        text = card.read_text()
        assert "Official source" in text
        assert "checkpoint" in text.lower()
        assert "**Tasks:** Monocular Motion Capture" in text
        assert "## Evaluation Results" in text
        expected_task = task if links_leaderboard else "Monocular Motion Capture"
        assert f"| [{method}]({filename}) | {expected_task} |" in model_zoo

    prompthmr = (ROOT / "docs/model_zoo/prompthmr.md").read_text()
    assert "Official source" in prompthmr
    assert "| Task status | Restricted upstream runtime |" in prompthmr
    assert "Pipeline.from_pretrained" in prompthmr
    assert (
        "| [PromptHMR-Video](prompthmr.md) | "
        "**Restricted upstream runtime** |"
    ) in model_zoo


def test_monocular_leaderboard_publication_has_verified_results_and_demos():
    path = (
        ROOT
        / "docs/leaderboards/hf_space_monocular_capture/"
        "monocular_capture_results.json"
    )
    results = json.loads(path.read_text())
    assert results["rows"][0]["method"] == "GT"
    assert len(results["rows"]) == 5
    assert "publication is paused" not in results["verification_policy"].lower()
    assert all(method.get("demo", {}).get("video") for method in results["methods"])


def test_monocular_runtime_legal_files_are_packaged():
    pyproject = (ROOT / "pyproject.toml").read_text()
    expected = (
        '"motius.models.gvhmr" = ["ATTRIBUTIONS.md"]',
        '"motius.models.prompthmr" = ["ATTRIBUTIONS.md"]',
        '"motius.models.gem_smpl" = ["ATTRIBUTIONS.md"]',
        '"motius.models.gem_smpl.vendor" = [',
        '"GENMO_LICENSE",',
        '"GVHMR_LICENSE",',
        '"motius.models.gem_x" = ["ATTRIBUTIONS.md"]',
        '"motius.models.gem_x.vendor" = [',
        '"GEM_X_LICENSE",',
        '"third_party/**/*LICENSE*",',
    )
    assert all(entry in pyproject for entry in expected)
