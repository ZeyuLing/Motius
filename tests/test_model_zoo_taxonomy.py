import json
from pathlib import Path

from tools.audit_model_zoo_release import TASK_LABELS, _read_model_rows, _task_status


ROOT = Path(__file__).resolve().parents[1]


def test_model_zoo_uses_canonical_task_labels() -> None:
    rows = _read_model_rows()
    assert len(rows) == 30

    for row in rows:
        card_text = row.card_path.read_text()
        status, note = _task_status(row.task_cell, card_text)
        assert status == "valid", f"{row.method}: {note}"


def test_release_manifest_task_labels_are_canonical() -> None:
    manifest = json.loads(
        (ROOT / "docs/model_zoo/release_manifest.json").read_text()
    )
    for model in manifest["models"].values():
        values = model.get("tasks")
        if values is None and "task" in model:
            values = [part.strip() for part in model["task"].split(",")]
        if values is not None:
            assert set(values).issubset(TASK_LABELS)


def test_root_readme_uses_only_representation_conversion_visuals() -> None:
    readme = (ROOT / "README.md").read_text()
    assert "<table>" not in readme
    assert "assets/model_zoo/" not in readme
    assert "004822_hml_smpl_soma_core_g1.gif" in readme
    assert "interx_smplh_gt_G021T002A012R014_skeleton_smpl_mesh.gif" in readme
    assert "004822_skeleton_smpl_mixamo_1440_30fps.gif" in readme
