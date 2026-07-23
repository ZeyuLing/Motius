import json
from pathlib import Path

import pytest

from motius.evaluation.artifacts import EvaluationArtifactLayout


ROOT = Path(__file__).resolve().parents[1]


def test_layout_initializes_and_validates_protocol_and_run(tmp_path: Path) -> None:
    layout = EvaluationArtifactLayout(
        task_id="text_to_motion",
        benchmark_id="text_to_motion_unitree_g1",
        protocol_id="unitree-g1-paper-eval-1024-v1",
        root=tmp_path,
    )
    layout.init_protocol({"evaluator": "motius-evaluator-g1-38d-tmr"})
    layout.protocol_manifest.write_text('{"id": "sample-1"}\n')
    run_root = layout.init_run(
        "hymotion-g1",
        "iter-20000-seed-20260707",
        {"checkpoint": "iter_20000"},
    )

    assert layout.validate(require_manifest=True) == []
    assert run_root == (
        tmp_path
        / "text_to_motion"
        / "text_to_motion_unitree_g1"
        / "unitree-g1-paper-eval-1024-v1"
        / "runs"
        / "hymotion-g1"
        / "iter-20000-seed-20260707"
    )
    assert (run_root / "predictions").is_dir()
    assert (run_root / "metrics").is_dir()
    assert (run_root / "visualization").is_dir()
    assert (run_root / "logs").is_dir()


def test_layout_rejects_ambiguous_or_unsafe_ids() -> None:
    with pytest.raises(ValueError):
        EvaluationArtifactLayout("Text to Motion", "benchmark", "v1")
    with pytest.raises(ValueError):
        EvaluationArtifactLayout("text_to_motion", "../benchmark", "v1")


def test_layout_detects_metadata_identity_mismatch(tmp_path: Path) -> None:
    layout = EvaluationArtifactLayout("task", "benchmark", "protocol", tmp_path)
    layout.init_protocol()
    metadata = json.loads(layout.protocol_metadata.read_text())
    metadata["protocol_id"] = "other"
    layout.protocol_metadata.write_text(json.dumps(metadata))

    assert any("protocol_id='other'" in error for error in layout.validate())


def test_every_benchmark_registers_a_canonical_artifact_root() -> None:
    registry = json.loads((ROOT / "docs/tasks/taxonomy.json").read_text())
    for benchmark in registry["benchmarks"]:
        protocol_id = benchmark["protocol_id"]
        expected = (
            "outputs/evaluation/"
            f"{benchmark['task']}/{benchmark['id']}/{protocol_id}"
        )
        assert benchmark["artifact_root"] == expected
        EvaluationArtifactLayout(
            benchmark["task"], benchmark["id"], protocol_id
        )
