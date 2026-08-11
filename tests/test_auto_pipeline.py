import json
from pathlib import Path

import pytest

from motius.models.base_model_bundle import ModelBundle
from motius.pipelines.base_pipeline import BasePipeline
from motius.pipelines import Pipeline, PipelineArtifactError


def _write_index(path: Path, **overrides) -> Path:
    payload = {
        "_class_name": "MDMPipeline",
        "_library_name": "motius",
        "pipeline_class": "motius.pipelines.mdm.MDMPipeline",
        "bundle_class": "motius.models.mdm.MDMBundle",
        "tasks": ["text_to_motion"],
        "required_files": [],
    }
    payload.update(overrides)
    path.mkdir(parents=True)
    (path / "model_index.json").write_text(json.dumps(payload))
    return path


def test_resolve_local_pretrained_manifest(tmp_path):
    artifact = _write_index(tmp_path / "artifact")

    metadata = Pipeline.resolve_pretrained(artifact)

    assert metadata.pipeline_class_path == "motius.pipelines.mdm.MDMPipeline"
    assert metadata.bundle_class_path == "motius.models.mdm.MDMBundle"
    assert metadata.tasks == ("text_to_motion",)


def test_legacy_class_name_uses_closed_motius_mapping(tmp_path):
    artifact = _write_index(
        tmp_path / "artifact",
        pipeline_class="another_package.pipelines.mdm.MDMPipeline",
    )

    metadata = Pipeline.resolve_pretrained(artifact)

    assert metadata.pipeline_class_path == "motius.pipelines.mdm.MDMPipeline"


def test_registered_submodule_path_is_normalized_to_canonical_class(tmp_path):
    artifact = _write_index(
        tmp_path / "artifact",
        pipeline_class="motius.pipelines.mdm.pipeline.MDMPipeline",
    )

    metadata = Pipeline.resolve_pretrained(artifact)

    assert metadata.pipeline_class_path == "motius.pipelines.mdm.MDMPipeline"


def test_unregistered_motius_pipeline_class_is_rejected(tmp_path):
    artifact = _write_index(
        tmp_path / "artifact",
        _class_name="UnknownPipeline",
        pipeline_class="motius.pipelines.mdm.pipeline.UnknownPipeline",
    )

    with pytest.raises(PipelineArtifactError, match="trusted Motius pipeline"):
        Pipeline.resolve_pretrained(artifact)


def test_legacy_supported_tasks_mapping_is_resolved(tmp_path):
    artifact = _write_index(
        tmp_path / "artifact",
        tasks=None,
        supported_tasks={
            "text_to_motion": "generate motion from text",
            "kinematic_motion_control": "apply native constraints",
        },
    )

    metadata = Pipeline.resolve_pretrained(artifact)

    assert metadata.tasks == ("text_to_motion", "kinematic_motion_control")


def test_untrusted_unknown_pipeline_is_rejected(tmp_path):
    artifact = _write_index(
        tmp_path / "artifact",
        _class_name="UnknownPipeline",
        pipeline_class="another_package.remote.UnknownPipeline",
    )

    with pytest.raises(PipelineArtifactError, match="trusted Motius pipeline"):
        Pipeline.resolve_pretrained(artifact)


def test_required_files_are_fail_closed(tmp_path):
    artifact = _write_index(
        tmp_path / "artifact",
        required_files=["weights/model.safetensors"],
    )
    metadata = Pipeline.resolve_pretrained(artifact)

    with pytest.raises(PipelineArtifactError, match="weights/model.safetensors"):
        Pipeline._validate_required_files(artifact, metadata.manifest)


def test_invalid_required_files_type_is_rejected(tmp_path):
    artifact = _write_index(tmp_path / "artifact", required_files={"bad": "shape"})
    metadata = Pipeline.resolve_pretrained(artifact)

    with pytest.raises(PipelineArtifactError, match="JSON list"):
        Pipeline._validate_required_files(artifact, metadata.manifest)


def test_generic_loader_separates_bundle_and_pipeline_kwargs(tmp_path, monkeypatch):
    artifact = _write_index(tmp_path / "artifact")
    calls = {}

    class FakeBundle(ModelBundle):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["bundle"] = (path, kwargs)
            return cls()

    class FakePipeline(BasePipeline):
        BUNDLE_CLS = FakeBundle

        def __init__(self, bundle, *, device=None, **kwargs):
            super().__init__(bundle, **kwargs)
            self.requested_device = device

    import motius.pipelines.auto as auto

    monkeypatch.setattr(auto, "_import_object", lambda _path: FakePipeline)
    pipeline = Pipeline.from_pretrained(
        artifact,
        bundle_kwargs={"precision": "bf16"},
        device="cuda",
    )

    assert calls["bundle"] == (str(artifact), {"precision": "bf16"})
    assert pipeline.requested_device == "cuda"


def test_generic_loader_merges_kwargs_for_legacy_non_base_pipeline(
    tmp_path,
    monkeypatch,
):
    artifact = _write_index(tmp_path / "artifact")
    calls = {}

    class LegacyPipeline:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls["load"] = (path, kwargs)
            return cls()

    import motius.pipelines.auto as auto

    monkeypatch.setattr(auto, "_import_object", lambda _path: LegacyPipeline)
    Pipeline.from_pretrained(
        artifact,
        bundle_kwargs={"precision": "bf16"},
        num_steps=25,
    )

    assert calls["load"] == (
        str(artifact),
        {"precision": "bf16", "num_steps": 25},
    )
