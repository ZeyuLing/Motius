"""Automatic loader for self-describing Motius pipeline artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional


MODEL_INDEX_NAME = "model_index.json"

# Old artifacts did not all include a fully-qualified pipeline_class. Resolve
# their public class name through a closed allowlist instead of importing an
# arbitrary path from remote metadata.
PIPELINE_CLASS_PATHS = {
    "Any2TrackPipeline": "motius.pipelines.any2track.Any2TrackPipeline",
    "ARDYPipeline": "motius.pipelines.ardy.ARDYPipeline",
    "BailandoPipeline": "motius.pipelines.bailando.BailandoPipeline",
    "BeyondMimicPipeline": "motius.pipelines.beyondmimic.BeyondMimicPipeline",
    "CondMDIPipeline": "motius.pipelines.condmdi.CondMDIPipeline",
    "DARTPipeline": "motius.pipelines.dart.DARTPipeline",
    "EDGEPipeline": "motius.pipelines.edge.EDGEPipeline",
    "FlowMDMPipeline": "motius.pipelines.flowmdm.FlowMDMPipeline",
    "GemSmplPipeline": "motius.pipelines.gem_smpl.GemSmplPipeline",
    "GemXPipeline": "motius.pipelines.gem_x.GemXPipeline",
    "GVHMRPipeline": "motius.pipelines.gvhmr.GVHMRPipeline",
    "HumanoidGPTPipeline": "motius.pipelines.humanoid_gpt.HumanoidGPTPipeline",
    "HyMotionT2MPipeline": "motius.pipelines.hymotion_t2m.HyMotionT2MPipeline",
    "InterGenPipeline": "motius.pipelines.intergen.InterGenPipeline",
    "InterMaskPipeline": "motius.pipelines.intermask.InterMaskPipeline",
    "KIMODOPipeline": "motius.pipelines.kimodo.KIMODOPipeline",
    "MaskControlPipeline": "motius.pipelines.maskcontrol.MaskControlPipeline",
    "MDMPipeline": "motius.pipelines.mdm.MDMPipeline",
    "MLDPipeline": "motius.pipelines.mld.MLDPipeline",
    "MoGenTSPipeline": "motius.pipelines.mogents.MoGenTSPipeline",
    "MoMaskPipeline": "motius.pipelines.momask.MoMaskPipeline",
    "MotionBricksPipeline": "motius.pipelines.motionbricks.MotionBricksPipeline",
    "MotionCanvasPipeline": "motius.pipelines.motioncanvas.MotionCanvasPipeline",
    "MotionCLRPipeline": "motius.pipelines.motionclr.MotionCLRPipeline",
    "MotionGPTPipeline": "motius.pipelines.motiongpt.MotionGPTPipeline",
    "MotionGPT3Pipeline": "motius.pipelines.motiongpt3.MotionGPT3Pipeline",
    "MotionLCMPipeline": "motius.pipelines.motionlcm.MotionLCMPipeline",
    "MotionMillionPipeline": "motius.pipelines.motionmillion.MotionMillionPipeline",
    "MotionStreamerPipeline": "motius.pipelines.motionstreamer.MotionStreamerPipeline",
    "OmniControlPipeline": "motius.pipelines.omnicontrol.OmniControlPipeline",
    "PRISMPipeline": "motius.pipelines.prism.PRISMPipeline",
    "ProjFlowPipeline": "motius.pipelines.projflow.ProjFlowPipeline",
    "ProtoMotionsPipeline": "motius.pipelines.protomotions.ProtoMotionsPipeline",
    "SONICPipeline": "motius.pipelines.sonic.SONICPipeline",
    "T2MGPTPipeline": "motius.pipelines.t2mgpt.T2MGPTPipeline",
    "TM2DPipeline": "motius.pipelines.tm2d.TM2DPipeline",
    "TM2TPipeline": "motius.pipelines.tm2t.TM2TPipeline",
    "UniMuMoPipeline": "motius.pipelines.unimumo.UniMuMoPipeline",
    "VermoPipeline": "motius.pipelines.vermo.VermoPipeline",
    "ViMoGenPipeline": "motius.pipelines.vimogen.ViMoGenPipeline",
}


class PipelineArtifactError(RuntimeError):
    """Raised when an artifact cannot satisfy the Motius pipeline contract."""


@dataclass(frozen=True)
class PipelineMetadata:
    """Resolved, trusted metadata for one local or Hub artifact."""

    source: str
    revision: Optional[str]
    model_index_path: Path
    pipeline_class_path: str
    bundle_class_path: Optional[str]
    tasks: tuple[str, ...]
    manifest: Mapping[str, Any]

    @property
    def pipeline_class_name(self) -> str:
        return self.pipeline_class_path.rsplit(".", 1)[-1]


def _import_object(path: str):
    module_name, separator, object_name = path.rpartition(".")
    if not separator:
        raise PipelineArtifactError(f"Invalid Python class path: {path!r}")
    module = importlib.import_module(module_name)
    return getattr(module, object_name)


def _trusted_pipeline_class_path(value: Any) -> Optional[str]:
    """Normalize a declared path to one registered local pipeline class."""

    if not isinstance(value, str):
        return None
    class_name = value.rsplit(".", 1)[-1]
    canonical = PIPELINE_CLASS_PATHS.get(class_name)
    if canonical is None:
        return None
    package = canonical.rsplit(".", 1)[0]
    if value == canonical or (
        value.startswith(f"{package}.") and value.endswith(f".{class_name}")
    ):
        return canonical
    return None


def _pipeline_path_from_manifest(manifest: Mapping[str, Any]) -> str:
    explicit = _trusted_pipeline_class_path(manifest.get("pipeline_class"))
    if explicit is not None:
        return explicit

    raw_legacy_pipeline = manifest.get("pipeline")
    legacy_pipeline = _trusted_pipeline_class_path(raw_legacy_pipeline)
    if legacy_pipeline is not None:
        return legacy_pipeline
    if isinstance(raw_legacy_pipeline, Mapping):
        module = raw_legacy_pipeline.get("module")
        class_name = raw_legacy_pipeline.get("class_name")
        legacy_pipeline = _trusted_pipeline_class_path(f"{module}.{class_name}")
        if legacy_pipeline is not None:
            return legacy_pipeline

    class_name = manifest.get("_class_name")
    if isinstance(class_name, str) and class_name in PIPELINE_CLASS_PATHS:
        return PIPELINE_CLASS_PATHS[class_name]

    raise PipelineArtifactError(
        "model_index.json does not declare a trusted Motius pipeline. "
        "Expected pipeline_class='motius.pipelines....' and a registered class."
    )


def _bundle_path_from_manifest(manifest: Mapping[str, Any]) -> Optional[str]:
    value = manifest.get("bundle_class")
    if isinstance(value, str) and value.startswith("motius.models."):
        return value
    value = manifest.get("bundle")
    if isinstance(value, str) and value.startswith("motius.models."):
        return value
    return None


def _tasks_from_manifest(manifest: Mapping[str, Any]) -> tuple[str, ...]:
    tasks = manifest.get("tasks")
    if tasks is None:
        tasks = manifest.get("supported_tasks", ())
    if isinstance(tasks, Mapping):
        tasks = tuple(tasks)
    if isinstance(tasks, str):
        tasks = [tasks]
    if not isinstance(tasks, (list, tuple)):
        return ()
    return tuple(str(task) for task in tasks)


def _load_model_index(path: Path) -> Mapping[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PipelineArtifactError(
            f"Missing required {MODEL_INDEX_NAME}: {path}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PipelineArtifactError(f"Invalid {MODEL_INDEX_NAME}: {path}") from exc
    if not isinstance(manifest, Mapping):
        raise PipelineArtifactError(f"{MODEL_INDEX_NAME} must contain a JSON object")
    return manifest


class Pipeline:
    """Load any self-describing Motius checkpoint from a path or Hub repo.

    The remote artifact selects a class only through a closed Motius allowlist.
    No remote Python code is downloaded or executed.
    """

    @classmethod
    def resolve_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
    ) -> PipelineMetadata:
        source = str(pretrained_model_name_or_path)
        local_path = Path(source).expanduser()
        resolved_revision = revision

        if local_path.is_dir():
            index_path = local_path / MODEL_INDEX_NAME
        elif local_path.is_file() and local_path.name == MODEL_INDEX_NAME:
            index_path = local_path
        else:
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as exc:
                raise PipelineArtifactError(
                    "huggingface_hub is required to load a Hub checkpoint"
                ) from exc
            try:
                index_path = Path(
                    hf_hub_download(
                        repo_id=source,
                        filename=MODEL_INDEX_NAME,
                        revision=revision,
                        cache_dir=str(cache_dir) if cache_dir is not None else None,
                        token=token,
                        local_files_only=local_files_only,
                    )
                )
            except Exception as exc:
                raise PipelineArtifactError(
                    f"Could not resolve {MODEL_INDEX_NAME} for {source!r}: {exc}"
                ) from exc
            snapshot_component = next(
                (
                    part
                    for part in index_path.parts
                    if len(part) == 40
                    and all(char in "0123456789abcdef" for char in part.lower())
                ),
                None,
            )
            resolved_revision = snapshot_component or revision

        manifest = _load_model_index(index_path)
        return PipelineMetadata(
            source=source,
            revision=resolved_revision,
            model_index_path=index_path,
            pipeline_class_path=_pipeline_path_from_manifest(manifest),
            bundle_class_path=_bundle_path_from_manifest(manifest),
            tasks=_tasks_from_manifest(manifest),
            manifest=manifest,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
        bundle_kwargs: Optional[Mapping[str, Any]] = None,
        **pipeline_kwargs,
    ):
        metadata = cls.resolve_pretrained(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        source_path = Path(str(pretrained_model_name_or_path)).expanduser()
        if source_path.is_file() and source_path.name == MODEL_INDEX_NAME:
            source_path = source_path.parent
        elif not source_path.is_dir():
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise PipelineArtifactError(
                    "huggingface_hub is required to load a Hub checkpoint"
                ) from exc
            source_path = Path(
                snapshot_download(
                    repo_id=str(pretrained_model_name_or_path),
                    revision=metadata.revision or revision,
                    cache_dir=str(cache_dir) if cache_dir is not None else None,
                    token=token,
                    local_files_only=local_files_only,
                )
            )

        cls._validate_required_files(source_path, metadata.manifest)
        pipeline_class = _import_object(metadata.pipeline_class_path)

        from motius.pipelines.base_pipeline import BasePipeline

        if isinstance(pipeline_class, type) and issubclass(pipeline_class, BasePipeline):
            return pipeline_class.from_pretrained(
                str(source_path),
                bundle_kwargs=dict(bundle_kwargs or {}),
                **pipeline_kwargs,
            )
        load_kwargs = dict(bundle_kwargs or {})
        load_kwargs.update(pipeline_kwargs)
        return pipeline_class.from_pretrained(str(source_path), **load_kwargs)

    @staticmethod
    def _validate_required_files(
        artifact_path: Path,
        manifest: Mapping[str, Any],
    ) -> None:
        required_files = manifest.get("required_files", ())
        if isinstance(required_files, str):
            required_files = [required_files]
        if not isinstance(required_files, (list, tuple)):
            raise PipelineArtifactError("required_files must be a JSON list")
        missing = [
            str(relative)
            for relative in required_files
            if not (artifact_path / str(relative)).exists()
        ]
        if missing:
            raise PipelineArtifactError(
                f"Checkpoint artifact is incomplete; missing: {', '.join(missing)}"
            )
