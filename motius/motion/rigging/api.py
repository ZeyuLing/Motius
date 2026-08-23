"""Public static-character auto-rigging API."""

from __future__ import annotations

import json
import subprocess
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from motius.motion.fbx.api import resolve_blender_executable

from .mia import (
    DEFAULT_MIA_SPACE,
    MIA_REST_POSES,
    MIXAMO_TO_SMPL22,
    request_make_it_animatable,
)
from .template import TemplateRiggingConfig

SUPPORTED_CHARACTER_INPUTS = frozenset(
    {".fbx", ".glb", ".gltf", ".obj", ".ply", ".stl"}
)
SUPPORTED_RIG_OUTPUTS = frozenset({".fbx", ".glb", ".gltf"})
AUTO_RIG_METHODS = frozenset({"template", "make_it_animatable"})


class CharacterRiggingError(RuntimeError):
    """Raised when a character cannot be imported, bound, or exported."""


@dataclass(frozen=True)
class CharacterRiggingResult:
    """Paths and diagnostics produced by :func:`auto_rig_character`."""

    output_path: Path
    manifest_path: Path
    method: str
    armature_name: str
    mesh_names: tuple[str, ...]
    joint_names: tuple[str, ...]
    warnings: tuple[str, ...]
    metadata: Mapping[str, object]


def auto_rig_character(
    character_path: str | Path,
    output_path: str | Path,
    *,
    method: str = "template",
    blender_executable: str | Path | None = None,
    up_axis: str = "auto",
    top_k: int = 4,
    weight_falloff: float = 1.75,
    side_penalty: float = 0.025,
    weight_method: str = "capsules",
    replace_existing_rig: bool = False,
    mia_space: str = DEFAULT_MIA_SPACE,
    mia_rest_pose: str = "No",
) -> CharacterRiggingResult:
    """Automatically fit and skin an upright humanoid character.

    ``character_path`` may be FBX, GLB/GLTF, OBJ, PLY, or STL.  Output must be
    FBX, GLB, or GLTF.  The built-in ``template`` method targets a T/A-pose
    humanoid and creates the canonical Motius SMPL22 bone names so the FBX can
    be passed directly to :func:`motius.motion.retarget_smpl_to_fbx`.

    ``method="make_it_animatable"`` selects the optional, higher-coverage
    Make-It-Animatable backend used by the public multi-character demo. It
    uploads the source to ``mia_space``, downloads a Mixamo-compatible FBX,
    and normalizes its canonical bone subset to Motius SMPL22 in Blender. Use
    a trusted self-hosted Space for private character assets.

    Blender 3.6+ performs material-preserving asset I/O.  It is an external
    runtime, not a Python package dependency.
    """

    source = Path(character_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Character asset does not exist: {source}.")
    if source.suffix.casefold() not in SUPPORTED_CHARACTER_INPUTS:
        supported = ", ".join(sorted(SUPPORTED_CHARACTER_INPUTS))
        raise ValueError(
            f"Unsupported character format {source.suffix!r}; expected one of {supported}."
        )
    if output.suffix.casefold() not in SUPPORTED_RIG_OUTPUTS:
        supported = ", ".join(sorted(SUPPORTED_RIG_OUTPUTS))
        raise ValueError(
            f"Unsupported rig output {output.suffix!r}; expected one of {supported}."
        )
    if source == output:
        raise ValueError("character_path and output_path must be different files.")

    method = str(method).strip().casefold().replace("-", "_")
    if method == "mia":
        method = "make_it_animatable"
    if method not in AUTO_RIG_METHODS:
        choices = ", ".join(sorted(AUTO_RIG_METHODS))
        raise ValueError(f"method must be one of {choices}.")
    up_axis = str(up_axis).upper() if str(up_axis).casefold() != "auto" else "auto"
    if up_axis not in {"auto", "X", "Y", "Z", "-X", "-Y", "-Z"}:
        raise ValueError("up_axis must be auto, X, Y, Z, -X, -Y, or -Z.")
    if up_axis != "auto" and source.suffix.casefold() in {".fbx", ".glb", ".gltf"}:
        raise ValueError(
            "Explicit up_axis is only valid for OBJ/PLY/STL. FBX and GLTF "
            "declare their coordinate system and Blender converts it on import."
        )
    if method == "make_it_animatable" and up_axis != "auto":
        raise ValueError(
            "The Make-It-Animatable backend currently requires up_axis='auto'; "
            "normalize OBJ/PLY/STL coordinates before uploading."
        )
    weight_method = str(weight_method).casefold()
    if weight_method not in {"automatic", "capsules"}:
        raise ValueError("weight_method must be 'automatic' or 'capsules'.")
    if mia_rest_pose not in MIA_REST_POSES:
        choices = ", ".join(sorted(MIA_REST_POSES))
        raise ValueError(f"mia_rest_pose must be one of {choices}.")
    if method == "make_it_animatable" and replace_existing_rig:
        raise ValueError(
            "The Make-It-Animatable backend expects a static unrigged input; "
            "replace_existing_rig is only supported by the local template method."
        )
    blender = resolve_blender_executable(blender_executable)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = Path(f"{output}.json")

    with tempfile.TemporaryDirectory(prefix=".motius_rig_", dir=output.parent) as tmp:
        job_path = Path(tmp) / "job.json"
        if method == "template":
            config = TemplateRiggingConfig(
                top_k=top_k,
                weight_falloff=weight_falloff,
                side_penalty=side_penalty,
            )
            script = Path(__file__).with_name("_blender.py").resolve()
            template_module = Path(__file__).with_name("template.py").resolve()
            job = {
                "schema_version": 1,
                "method": method,
                "character_path": str(source),
                "output_path": str(output),
                "manifest_path": str(manifest),
                "template_module": str(template_module),
                "up_axis": up_axis,
                "replace_existing_rig": bool(replace_existing_rig),
                "weight_method": weight_method,
                "config": {
                    "top_k": int(config.top_k),
                    "weight_falloff": float(config.weight_falloff),
                    "side_penalty": float(config.side_penalty),
                    "chunk_size": int(config.chunk_size),
                },
            }
        else:
            raw_fbx = Path(tmp) / "make_it_animatable_raw.fbx"
            try:
                backend = request_make_it_animatable(
                    source,
                    raw_fbx,
                    space=mia_space,
                    rest_pose=mia_rest_pose,
                )
            except Exception as error:
                raise CharacterRiggingError(
                    f"Make-It-Animatable failed for {source}: {error}"
                ) from error
            script = Path(__file__).with_name("_blender_mia.py").resolve()
            job = {
                "schema_version": 1,
                "method": method,
                "source_character": str(source),
                "raw_fbx": str(raw_fbx),
                "output_path": str(output),
                "manifest_path": str(manifest),
                "bone_mapping": MIXAMO_TO_SMPL22,
                "backend": backend,
            }
        job_path.write_text(json.dumps(job, indent=2) + "\n", encoding="utf-8")
        command = [
            str(blender),
            "--background",
            "--factory-startup",
            "--python-exit-code",
            "1",
            "--python",
            str(script),
            "--",
            "--job",
            str(job_path),
        ]
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if completed.returncode != 0:
            tail = "\n".join((completed.stdout or "").splitlines()[-100:])
            raise CharacterRiggingError(
                f"Blender character rigging failed with exit code "
                f"{completed.returncode}:\n{tail}"
            )

    if not output.is_file() or output.stat().st_size == 0:
        raise CharacterRiggingError(
            f"Blender did not create a non-empty rigged character at {output}."
        )
    if not manifest.is_file():
        raise CharacterRiggingError(
            f"Blender did not create the rigging manifest at {manifest}."
        )
    metadata = json.loads(manifest.read_text(encoding="utf-8"))
    return CharacterRiggingResult(
        output_path=output,
        manifest_path=manifest,
        method=str(metadata["method"]),
        armature_name=str(metadata["armature_name"]),
        mesh_names=tuple(str(value) for value in metadata["mesh_names"]),
        joint_names=tuple(str(value) for value in metadata["joint_names"]),
        warnings=tuple(str(value) for value in metadata.get("warnings", ())),
        metadata=metadata,
    )


__all__ = [
    "AUTO_RIG_METHODS",
    "SUPPORTED_CHARACTER_INPUTS",
    "SUPPORTED_RIG_OUTPUTS",
    "CharacterRiggingError",
    "CharacterRiggingResult",
    "auto_rig_character",
]
