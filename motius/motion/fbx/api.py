"""Public FBX export and rigged-character retargeting API."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np

from motius.motion.fbx._mapping import SMPL22_BONE_NAMES
from motius.motion.representation.rotation import (
    axis_angle_to_matrix,
    rotation_6d_to_matrix,
)
from motius.motion.skeleton.body_models import (
    _dense_array,
    _load_model_data,
    resolve_smpl_model_path,
)


SMPL_TO_BLENDER = np.asarray(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


class FBXExportError(RuntimeError):
    """Raised when an FBX backend cannot create the requested artifact."""


@dataclass(frozen=True)
class SMPLAnimation:
    """A constant-shape SMPL-family animation in local joint rotations."""

    local_rotations: np.ndarray
    translations: np.ndarray
    betas: np.ndarray
    fps: float = 30.0

    def __post_init__(self) -> None:
        rotations = np.asarray(self.local_rotations, dtype=np.float64)
        translations = np.asarray(self.translations, dtype=np.float64)
        betas = np.asarray(self.betas, dtype=np.float64).reshape(-1)
        if rotations.ndim != 4 or rotations.shape[-2:] != (3, 3):
            raise ValueError(
                "local_rotations must have shape (T,J,3,3), got "
                f"{rotations.shape}."
            )
        if translations.shape != (len(rotations), 3):
            raise ValueError(
                f"translations must have shape ({len(rotations)},3), got {translations.shape}."
            )
        if len(rotations) < 1 or rotations.shape[1] < 1:
            raise ValueError("SMPLAnimation requires at least one frame and one joint.")
        if not np.isfinite(rotations).all() or not np.isfinite(translations).all():
            raise ValueError("SMPLAnimation contains non-finite motion values.")
        if not np.isfinite(betas).all():
            raise ValueError("SMPLAnimation betas contain non-finite values.")
        if float(self.fps) <= 0:
            raise ValueError("fps must be positive.")
        identity = np.eye(3, dtype=np.float64)
        gram = np.swapaxes(rotations, -1, -2) @ rotations
        if not np.allclose(gram, identity, atol=1e-5, rtol=0):
            raise ValueError("local_rotations contain non-orthonormal matrices.")
        if not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-5, rtol=0):
            raise ValueError("local_rotations contain reflections.")
        rotations = np.ascontiguousarray(rotations)
        translations = np.ascontiguousarray(translations)
        betas = np.ascontiguousarray(betas)
        rotations.setflags(write=False)
        translations.setflags(write=False)
        betas.setflags(write=False)
        object.__setattr__(self, "local_rotations", rotations)
        object.__setattr__(self, "translations", translations)
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "fps", float(self.fps))

    @property
    def frames(self) -> int:
        return int(len(self.local_rotations))

    @classmethod
    def from_motion135(
        cls,
        motion135,
        *,
        betas=None,
        fps: float = 30.0,
    ) -> "SMPLAnimation":
        """Create an animation from Motius root-translation plus rot6d motion."""

        motion = np.asarray(motion135, dtype=np.float64)
        if motion.ndim != 2 or motion.shape[1] != 135:
            raise ValueError(f"motion135 must have shape (T,135), got {motion.shape}.")
        rotations = rotation_6d_to_matrix(
            motion[:, 3:].reshape(len(motion), 22, 6), convention="row"
        )
        shape = np.zeros(10, dtype=np.float64) if betas is None else betas
        return cls(rotations, motion[:, :3], shape, fps)

    @classmethod
    def from_smpl(
        cls,
        global_orient,
        body_pose,
        transl,
        *,
        betas=None,
        fps: float = 30.0,
    ) -> "SMPLAnimation":
        """Create an animation from standard axis-angle SMPL parameters."""

        root = np.asarray(global_orient, dtype=np.float64).reshape(-1, 3)
        frames = len(root)
        pose = np.asarray(body_pose, dtype=np.float64).reshape(frames, -1, 3)
        if pose.shape[1] < 1:
            raise ValueError("body_pose must contain at least one non-root joint.")
        translations = np.asarray(transl, dtype=np.float64)
        if translations.ndim == 1:
            translations = np.broadcast_to(translations, (frames, 3))
        translations = translations.reshape(frames, 3)
        local_axis_angle = np.concatenate([root[:, None], pose], axis=1)
        rotations = axis_angle_to_matrix(local_axis_angle.reshape(-1, 3)).reshape(
            frames, local_axis_angle.shape[1], 3, 3
        )
        shape = _constant_betas(betas, frames)
        return cls(rotations, translations, shape, fps)


@dataclass(frozen=True)
class FBXExportResult:
    output_path: Path
    manifest_path: Path
    mode: str
    frames: int
    fps: float
    armature_name: str
    bone_map: Mapping[str, str]
    metadata: Mapping[str, object]


def _constant_betas(value, frames: int) -> np.ndarray:
    if value is None:
        return np.zeros(10, dtype=np.float64)
    betas = np.asarray(value, dtype=np.float64)
    if betas.ndim == 1:
        return betas
    betas = betas.reshape(betas.shape[0], -1)
    if len(betas) == 1:
        return betas[0]
    if len(betas) != frames:
        raise ValueError(f"betas must have one row or {frames} rows, got {betas.shape}.")
    if not np.allclose(betas, betas[0], atol=1e-8, rtol=0):
        raise ValueError("FBX skin binding requires constant betas across the clip.")
    return betas[0]


def resolve_blender_executable(value: str | Path | None = None) -> Path:
    """Resolve an explicit Blender binary, ``MOTIUS_BLENDER``, or ``PATH``."""

    candidate = value or os.environ.get("MOTIUS_BLENDER") or shutil.which("blender")
    if not candidate:
        raise FileNotFoundError(
            "Blender was not found. Install Blender 3.6+ and pass blender_executable=, "
            "set MOTIUS_BLENDER, or add blender to PATH."
        )
    path = Path(candidate).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Blender executable does not exist: {path}.")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Blender executable is not executable: {path}.")
    return path.resolve()


def _fbxsdk_environment(module_path: Path | None) -> dict[str, str]:
    environment = os.environ.copy()
    if module_path is not None:
        current = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            f"{module_path}{os.pathsep}{current}" if current else str(module_path)
        )
    return environment


@lru_cache(maxsize=8)
def _probe_fbxsdk_runtime(python: str, module_path: str | None) -> None:
    root = Path(module_path) if module_path else None
    completed = subprocess.run(
        [python, "-c", "import fbx, FbxCommon, numpy, scipy"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_fbxsdk_environment(root),
        timeout=120,
    )
    if completed.returncode != 0:
        detail = "\n".join(completed.stdout.splitlines()[-20:])
        raise ImportError(
            f"Python runtime {python} cannot import Autodesk FBX SDK dependencies:\n{detail}"
        )


def resolve_fbxsdk_runtime(
    python_executable: str | Path | None = None,
    module_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """Resolve and validate a Python runtime containing Autodesk FBX SDK."""

    candidate = (
        python_executable
        or os.environ.get("MOTIUS_FBXSDK_PYTHON")
        or shutil.which("python3.10")
    )
    if not candidate:
        raise FileNotFoundError(
            "No FBX SDK Python runtime was found. Pass fbxsdk_python= or set "
            "MOTIUS_FBXSDK_PYTHON."
        )
    resolved_command = shutil.which(str(candidate))
    python = Path(resolved_command or candidate).expanduser()
    if not python.is_file() or not os.access(python, os.X_OK):
        raise FileNotFoundError(f"FBX SDK Python executable does not exist: {python}.")

    root_value = module_path or os.environ.get("MOTIUS_FBXSDK_PYTHONPATH")
    if root_value is None:
        standard = (
            Path(__file__).resolve().parents[3]
            / "checkpoints"
            / "fbxsdk"
            / "cp310"
        )
        root = standard if standard.is_dir() else None
    else:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"FBX SDK Python module path does not exist: {root}.")
    python = python.resolve()
    _probe_fbxsdk_runtime(str(python), str(root) if root else None)
    return python, root


def _model_joint_names(count: int) -> tuple[str, ...]:
    names = list(SMPL22_BONE_NAMES[: min(count, len(SMPL22_BONE_NAMES))])
    if count == 24:
        names.extend(["L_Hand", "R_Hand"])
    else:
        names.extend(f"SMPL_Joint_{index:02d}" for index in range(len(names), count))
    return tuple(names)


def _parents_from_tree(tree: np.ndarray, count: int) -> np.ndarray:
    tree = np.asarray(tree)
    if tree.ndim == 2 and tree.shape[0] >= 2:
        parent_ids = tree[0, :count]
        joint_ids = tree[1, :count]
        index_by_id = {int(joint_id): index for index, joint_id in enumerate(joint_ids)}
        parents = np.full(count, -1, dtype=np.int64)
        for index in range(1, count):
            parent_id = int(parent_ids[index])
            if parent_id not in index_by_id:
                raise ValueError(f"SMPL kinematic parent id {parent_id} is not in the tree.")
            parents[index] = index_by_id[parent_id]
        return parents
    parents = np.asarray(tree, dtype=np.int64).reshape(-1)[:count].copy()
    parents[0] = -1
    return parents


def _prepare_payload(
    animation: SMPLAnimation,
    *,
    model_path: str | Path,
    model_type: str,
    gender: str,
) -> tuple[dict[str, np.ndarray], Path]:
    resolved = resolve_smpl_model_path(
        model_path, model_type=model_type, gender=gender
    ).resolve()
    data = _load_model_data(resolved)
    required = {"v_template", "shapedirs", "J_regressor", "kintree_table", "weights"}
    missing = sorted(required.difference(data))
    if missing:
        raise KeyError(f"SMPL model {resolved} is missing FBX arrays: {missing}.")
    face_key = "f" if "f" in data else "faces" if "faces" in data else None
    if face_key is None:
        raise KeyError(f"SMPL model {resolved} is missing triangle faces ('f' or 'faces').")

    template = _dense_array(data["v_template"]).astype(np.float64)
    shapedirs = _dense_array(data["shapedirs"]).astype(np.float64)
    regressor = _dense_array(data["J_regressor"]).astype(np.float64)
    weights = _dense_array(data["weights"]).astype(np.float64)
    faces = _dense_array(data[face_key]).astype(np.int32)
    if shapedirs.ndim != 3 or shapedirs.shape[:2] != template.shape:
        raise ValueError(f"Invalid shapedirs {shapedirs.shape} for {template.shape}.")
    if weights.ndim != 2 or weights.shape[0] != len(template):
        raise ValueError(f"Invalid skin weights {weights.shape} for {template.shape}.")
    joint_count = int(weights.shape[1])
    if regressor.shape[0] < joint_count or regressor.shape[1] != len(template):
        raise ValueError(
            f"Invalid J_regressor {regressor.shape} for {joint_count} skin joints."
        )
    if animation.local_rotations.shape[1] > joint_count:
        raise ValueError(
            f"Animation has {animation.local_rotations.shape[1]} joints, model has {joint_count}."
        )
    beta_count = min(len(animation.betas), shapedirs.shape[-1])
    vertices = template + np.einsum(
        "vcb,b->vc", shapedirs[..., :beta_count], animation.betas[:beta_count]
    )
    rest_joints = regressor[:joint_count] @ vertices
    parents = _parents_from_tree(_dense_array(data["kintree_table"]), joint_count)
    if np.any(parents[1:] < 0) or np.any(parents[1:] >= np.arange(1, joint_count)):
        raise ValueError(f"SMPL parents are not topologically ordered: {parents.tolist()}.")
    if np.any(weights < -1e-8):
        raise ValueError("SMPL skin weights contain negative values.")
    weight_sum = weights.sum(axis=1, keepdims=True)
    if np.any(weight_sum <= 1e-8):
        raise ValueError("SMPL skin weights contain unbound vertices.")
    weights = weights / weight_sum

    frames = animation.frames
    local = np.broadcast_to(np.eye(3), (frames, joint_count, 3, 3)).copy()
    local[:, : animation.local_rotations.shape[1]] = animation.local_rotations
    global_rotations = np.empty_like(local)
    joints = np.empty((frames, joint_count, 3), dtype=np.float64)
    offsets = np.empty_like(rest_joints)
    offsets[0] = rest_joints[0]
    for joint in range(1, joint_count):
        offsets[joint] = rest_joints[joint] - rest_joints[parents[joint]]
    for joint, parent in enumerate(parents):
        if parent < 0:
            global_rotations[:, joint] = local[:, joint]
            joints[:, joint] = animation.translations + offsets[joint]
        else:
            global_rotations[:, joint] = global_rotations[:, parent] @ local[:, joint]
            joints[:, joint] = joints[:, parent] + (
                global_rotations[:, parent] @ offsets[joint, :, None]
            ).squeeze(-1)

    transform = SMPL_TO_BLENDER
    payload = {
        "vertices": np.asarray(vertices @ transform.T, dtype=np.float32),
        "faces": np.asarray(faces, dtype=np.int32),
        "weights": np.asarray(weights, dtype=np.float32),
        "rest_joints": np.asarray(rest_joints @ transform.T, dtype=np.float32),
        "parents": np.asarray(parents, dtype=np.int32),
        "joint_names": np.asarray(_model_joint_names(joint_count)),
        "joints": np.asarray(joints @ transform.T, dtype=np.float32),
        "global_rotations": np.asarray(
            np.einsum(
                "ab,fjbc,cd->fjad", transform, global_rotations, transform.T
            ),
            dtype=np.float32,
        ),
    }
    return payload, resolved


def _run_export(
    animation: SMPLAnimation,
    output_path: str | Path,
    *,
    model_path: str | Path,
    model_type: str,
    gender: str,
    backend: str,
    blender_executable: str | Path | None,
    fbxsdk_python: str | Path | None,
    fbxsdk_module_path: str | Path | None,
    character_fbx: str | Path | None,
    bone_map: Mapping[str, str] | None,
    target_armature: str | None,
    strict_bone_map: bool,
    root_motion_scale: float | str,
    source_metadata: Mapping[str, object] | None,
) -> FBXExportResult:
    output = Path(output_path).expanduser().resolve()
    if output.suffix.casefold() != ".fbx":
        raise ValueError(f"output_path must end in .fbx, got {output}.")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload, resolved_model = _prepare_payload(
        animation,
        model_path=model_path,
        model_type=model_type,
        gender=gender,
    )
    target = None
    if character_fbx is not None:
        target = Path(character_fbx).expanduser().resolve()
        if not target.is_file() or target.suffix.casefold() != ".fbx":
            raise FileNotFoundError(f"Rigged character FBX does not exist: {target}.")
        if target == output:
            raise ValueError("character_fbx and output_path must be different files.")
    if isinstance(root_motion_scale, str) and root_motion_scale != "auto":
        raise ValueError("root_motion_scale must be a positive float or 'auto'.")
    if not isinstance(root_motion_scale, str) and float(root_motion_scale) <= 0:
        raise ValueError("root_motion_scale must be positive.")

    manifest = Path(f"{output}.json")
    mode = "character_retarget" if target else "smpl_export"
    backend = str(backend).casefold()
    if backend not in {"auto", "fbxsdk", "blender"}:
        raise ValueError("backend must be 'auto', 'fbxsdk', or 'blender'.")
    fbxsdk_runtime = None
    selected_backend = backend
    if selected_backend == "auto":
        if target is not None:
            try:
                fbxsdk_runtime = resolve_fbxsdk_runtime(
                    fbxsdk_python, fbxsdk_module_path
                )
                selected_backend = "fbxsdk"
            except (FileNotFoundError, ImportError, subprocess.SubprocessError):
                selected_backend = "blender"
        else:
            selected_backend = "blender"
    if selected_backend == "fbxsdk":
        if target is None:
            raise ValueError(
                "The FBX SDK backend animates an existing character FBX. "
                "Direct skinned-SMPL FBX construction currently uses backend='blender'."
            )
        if fbxsdk_runtime is None:
            fbxsdk_runtime = resolve_fbxsdk_runtime(
                fbxsdk_python, fbxsdk_module_path
            )
        script = Path(__file__).with_name("_fbxsdk.py").resolve()
    else:
        blender = resolve_blender_executable(blender_executable)
        script = Path(__file__).with_name("_blender.py").resolve()
    with tempfile.TemporaryDirectory(prefix=".motius_fbx_", dir=output.parent) as tmp:
        tmp_dir = Path(tmp)
        payload_path = tmp_dir / "animation.npz"
        job_path = tmp_dir / "job.json"
        np.savez_compressed(payload_path, **payload)
        job = {
            "schema_version": 1,
            "backend": selected_backend,
            "mode": mode,
            "payload_path": str(payload_path),
            "output_path": str(output),
            "manifest_path": str(manifest),
            "source_model_path": str(resolved_model),
            "model_type": str(model_type),
            "gender": str(gender),
            "frames": animation.frames,
            "fps": animation.fps,
            "character_fbx": str(target) if target else None,
            "bone_map": dict(bone_map or {}),
            "target_armature": target_armature,
            "strict_bone_map": bool(strict_bone_map),
            "root_motion_scale": root_motion_scale,
            "source_metadata": dict(source_metadata or {}),
        }
        job_path.write_text(json.dumps(job, indent=2) + "\n")
        if selected_backend == "fbxsdk":
            python, module_root = fbxsdk_runtime
            command = [str(python), str(script), "--job", str(job_path)]
            environment = _fbxsdk_environment(module_root)
        else:
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
            environment = None
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=environment,
        )
        if completed.returncode != 0:
            tail = "\n".join(completed.stdout.splitlines()[-80:])
            raise FBXExportError(
                f"{selected_backend} FBX {mode} failed with exit code "
                f"{completed.returncode}:\n{tail}"
            )
    if not output.is_file() or output.stat().st_size == 0:
        raise FBXExportError(
            f"{selected_backend} did not create a non-empty FBX at {output}."
        )
    if not manifest.is_file():
        raise FBXExportError(
            f"{selected_backend} did not create the export manifest at {manifest}."
        )
    metadata = json.loads(manifest.read_text())
    return FBXExportResult(
        output_path=output,
        manifest_path=manifest,
        mode=mode,
        frames=animation.frames,
        fps=animation.fps,
        armature_name=str(metadata["armature_name"]),
        bone_map=dict(metadata.get("bone_map", {})),
        metadata=metadata,
    )


def export_smpl_fbx(
    animation: SMPLAnimation,
    output_path: str | Path,
    *,
    model_path: str | Path,
    model_type: str = "smpl",
    gender: str = "neutral",
    backend: str = "auto",
    blender_executable: str | Path | None = None,
    fbxsdk_python: str | Path | None = None,
    fbxsdk_module_path: str | Path | None = None,
    source_metadata: Mapping[str, object] | None = None,
) -> FBXExportResult:
    """Export an animated, skinned SMPL-family FBX."""

    return _run_export(
        animation,
        output_path,
        model_path=model_path,
        model_type=model_type,
        gender=gender,
        backend=backend,
        blender_executable=blender_executable,
        fbxsdk_python=fbxsdk_python,
        fbxsdk_module_path=fbxsdk_module_path,
        character_fbx=None,
        bone_map=None,
        target_armature=None,
        strict_bone_map=True,
        root_motion_scale=1.0,
        source_metadata=source_metadata,
    )


def retarget_smpl_to_fbx(
    animation: SMPLAnimation,
    character_fbx: str | Path,
    output_path: str | Path,
    *,
    model_path: str | Path,
    model_type: str = "smpl",
    gender: str = "neutral",
    bone_map: Mapping[str, str] | None = None,
    target_armature: str | None = None,
    strict_bone_map: bool = True,
    root_motion_scale: float | str = "auto",
    backend: str = "auto",
    blender_executable: str | Path | None = None,
    fbxsdk_python: str | Path | None = None,
    fbxsdk_module_path: str | Path | None = None,
    source_metadata: Mapping[str, object] | None = None,
) -> FBXExportResult:
    """Bake SMPL motion onto an already rigged and skinned character FBX."""

    return _run_export(
        animation,
        output_path,
        model_path=model_path,
        model_type=model_type,
        gender=gender,
        backend=backend,
        blender_executable=blender_executable,
        fbxsdk_python=fbxsdk_python,
        fbxsdk_module_path=fbxsdk_module_path,
        character_fbx=character_fbx,
        bone_map=bone_map,
        target_armature=target_armature,
        strict_bone_map=strict_bone_map,
        root_motion_scale=root_motion_scale,
        source_metadata=source_metadata,
    )


__all__ = [
    "FBXExportError",
    "FBXExportResult",
    "SMPLAnimation",
    "SMPL_TO_BLENDER",
    "export_smpl_fbx",
    "resolve_blender_executable",
    "resolve_fbxsdk_runtime",
    "retarget_smpl_to_fbx",
]
