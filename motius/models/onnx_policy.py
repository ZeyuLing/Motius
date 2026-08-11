"""Small, dependency-light runtime for self-describing ONNX policy artifacts."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from motius.models.base_model_bundle import ModelBundle


def resolve_pretrained_directory(
    pretrained_model_name_or_path: str | Path,
    *,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    token: Optional[str] = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local directory or one immutable Hugging Face snapshot."""

    local = Path(str(pretrained_model_name_or_path)).expanduser()
    if local.is_dir():
        return local.resolve()
    if local.is_file():
        return local.resolve().parent

    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            repo_id=str(pretrained_model_name_or_path),
            revision=revision,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            token=token,
            local_files_only=local_files_only,
        )
    )


def as_numpy(value: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if dtype is not None and array.dtype != dtype:
        array = array.astype(dtype, copy=False)
    return np.ascontiguousarray(array)


def _copy_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()
    if mode == "hardlink":
        try:
            os.link(source.resolve(), destination)
            return
        except OSError:
            pass
    elif mode != "copy":
        raise ValueError(f"Unsupported artifact copy mode: {mode!r}")
    shutil.copy2(source, destination)


class OnnxRuntimeSession:
    """Validated NumPy-facing wrapper around one ONNX Runtime session."""

    _DTYPES = {
        "tensor(float)": np.dtype(np.float32),
        "tensor(double)": np.dtype(np.float64),
        "tensor(int64)": np.dtype(np.int64),
        "tensor(int32)": np.dtype(np.int32),
        "tensor(bool)": np.dtype(np.bool_),
    }

    def __init__(
        self,
        model_path: str | Path,
        *,
        providers: Optional[Sequence[str]] = None,
        provider_options: Optional[Sequence[Mapping[str, Any]]] = None,
        session_options: Any = None,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Missing ONNX policy: {self.model_path}")
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "ONNX motion tracking requires `pip install 'motius[motion-tracking]'`."
            ) from exc

        selected = list(providers) if providers is not None else None
        if selected is None:
            available = set(ort.get_available_providers())
            selected = []
            if "CUDAExecutionProvider" in available:
                selected.append("CUDAExecutionProvider")
            selected.append("CPUExecutionProvider")
        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=selected,
                provider_options=(
                    list(provider_options) if provider_options is not None else None
                ),
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not load ONNX policy {self.model_path}. Ensure that the "
                "installed onnxruntime supports the model IR/opset version."
            ) from exc

        self.inputs = {item.name: item for item in self.session.get_inputs()}
        self.outputs = {item.name: item for item in self.session.get_outputs()}

    @property
    def metadata(self) -> dict[str, str]:
        return dict(self.session.get_modelmeta().custom_metadata_map)

    def _coerce_input(self, name: str, value: Any) -> np.ndarray:
        spec = self.inputs[name]
        array = as_numpy(value, self._DTYPES.get(spec.type))
        expected = tuple(spec.shape)
        if len(array.shape) != len(expected):
            raise ValueError(
                f"Input {name!r} expects rank {len(expected)}, got shape {array.shape}."
            )
        for axis, (actual, declared) in enumerate(zip(array.shape, expected)):
            if isinstance(declared, int) and declared >= 0 and actual != declared:
                raise ValueError(
                    f"Input {name!r} axis {axis} expects {declared}, got {actual}."
                )
        return array

    def run(
        self,
        feeds: Mapping[str, Any],
        output_names: Optional[Sequence[str]] = None,
    ) -> dict[str, np.ndarray]:
        missing = sorted(set(self.inputs) - set(feeds))
        extra = sorted(set(feeds) - set(self.inputs))
        if missing or extra:
            raise ValueError(
                f"ONNX feed mismatch for {self.model_path.name}: "
                f"missing={missing}, extra={extra}."
            )
        names = list(output_names) if output_names is not None else list(self.outputs)
        unknown = sorted(set(names) - set(self.outputs))
        if unknown:
            raise ValueError(f"Unknown ONNX outputs requested: {unknown}")
        arrays = {
            name: self._coerce_input(name, value)
            for name, value in feeds.items()
        }
        values = self.session.run(names, arrays)
        return dict(zip(names, values))


class OnnxTrackingBundle(ModelBundle):
    """Base bundle for method-specific, self-describing ONNX policies."""

    CONFIG_NAME = "tracking_config.json"
    MODEL_TYPE = "onnx_motion_tracker"
    METHOD_NAME = "ONNX motion tracker"
    PIPELINE_CLASS = ""
    BUNDLE_CLASS = ""
    DEFAULT_FILES: Mapping[str, str] = {}
    ONNX_ROLES: tuple[str, ...] = ()
    SUPPORTED_TASKS = {"motion_tracking": "closed-loop reference-motion tracking"}
    ARTIFACT_FORMAT = "motius-onnx-motion-tracking-v1"

    def __init__(
        self,
        *,
        artifact_dir: Optional[str | Path] = None,
        files: Optional[Mapping[str, str]] = None,
        file_paths: Optional[Mapping[str, str | Path]] = None,
        artifact_metadata: Optional[Mapping[str, Any]] = None,
        providers: Optional[Sequence[str]] = None,
        provider_options: Optional[Sequence[Mapping[str, Any]]] = None,
        load_model: bool = True,
    ) -> None:
        super().__init__()
        self.artifact_dir = (
            Path(artifact_dir).expanduser().resolve()
            if artifact_dir is not None
            else None
        )
        self.artifact_files = dict(self.DEFAULT_FILES)
        if files:
            self.artifact_files.update({str(k): str(v) for k, v in files.items()})
        explicit_paths = {
            str(key): Path(value).expanduser().resolve()
            for key, value in (file_paths or {}).items()
        }
        unknown_roles = sorted(set(explicit_paths) - set(self.artifact_files))
        if unknown_roles:
            raise ValueError(
                f"Unknown {self.METHOD_NAME} artifact roles: {unknown_roles}."
            )
        self.file_paths: dict[str, Path] = {}
        for role, relative in self.artifact_files.items():
            if role in explicit_paths:
                self.file_paths[role] = explicit_paths[role]
            elif self.artifact_dir is not None:
                self.file_paths[role] = self.artifact_dir / relative
            else:
                self.file_paths[role] = Path(relative).expanduser().resolve()
        self.artifact_metadata = dict(artifact_metadata or {})
        self.providers = tuple(providers) if providers is not None else None
        self.provider_options = (
            tuple(dict(item) for item in provider_options)
            if provider_options is not None
            else None
        )
        self._sessions: dict[str, OnnxRuntimeSession] = {}
        self.validate_files()
        if load_model:
            self.load_model()

    def validate_files(self) -> None:
        missing = [
            f"{role}={path}"
            for role, path in self.file_paths.items()
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"Incomplete {self.METHOD_NAME} artifact: " + ", ".join(missing)
            )

    def load_model(self) -> "OnnxTrackingBundle":
        for role in self.ONNX_ROLES:
            if role not in self._sessions:
                self._sessions[role] = OnnxRuntimeSession(
                    self.file_paths[role],
                    providers=self.providers,
                    provider_options=self.provider_options,
                )
        return self

    def session(self, role: str) -> OnnxRuntimeSession:
        if role not in self.ONNX_ROLES:
            raise KeyError(f"{role!r} is not an ONNX role for {self.METHOD_NAME}.")
        if role not in self._sessions:
            self.load_model()
        return self._sessions[role]

    def artifact_contract(self) -> dict[str, Any]:
        return {}

    def artifact_source(self) -> dict[str, Any]:
        return {}

    def _artifact_payload(self) -> dict[str, Any]:
        payload = {
            "format": self.ARTIFACT_FORMAT,
            "model_type": self.MODEL_TYPE,
            "method": self.METHOD_NAME,
            "runtime": {"files": dict(self.artifact_files)},
            "contract": self.artifact_contract(),
            "source": self.artifact_source(),
        }
        reserved = {
            "contract",
            "format",
            "method",
            "model_type",
            "runtime",
            "source",
        }
        for key, value in self.artifact_metadata.items():
            if key not in reserved:
                payload[key] = value
        return payload

    def save_pretrained(
        self,
        save_directory: str | Path,
        *,
        copy_mode: str = "copy",
        readme: Optional[str] = None,
        **_: Any,
    ) -> str:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        for role, relative in self.artifact_files.items():
            _copy_file(self.file_paths[role], save_dir / relative, copy_mode)

        payload = self._artifact_payload()
        (save_dir / self.CONFIG_NAME).write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
        required_files = [self.CONFIG_NAME, *self.artifact_files.values()]
        model_index = {
            "_class_name": self.PIPELINE_CLASS.rsplit(".", 1)[-1],
            "_library_name": "motius",
            "format": self.ARTIFACT_FORMAT,
            "model_type": self.MODEL_TYPE,
            "bundle_class": self.BUNDLE_CLASS,
            "pipeline_class": self.PIPELINE_CLASS,
            "tasks": list(self.SUPPORTED_TASKS),
            "required_files": required_files,
            "api": {
                "loader": "motius.Pipeline.from_pretrained",
                "inference": "infer_motion_tracking",
                "physical_rollout": "rollout_motion_tracking",
            },
        }
        (save_dir / "model_index.json").write_text(
            json.dumps(model_index, indent=2) + "\n",
            encoding="utf-8",
        )
        if readme is not None:
            (save_dir / "README.md").write_text(
                readme.rstrip() + "\n",
                encoding="utf-8",
            )
        return str(save_dir)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        revision: Optional[str] = None,
        cache_dir: Optional[str | Path] = None,
        token: Optional[str] = None,
        local_files_only: bool = False,
        **kwargs: Any,
    ) -> "OnnxTrackingBundle":
        root = resolve_pretrained_directory(
            pretrained_model_name_or_path,
            revision=revision,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
        )
        config_path = root / cls.CONFIG_NAME
        if not config_path.is_file():
            raise FileNotFoundError(
                f"{cls.METHOD_NAME} artifact is missing {cls.CONFIG_NAME}: {root}"
            )
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        if payload.get("model_type") != cls.MODEL_TYPE:
            raise ValueError(
                f"Expected model_type={cls.MODEL_TYPE!r}, got "
                f"{payload.get('model_type')!r}."
            )
        files = payload.get("runtime", {}).get("files")
        if not isinstance(files, Mapping):
            raise ValueError(f"{config_path} has no runtime.files mapping.")
        return cls(
            artifact_dir=root,
            files=files,
            artifact_metadata=payload,
            **kwargs,
        )


__all__ = [
    "OnnxRuntimeSession",
    "OnnxTrackingBundle",
    "as_numpy",
    "resolve_pretrained_directory",
]
