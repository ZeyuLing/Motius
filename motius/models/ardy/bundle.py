"""Motius model bundle for NVIDIA ARDY checkpoints."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


ARDY_CHECKPOINTS = {
    "core": "nvidia/ARDY-Core-RP-20FPS-Horizon40",
    "core40": "nvidia/ARDY-Core-RP-20FPS-Horizon40",
    "core8": "nvidia/ARDY-Core-RP-20FPS-Horizon8",
    "g1": "nvidia/ARDY-G1-RP-25FPS-Horizon52",
    "g152": "nvidia/ARDY-G1-RP-25FPS-Horizon52",
    "g18": "nvidia/ARDY-G1-RP-25FPS-Horizon8",
}
_ARTIFACT_FORMAT = "motius-ardy-wrapper-v1"


def _normalize_model_name(value: str) -> str:
    return ARDY_CHECKPOINTS.get(value.lower(), value)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    if "/" not in name_or_path:
        return local
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


def _copy_tree(src: Path, dst: Path, copy_mode: str = "copy") -> None:
    if not src.exists():
        raise FileNotFoundError(f"ARDY artifact source does not exist: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    copy_function = shutil.copy2
    if copy_mode == "hardlink":
        def copy_function(src, dst):
            try:
                os.link(Path(src).resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
    elif copy_mode != "copy":
        raise ValueError(f"Unsupported ARDY artifact copy mode: {copy_mode}")
    shutil.copytree(src, dst, symlinks=False, copy_function=copy_function)


def _is_local_relative_path(value: Optional[str]) -> bool:
    if not value:
        return False
    path = str(value)
    return "://" not in path and not Path(path).is_absolute()


def _resolve_relative_path(value: Optional[str], base: Path) -> Optional[str]:
    if not _is_local_relative_path(value):
        return value
    return str(base / str(value))


@MODEL_BUNDLES.register_module()
class ARDYBundle(ModelBundle):
    """Load and own one released ARDY denoiser/tokenizer pair.

    The official text encoder is gated LLM2Vec/Llama 3. Pass
    ``text_encoder=False`` when supplying precomputed ``text_feat`` tensors to
    the pipeline, or pass a compatible callable encoder instance.
    """

    SUPPORTED_TASKS = {
        "text_to_motion": "offline autoregressive text-to-motion generation",
        "streaming_text_to_motion": "stateful one-horizon generation with online prompt changes",
        "kinematic_control": "root path, full-body keyframe, and sparse end-effector constraints",
    }
    CHECKPOINTS = ARDY_CHECKPOINTS

    def __init__(
        self,
        model_name: str = "core",
        device: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        checkpoints_dir: Optional[str] = None,
        text_encoder: Any = None,
        text_encoder_mode: str = "local",
        text_encoder_url: Optional[str] = None,
        text_encoder_fp32: bool = False,
        text_encoders_dir: Optional[str] = None,
        load_model: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model_name = _normalize_model_name(str(model_name))
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = checkpoint_path
        self.checkpoints_dir = checkpoints_dir
        self.text_encoder = text_encoder
        self.text_encoder_mode = text_encoder_mode
        self.text_encoder_url = text_encoder_url
        self.text_encoder_fp32 = bool(text_encoder_fp32)
        self.text_encoders_dir = text_encoders_dir
        self._model = None
        self._model_config = None
        if load_model:
            self.load_model()

    @property
    def model(self):
        return self.load_model()

    @property
    def skeleton(self):
        return self.model.skeleton

    @property
    def motion_rep(self):
        return self.model.motion_rep

    @property
    def fps(self) -> float:
        return float(self.motion_rep.fps)

    @property
    def horizon(self) -> int:
        return int(self.model.gen_horizon_len)

    @property
    def device(self) -> torch.device:
        model = self.model
        try:
            return next(model.parameters()).device
        except StopIteration:
            return torch.device(model.device)

    def load_model(self):
        if self._model is not None:
            return self._model

        try:
            from .network import load_model
        except ImportError as exc:
            if exc.name == "vector_quantize_pytorch":
                raise ImportError(
                    "ARDY requires the optional dependencies. Install with "
                    "`pip install -e '.[ardy]'`."
                ) from exc
            raise

        model_name = self.model_name
        if self.checkpoint_path:
            model_name = Path(self.checkpoint_path).name
        old_text_encoders_dir = os.environ.get("TEXT_ENCODERS_DIR")
        if self.text_encoders_dir:
            os.environ["TEXT_ENCODERS_DIR"] = str(Path(self.text_encoders_dir).resolve())
        try:
            self._model, self._model_config = load_model(
                model_name,
                device=self.device_name,
                text_encoder=self.text_encoder,
                text_encoder_mode=self.text_encoder_mode,
                text_encoder_url=self.text_encoder_url,
                text_encoder_fp32=self.text_encoder_fp32,
                return_config=True,
                checkpoints_dir=self.checkpoints_dir,
                model_path=self.checkpoint_path,
            )
        finally:
            if self.text_encoders_dir:
                if old_text_encoders_dir is None:
                    os.environ.pop("TEXT_ENCODERS_DIR", None)
                else:
                    os.environ["TEXT_ENCODERS_DIR"] = old_text_encoders_dir
        return self._model

    def to(self, *args, **kwargs):
        if self._model is not None:
            self._model.to(*args, **kwargs)
            if args:
                self._model.device = torch.device(args[0])
            elif kwargs.get("device") is not None:
                self._model.device = torch.device(kwargs["device"])
        return super().to(*args, **kwargs)

    @classmethod
    def _bundle_config_from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(str(pretrained_model_name_or_path))
        config = dict(kwargs)
        if path.is_dir():
            config.setdefault("checkpoint_path", str(path))
            config.setdefault("model_name", path.name)
        else:
            config.setdefault("model_name", _normalize_model_name(str(pretrained_model_name_or_path)))
        return config

    def save_pretrained(
        self,
        save_directory: str,
        *,
        include_checkpoint: bool = True,
        include_text_encoder: bool = False,
        checkpoint_source: Optional[str] = None,
        text_encoders_source: Optional[str] = None,
        checkpoint_subdir: str = "ardy_checkpoint",
        text_encoder_subdir: str = "text_encoders",
        copy_mode: str = "copy",
        **kwargs,
    ):
        """Save a HuggingFace-style Motius ARDY artifact.

        The ARDY denoiser/tokenizer checkpoint can be stored inside the
        artifact. The LLM2Vec/Llama text encoder is gated; by default the
        artifact records the required repos and expects the caller's
        HuggingFace token at load time. Set ``include_text_encoder=True`` only
        when ``text_encoders_source`` points at a local, license-compliant
        mirror of the full text encoder stack.
        """
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        config = {
            "model_name": self.model_name,
            "device": self.device_name,
            "checkpoint_path": self.checkpoint_path,
            "checkpoints_dir": self.checkpoints_dir,
            "text_encoder_mode": self.text_encoder_mode,
            "text_encoder_url": self.text_encoder_url,
            "text_encoder_fp32": self.text_encoder_fp32,
            "text_encoders_dir": self.text_encoders_dir,
        }
        artifacts = {}
        if include_checkpoint:
            if checkpoint_source:
                ckpt_src = Path(checkpoint_source)
            elif self.checkpoint_path:
                ckpt_src = Path(self.checkpoint_path)
            else:
                from huggingface_hub import snapshot_download

                ckpt_src = Path(snapshot_download(repo_id=self.model_name, repo_type="model"))
            ckpt_dst = save_dir / checkpoint_subdir
            _copy_tree(ckpt_src, ckpt_dst, copy_mode=copy_mode)
            config["checkpoint_path"] = checkpoint_subdir
            config["checkpoints_dir"] = None
            artifacts["ardy_checkpoint"] = checkpoint_subdir

        text_encoder_components = {
            "backend": "llm2vec",
            "base_model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "mntp_adapter": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
            "supervised_adapter": "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
            "requires_hf_auth": True,
        }
        if include_text_encoder:
            text_src = Path(text_encoders_source or self.text_encoders_dir or "")
            if not text_src.exists():
                raise FileNotFoundError(
                    "include_text_encoder=True requires a local text_encoders_source"
                )
            text_dst = save_dir / text_encoder_subdir
            _copy_tree(text_src, text_dst, copy_mode=copy_mode)
            config["text_encoders_dir"] = text_encoder_subdir
            text_encoder_components["stored_in_artifact"] = True
            text_encoder_components["path"] = text_encoder_subdir
            artifacts["text_encoders"] = text_encoder_subdir
        else:
            text_encoder_components["stored_in_artifact"] = False

        meta = {
            "model_type": "ardy",
            "format": _ARTIFACT_FORMAT,
            "bundle_class": "motius.models.ardy.ARDYBundle",
            "pipeline_class": "motius.pipelines.ardy.ARDYPipeline",
            "artifacts": artifacts,
            "text_encoder": text_encoder_components,
            "config": config,
        }
        (save_dir / "ardy_config.json").write_text(json.dumps(meta, indent=2) + "\n")
        (save_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "ARDYPipeline",
                    "_library_name": "motius",
                    "model_type": "ardy",
                    "format": _ARTIFACT_FORMAT,
                    "bundle_class": meta["bundle_class"],
                    "pipeline_class": meta["pipeline_class"],
                    "artifacts": artifacts,
                    "text_encoder": text_encoder_components,
                    "supported_tasks": self.SUPPORTED_TASKS,
                    "api": {
                        "from_pretrained": "motius.pipelines.ardy.ARDYPipeline.from_pretrained",
                    },
                },
                indent=2,
            )
            + "\n"
        )
        readme = save_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "---\n"
                "library_name: motius\n"
                "tags:\n"
                "- motion-generation\n"
                "- text-to-motion\n"
                "- ardy\n"
                "- kinematic-control\n"
                "license: other\n"
                "---\n\n"
                "# ARDY Motius Artifact\n\n"
                "Load with `ARDYPipeline.from_pretrained(...)`. The ARDY "
                "checkpoint is stored in this artifact. The official text "
                "encoder uses gated LLM2Vec/Llama repositories; make sure the "
                "runtime has a Hugging Face token with access to "
                "`meta-llama/Meta-Llama-3-8B-Instruct`.\n\n"
                "```python\n"
                "from motius.pipelines.ardy import ARDYPipeline\n\n"
                "pipe = ARDYPipeline.from_pretrained(\"<artifact-path-or-repo>\", bundle_kwargs={\"device\": \"cuda\"})\n"
                "out = pipe.text_to_motion(\"a person walks forward and waves.\", 120, num_denoising_steps=4)\n"
                "```\n"
            )
        return save_directory

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, **kwargs):
        base_dir = None
        if isinstance(cfg, (str, Path)):
            cfg_path = Path(cfg)
            if cfg_path.is_dir():
                cfg_path = cfg_path / "ardy_config.json"
            base_dir = cfg_path.parent
            cfg = _read_json(cfg_path)
        cfg_dict = cls._to_plain_dict(cfg)
        if cfg_dict.get("model_type") == "ardy" and "config" in cfg_dict:
            cfg_dict = dict(cfg_dict["config"])
        cfg_dict.pop("format", None)
        cfg_dict.pop("artifacts", None)
        cfg_dict.pop("text_encoder_components", None)
        if base_dir is not None:
            cfg_dict["checkpoint_path"] = _resolve_relative_path(
                cfg_dict.get("checkpoint_path"),
                base_dir,
            )
            cfg_dict["checkpoints_dir"] = _resolve_relative_path(
                cfg_dict.get("checkpoints_dir"),
                base_dir,
            )
            cfg_dict["text_encoders_dir"] = _resolve_relative_path(
                cfg_dict.get("text_encoders_dir"),
                base_dir,
            )
        return super().from_config(cfg_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(str(pretrained_model_name_or_path))
        if not (path / "ardy_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "ardy_config.json"
        if cfg_file.exists():
            return cls.from_config(cfg_file, **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - pipeline owns sampling
        raise NotImplementedError("Use ARDYPipeline for autoregressive inference")


__all__ = ["ARDYBundle", "ARDY_CHECKPOINTS"]
