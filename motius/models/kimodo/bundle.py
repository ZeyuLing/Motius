"""KIMODO bundle wrapper backed by native Motius runtime modules.

Motius artifacts store the selected KIMODO checkpoint folder and local
LLM2Vec text encoders next to ``kimodo_config.json`` so ``from_pretrained`` can
run offline without importing an external source checkout.
"""

from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


# Repo root: motius/models/kimodo/bundle.py -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_ARTIFACT_FORMAT = "motius-kimodo-wrapper-v1"
_DEFAULT_MODEL = "Kimodo-SOMA-RP-v1"
_DEFAULT_CHECKPOINT_DIR = _REPO_ROOT / "checkpoints" / "kimodo" / "local_models"
_DEFAULT_TEXT_ENCODERS_DIR = _REPO_ROOT / "checkpoints" / "kimodo" / "text_encoders"


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    if "/" not in name_or_path:
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local


def _normalize_model_name(model_name: str) -> str:
    if "/" in model_name and model_name.lower().startswith("nvidia/"):
        return model_name.split("/", 1)[1]
    return model_name


def _is_local_relative_path(value: Optional[str]) -> bool:
    if not value:
        return False
    path = str(value)
    return "://" not in path and not Path(path).is_absolute()


def _resolve_relative_path(value: Optional[str], base: Path) -> Optional[str]:
    if not value or not _is_local_relative_path(value):
        return value
    return str(base / value)


def _copy_tree(
    src: Path,
    dst: Path,
    ignore_patterns: Optional[Sequence[str]] = None,
    copy_mode: str = "copy",
) -> None:
    if not src.exists():
        raise FileNotFoundError(f"KIMODO artifact source does not exist: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(*(ignore_patterns or ())) if ignore_patterns else None
    copy_function = shutil.copy2
    if copy_mode == "hardlink":
        def copy_function(src, dst):
            os.link(Path(src).resolve(), dst)
    elif copy_mode != "copy":
        raise ValueError(f"Unsupported copy_mode for KIMODO artifact: {copy_mode}")
    shutil.copytree(src, dst, symlinks=False, ignore=ignore, copy_function=copy_function)


def _patch_llm2vec_adapter_configs(text_encoders_dir: Optional[str]) -> None:
    """Point packaged LLM2Vec adapters at the packaged Meta-Llama base model.

    KIMODO's wrapper prepends ``TEXT_ENCODERS_DIR`` to the two LLM2Vec adapter
    repo names, but PEFT then reads ``adapter_config.json`` and follows its
    ``base_model_name_or_path`` verbatim. The upstream value is the gated
    ``meta-llama/Meta-Llama-3-8B-Instruct`` Hub id, so a supposedly local
    artifact still tries to hit the network unless we rewrite it to the
    artifact-local base model directory.
    """
    if not text_encoders_dir:
        return
    root = Path(text_encoders_dir)
    base_model = root / "meta-llama" / "Meta-Llama-3-8B-Instruct"
    if not base_model.exists():
        return
    for rel in (
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp/adapter_config.json",
        "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised/adapter_config.json",
    ):
        cfg_path = root / rel
        if not cfg_path.exists():
            continue
        try:
            cfg = _read_json(cfg_path)
        except Exception:
            continue
        local_base = str(base_model.resolve())
        if cfg.get("base_model_name_or_path") == local_base:
            continue
        cfg["base_model_name_or_path"] = local_base
        cfg_path.write_text(json.dumps(cfg, indent=2))


@contextmanager
def _patched_env(env: Dict[str, Optional[str]]):
    old = {}
    for key, value in env.items():
        old[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@MODEL_BUNDLES.register_module()
class KIMODOBundle(ModelBundle):
    """Thin Motius facade around NVIDIA KIMODO.

    Supported official control modes:

    - text-to-motion
    - multi-prompt transition/stitching
    - full-body pose keyframes
    - end-effector hand/foot position and rotation controls
    - 2-D root path / waypoint controls
    - direct loading of saved KIMODO constraint JSON
    """

    SUPPORTED_TASKS = {
        "text_to_motion": "prompt only, no constraints",
        "multi_prompt": "ordered prompt segments with transition blending",
        "fullbody_keyframe": "FullBodyConstraintSet on selected frames",
        "end_effector": "EndEffectorConstraintSet or hand/foot subclasses",
        "root2d": "Root2DConstraintSet for 2-D paths and waypoints",
        "constraint_json": "constraints saved by the KIMODO demo/CLI",
    }

    SUPPORTED_MODELS = {
        "Kimodo-SOMA-RP-v1",
        "Kimodo-G1-RP-v1",
        "Kimodo-SOMA-SEED-v1",
        "Kimodo-G1-SEED-v1",
        "Kimodo-SMPLX-RP-v1",
    }

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        device: Optional[str] = None,
        diffusion_steps: int = 100,
        cfg_type: Optional[str] = None,
        cfg_weight: Optional[Any] = None,
        post_processing: bool = False,
        local_cache: bool = True,
        hf_home: Optional[str] = None,
        checkpoint_dir: Optional[str] = None,
        text_encoder_mode: str = "auto",
        text_encoder: str = "llm2vec",
        text_encoders_dir: Optional[str] = None,
        text_encoders_repo: Optional[str] = None,
        text_encoders_subdir: str = "text_encoders",
        text_encoder_url: Optional[str] = None,
        load_model: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.model_name = _normalize_model_name(model_name)
        self.device_name = device
        self.diffusion_steps = int(diffusion_steps)
        self.cfg_type = cfg_type
        self.cfg_weight = cfg_weight
        self.post_processing = bool(post_processing)
        self.local_cache = bool(local_cache)
        self.hf_home = hf_home
        self.checkpoint_dir = checkpoint_dir
        self.text_encoder_mode = text_encoder_mode
        self.text_encoder = text_encoder
        self.text_encoders_dir = text_encoders_dir
        self.text_encoders_repo = text_encoders_repo
        self.text_encoders_subdir = text_encoders_subdir
        self.text_encoder_url = text_encoder_url
        self._model = None
        self._resolved_model_name = None

        if load_model:
            self.load_model()

    @property
    def model(self):
        return self.load_model()

    @property
    def skeleton(self):
        return self.model.skeleton

    @property
    def resolved_model_name(self) -> str:
        if self._resolved_model_name is None:
            self.load_model()
        return self._resolved_model_name or self.model_name

    def _kimodo_env(self) -> Dict[str, Optional[str]]:
        env = {
            "LOCAL_CACHE": "true" if self.local_cache else "false",
            "TEXT_ENCODER_MODE": self.text_encoder_mode,
            "TEXT_ENCODER": self.text_encoder,
        }
        if self.hf_home:
            env["HF_HOME"] = self.hf_home
            env["HUGGINGFACE_HUB_CACHE"] = str(Path(self.hf_home) / "hub")
            env["TRANSFORMERS_CACHE"] = str(Path(self.hf_home) / "hub")
        if self.checkpoint_dir:
            env["CHECKPOINT_DIR"] = self.checkpoint_dir
        if self.text_encoders_dir:
            env["TEXT_ENCODERS_DIR"] = self.text_encoders_dir
        if self.text_encoder_url:
            env["TEXT_ENCODER_URL"] = self.text_encoder_url
        return env

    def load_model(self):
        if self._model is not None:
            return self._model

        _patch_llm2vec_adapter_configs(self.text_encoders_dir)
        with _patched_env(self._kimodo_env()):
            from .network import load_model

            self._model, self._resolved_model_name = load_model(
                self.model_name,
                device=self.device_name,
                default_family="Kimodo",
                return_resolved_name=True,
            )
        return self._model

    def generation_kwargs(self, **overrides) -> dict:
        kwargs = {
            "num_denoising_steps": self.diffusion_steps,
            "post_processing": self.post_processing,
        }
        if self.cfg_type is not None:
            kwargs["cfg_type"] = self.cfg_type
        if self.cfg_weight is not None:
            kwargs["cfg_weight"] = deepcopy(self.cfg_weight)
        kwargs.update({k: v for k, v in overrides.items() if v is not None})
        return kwargs

    def generate(
        self,
        prompts,
        num_frames,
        constraints: Optional[list] = None,
        multi_prompt: bool = False,
        return_numpy: bool = True,
        **kwargs,
    ) -> dict:
        """Run KIMODO generation through the official low-level API."""
        call_kwargs = self.generation_kwargs(**kwargs)
        return self.model(
            prompts=prompts,
            num_frames=num_frames,
            multi_prompt=multi_prompt,
            constraint_lst=constraints or [],
            return_numpy=return_numpy,
            **call_kwargs,
        )

    def config_dict(self) -> dict:
        components = {
            "kimodo_checkpoint": {
                "model_name": self.model_name,
                "stored_in_artifact": False,
                "path": self.checkpoint_dir,
            },
            "text_encoder": {
                "name": self.text_encoder,
                "stored_in_artifact": False,
                "path": self.text_encoders_dir,
            },
        }
        return {
            "format": _ARTIFACT_FORMAT,
            "model_name": self.model_name,
            "device": self.device_name,
            "diffusion_steps": self.diffusion_steps,
            "cfg_type": self.cfg_type,
            "cfg_weight": deepcopy(self.cfg_weight),
            "post_processing": self.post_processing,
            "local_cache": self.local_cache,
            "hf_home": self.hf_home,
            "checkpoint_dir": self.checkpoint_dir,
            "text_encoder_mode": self.text_encoder_mode,
            "text_encoder": self.text_encoder,
            "text_encoders_dir": self.text_encoders_dir,
            "text_encoders_repo": self.text_encoders_repo,
            "text_encoders_subdir": self.text_encoders_subdir,
            "text_encoder_url": self.text_encoder_url,
            "supported_tasks": deepcopy(self.SUPPORTED_TASKS),
            "components": components,
            "external_components": components,
        }

    def _resolve_checkpoint_source(self, checkpoint_source: Optional[str] = None) -> Path:
        if checkpoint_source is not None:
            return Path(checkpoint_source)
        checkpoint_dir = Path(self.checkpoint_dir) if self.checkpoint_dir else _DEFAULT_CHECKPOINT_DIR
        candidates = [
            checkpoint_dir / self.model_name,
            checkpoint_dir / _normalize_model_name(self.model_name),
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            "Could not find KIMODO checkpoint folder. Tried: "
            + ", ".join(str(p) for p in candidates)
        )

    def save_pretrained(
        self,
        save_directory: str,
        *,
        include_weights: bool = True,
        include_text_encoder: bool = True,
        checkpoint_source: Optional[str] = None,
        text_encoders_source: Optional[str] = None,
        checkpoint_subdir: str = "kimodo_checkpoint",
        text_encoder_subdir: str = "text_encoders",
        copy_mode: str = "copy",
        **kwargs,
    ):
        """Save a self-contained KIMODO Motius artifact."""
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        config = self.config_dict()

        artifacts = {}
        if include_weights:
            ckpt_src = self._resolve_checkpoint_source(checkpoint_source)
            ckpt_dst = save_dir / checkpoint_subdir / self.model_name
            _copy_tree(ckpt_src, ckpt_dst, copy_mode=copy_mode)
            config["checkpoint_dir"] = checkpoint_subdir
            config["local_cache"] = True
            artifacts["kimodo_checkpoint"] = f"{checkpoint_subdir}/{self.model_name}"
            config["components"]["kimodo_checkpoint"] = {
                "model_name": self.model_name,
                "stored_in_artifact": True,
                "path": artifacts["kimodo_checkpoint"],
            }
            config["external_components"]["kimodo_checkpoint"] = config["components"]["kimodo_checkpoint"]

        if include_text_encoder:
            text_src = Path(text_encoders_source or self.text_encoders_dir or _DEFAULT_TEXT_ENCODERS_DIR)
            text_dst = save_dir / text_encoder_subdir
            # The Meta Llama HF snapshot may contain an ``original/`` export with
            # duplicate .pth weights. Transformers loads the safetensors shards.
            _copy_tree(
                text_src,
                text_dst,
                ignore_patterns=("original", ".cache"),
                copy_mode=copy_mode,
            )
            config["text_encoder_mode"] = "local"
            config["text_encoders_dir"] = text_encoder_subdir
            artifacts["text_encoders"] = text_encoder_subdir
            config["components"]["text_encoder"] = {
                "name": self.text_encoder,
                "stored_in_artifact": True,
                "path": text_encoder_subdir,
            }
            config["external_components"]["text_encoder"] = config["components"]["text_encoder"]
        elif self.text_encoders_repo:
            config["components"]["text_encoder"] = {
                "name": self.text_encoder,
                "stored_in_artifact": False,
                "repo": self.text_encoders_repo,
                "path": self.text_encoders_subdir,
            }
            config["external_components"]["text_encoder"] = config["components"]["text_encoder"]

        meta = {
            "model_type": "kimodo",
            "format": _ARTIFACT_FORMAT,
            "bundle_class": "motius.models.kimodo.bundle.KIMODOBundle",
            "pipeline_class": "motius.pipelines.kimodo.kimodo_pipeline.KIMODOPipeline",
            "artifacts": artifacts,
            "config": config,
        }
        (save_dir / "kimodo_config.json").write_text(json.dumps(meta, indent=2))
        (save_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "KIMODOPipeline",
                    "_library_name": "motius",
                    "model_type": "kimodo",
                    "format": _ARTIFACT_FORMAT,
                    "bundle_class": meta["bundle_class"],
                    "pipeline_class": meta["pipeline_class"],
                    "artifacts": artifacts,
                    "components": meta["config"]["components"],
                    "external_components": meta["config"]["external_components"],
                    "supported_tasks": self.SUPPORTED_TASKS,
                    "api": {
                        "from_pretrained": (
                            "motius.models.kimodo.KIMODOBundle"
                            ".from_pretrained"
                        ),
                        "from_config": (
                            "motius.models.kimodo.KIMODOBundle"
                            ".from_config"
                        ),
                    },
                },
                indent=2,
            )
        )
        readme = save_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "---\n"
                "library_name: motius\n"
                "tags:\n"
                "- motion-generation\n"
                "- text-to-motion\n"
                "- kimodo\n"
                "- kinematic-control\n"
                "license: other\n"
                "---\n\n"
                "# KIMODO Motius Artifact\n\n"
                "This artifact stores a self-contained Motius KIMODO runtime "
                "wrapper, including the selected KIMODO checkpoint folder and "
                "local LLM2Vec / Meta-Llama text encoder files. Load it with "
                "`KIMODOBundle.from_pretrained(...)` and run it through "
                "`KIMODOPipeline`.\n\n"
                "```python\n"
                "from motius.models.kimodo import KIMODOBundle\n"
                "from motius.pipelines.kimodo.kimodo_pipeline import KIMODOPipeline\n\n"
                "bundle = KIMODOBundle.from_pretrained(\"<artifact-path-or-repo>\", device=\"cuda\")\n"
                "pipe = KIMODOPipeline(bundle)\n"
                "out = pipe.text_to_motion(\"a person walks forward.\", num_frames=150)\n"
                "```\n\n"
                "You can also call `KIMODOPipeline.from_pretrained(...)` "
                "directly; it forwards arguments to `KIMODOBundle.from_pretrained`.\n\n"
                "The artifact also supports `multi_prompt`, `constraints_from_json`, "
                "`root2d_constraint`, `fullbody_keyframe_constraint`, and "
                "`end_effector_constraint`. See `docs/model_zoo/kimodo.md` in "
                "Motius for the full task surface and retargeting notes.\n"
            )
        return save_directory

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, **kwargs):
        base_dir = None
        if isinstance(cfg, (str, Path)):
            cfg_path = Path(cfg)
            if cfg_path.is_dir():
                cfg_path = cfg_path / "kimodo_config.json"
            base_dir = cfg_path.parent
            cfg = _read_json(cfg_path)
        cfg_dict = cls._to_plain_dict(cfg)
        if cfg_dict.get("model_type") == "kimodo" and "config" in cfg_dict:
            cfg_dict = deepcopy(cfg_dict["config"])
        cfg_dict.pop("format", None)
        cfg_dict.pop("supported_tasks", None)
        cfg_dict.pop("components", None)
        cfg_dict.pop("external_components", None)
        if base_dir is not None:
            cfg_dict["checkpoint_dir"] = _resolve_relative_path(
                cfg_dict.get("checkpoint_dir"),
                base_dir,
            )
            cfg_dict["text_encoders_dir"] = _resolve_relative_path(
                cfg_dict.get("text_encoders_dir"),
                base_dir,
            )
        text_encoders_repo = cfg_dict.get("text_encoders_repo")
        text_encoders_subdir = cfg_dict.get("text_encoders_subdir") or "text_encoders"
        text_encoders_dir = cfg_dict.get("text_encoders_dir")
        effective_text_encoder = kwargs.get("text_encoder", cfg_dict.get("text_encoder"))
        effective_text_encoder_mode = kwargs.get(
            "text_encoder_mode",
            cfg_dict.get("text_encoder_mode"),
        )
        should_resolve_text_encoders = (
            kwargs.get("load_model", True)
            and effective_text_encoder != "dummy"
            and effective_text_encoder_mode != "api"
        )
        if should_resolve_text_encoders and text_encoders_repo and (
            not text_encoders_dir or not Path(text_encoders_dir).exists()
        ):
            from huggingface_hub import snapshot_download

            snapshot = snapshot_download(
                repo_id=text_encoders_repo,
                repo_type="model",
                allow_patterns=(f"{text_encoders_subdir}/**",),
            )
            cfg_dict["text_encoders_dir"] = str(Path(snapshot) / text_encoders_subdir)
        return super().from_config(cfg_dict, **kwargs)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / "kimodo_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "kimodo_config.json"
        if cfg_file.exists():
            return cls.from_config(cfg_file, **kwargs)
        return cls(model_name=_normalize_model_name(str(pretrained_model_name_or_path)), **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover - use generate/pipeline
        return self.generate(*args, **kwargs)
