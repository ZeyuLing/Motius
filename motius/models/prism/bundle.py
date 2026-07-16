"""Motius bundle for the PRISM 1.0 and PRISM-KT releases."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Optional

import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


PRISM_CHECKPOINTS = {
    "1.0": "ZeyuLing/motius-prism-1.0-humanml3d",
    "prism-1.0": "ZeyuLing/motius-prism-1.0-humanml3d",
    "kt": "ZeyuLing/motius-prism-kt-humanml3d",
    "prism-kt": "ZeyuLing/motius-prism-kt-humanml3d",
}


def _dtype(value: Any) -> torch.dtype:
    if isinstance(value, torch.dtype):
        return value
    aliases = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return aliases[str(value).lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported PRISM dtype: {value!r}") from exc


def _resolve_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path).expanduser()
    if path.exists():
        return path.resolve()
    repo_id = PRISM_CHECKPOINTS.get(name_or_path.lower(), name_or_path)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=repo_id, repo_type="model"))


@MODEL_BUNDLES.register_module()
class PRISMBundle(ModelBundle):
    """Own the self-contained PRISM denoiser, VAE, text stack, and processor."""

    CHECKPOINTS = PRISM_CHECKPOINTS
    SUPPORTED_TASKS = {
        "T2M": "text-to-motion generation",
        "TP2M": "text-guided prefix-conditioned generation",
        "Sequential Generation": "autoregressive generation from ordered prompts",
    }

    def __init__(
        self,
        model_name: str = "kt",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        transformer_dtype: Any = "bf16",
        text_dtype: Any = "bf16",
        default_kafs_mode: Optional[str] = None,
        load_model: bool = True,
    ):
        super().__init__()
        self.model_name = str(model_name)
        self.checkpoint_path = checkpoint_path
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_dtype = _dtype(transformer_dtype)
        self.text_dtype = _dtype(text_dtype)
        self.default_kafs_mode = default_kafs_mode
        self.artifact_dir: Optional[Path] = None
        self.artifact_config: dict = {}
        self.backend = None
        if load_model:
            self.load_model()

    @classmethod
    def _bundle_config_from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return {
            "model_name": str(pretrained_model_name_or_path),
            "checkpoint_path": str(pretrained_model_name_or_path),
            **kwargs,
        }

    @property
    def device(self) -> torch.device:
        if self.backend is None:
            return torch.device(self.device_name)
        return next(self.transformer.parameters()).device

    @property
    def variant(self) -> str:
        return str(self.artifact_config.get("variant", self.model_name))

    def load_model(self):
        if self.backend is not None:
            return self.backend

        source = self.checkpoint_path or PRISM_CHECKPOINTS.get(
            self.model_name.lower(), self.model_name
        )
        root = _resolve_artifact(source)
        config_path = root / "motius_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(
                f"PRISM artifact is missing motius_config.json: {root}"
            )
        config = json.loads(config_path.read_text())
        if config.get("artifact_format") != "motius-prism-v1":
            raise ValueError(f"Unsupported PRISM artifact format: {config.get('artifact_format')}")

        from diffusers import FlowMatchEulerDiscreteScheduler
        from transformers import AutoTokenizer, UMT5EncoderModel

        from motius.models.prism.autoencoder_kl_2d import AutoencoderKLPrism2DTK
        from motius.models.prism.network import PrismTransformerMotionModel
        from motius.models.prism.processor import PRISMMotionProcessor
        from motius.pipelines.prism.backend import PrismARPipeline

        components = config.get("components", {})
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "PRISM was requested on CUDA, but torch.cuda.is_available() is false"
            )
        transformer = PrismTransformerMotionModel.from_pretrained(
            root / components.get("transformer", "transformer"),
            torch_dtype=self.transformer_dtype,
        ).to(device)
        vae = AutoencoderKLPrism2DTK.from_pretrained(
            root / components.get("vae", "vae"),
            torch_dtype=torch.float32,
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            root / components.get("tokenizer", "tokenizer")
        )
        text_encoder = UMT5EncoderModel.from_pretrained(
            root / components.get("text_encoder", "text_encoder"),
            torch_dtype=self.text_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            root / components.get("scheduler", "scheduler")
        )
        processor = PRISMMotionProcessor(root / config.get("stats_file", "motion_stats.json"))

        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.processor = processor
        object.__setattr__(self, "tokenizer", tokenizer)
        object.__setattr__(self, "scheduler", scheduler)
        self.backend = PrismARPipeline(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vae=vae,
            scheduler=scheduler,
            smpl_processor=processor,
            transformer=transformer,
            expand_timesteps=bool(config.get("expand_timesteps", True)),
            is_causal=bool(config.get("is_causal", False)),
        )
        kafs_mode = self.default_kafs_mode
        if kafs_mode is None:
            kafs_mode = str(config.get("default_kafs_mode", "none"))
        self.backend.set_kafs_alpha(kafs_mode)
        self.artifact_dir = root
        self.artifact_config = config
        return self.backend

    def save_pretrained(self, save_directory: str, **kwargs):
        self.load_model()
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(output / "transformer", **kwargs)
        self.vae.save_pretrained(output / "vae", **kwargs)
        self.text_encoder.save_pretrained(output / "text_encoder", **kwargs)
        self.tokenizer.save_pretrained(output / "tokenizer")
        self.scheduler.save_pretrained(output / "scheduler")
        shutil.copy2(self.processor.stats_file, output / "motion_stats.json")
        config = dict(self.artifact_config)
        config.update(
            artifact_format="motius-prism-v1",
            stats_file="motion_stats.json",
            components={
                "transformer": "transformer",
                "vae": "vae",
                "text_encoder": "text_encoder",
                "tokenizer": "tokenizer",
                "scheduler": "scheduler",
            },
        )
        (output / "motius_config.json").write_text(json.dumps(config, indent=2) + "\n")
        return output
