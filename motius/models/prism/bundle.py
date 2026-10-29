"""Motius bundle for the PRISM 1.0 and PRISM-KT releases."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, List, Optional, Union

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
        training: bool = False,
        latent_sample_method: str = "mode",
        load_model: bool = True,
    ):
        super().__init__()
        self.model_name = str(model_name)
        self.checkpoint_path = checkpoint_path
        self.device_name = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.transformer_dtype = _dtype(transformer_dtype)
        self.text_dtype = _dtype(text_dtype)
        self.default_kafs_mode = default_kafs_mode
        self.training_enabled = bool(training)
        self.latent_sample_method = str(latent_sample_method).lower()
        if self.latent_sample_method not in {"mode", "sample"}:
            raise ValueError("latent_sample_method must be 'mode' or 'sample'")
        self.artifact_dir: Optional[Path] = None
        self.artifact_config: dict = {}
        if load_model:
            self.load_model()

    def _configure_training_modules(self) -> None:
        if not self.training_enabled:
            return
        self.transformer.requires_grad_(True)
        self.vae.requires_grad_(False).eval()
        self.text_encoder.requires_grad_(False).eval()
        self._trainable_modules = ["transformer"]
        self._save_ckpt_modules = ["transformer"]
        self._frozen_modules = ["vae", "text_encoder"]
        self._module_checkpoint_formats["transformer"] = "full"

    @classmethod
    def _bundle_config_from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        from motius.configuration.prism import runtime_options

        root = Path(pretrained_model_name_or_path)
        if not (root / "motius_config.json").is_file() and kwargs.get("load_model", True):
            root = _resolve_artifact(pretrained_model_name_or_path)
        saved_options = runtime_options(root) if (root / "motius_config.json").is_file() else {}
        return {
            "model_name": str(pretrained_model_name_or_path),
            "checkpoint_path": str(pretrained_model_name_or_path),
            **saved_options,
            **kwargs,
        }

    @property
    def device(self) -> torch.device:
        if self.artifact_dir is None:
            return torch.device(self.device_name)
        return next(self.transformer.parameters()).device

    @property
    def variant(self) -> str:
        return str(self.artifact_config.get("variant", self.model_name))

    def load_model(self):
        if self.artifact_dir is not None:
            return self

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

        from transformers import UMT5EncoderModel

        from motius.models.prism.autoencoder_kl_2d import AutoencoderKLPrism2DTK
        from motius.models.prism.network import PrismTransformerMotionModel

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
        text_encoder = UMT5EncoderModel.from_pretrained(
            root / components.get("text_encoder", "text_encoder"),
            torch_dtype=self.text_dtype,
            low_cpu_mem_usage=True,
        ).to(device)
        self.transformer = transformer
        self.vae = vae
        self.text_encoder = text_encoder
        self.artifact_dir = root
        self.artifact_config = config
        self._configure_training_modules()
        return self







    def encode_motion_parameters(self, channels):
        self.load_model()
        return self.vae.encode(channels)

    def encode_text_tokens(self, input_ids, attention_mask):
        self.load_model()
        return self.text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def predict_flow(self, **inputs):
        self.load_model()
        return self.transformer(**inputs)

    def save_pretrained(self, save_directory: str, **kwargs):
        self.load_model()
        output = Path(save_directory)
        output.mkdir(parents=True, exist_ok=True)
        self.transformer.save_pretrained(output / "transformer", **kwargs)
        self.vae.save_pretrained(output / "vae", **kwargs)
        self.text_encoder.save_pretrained(output / "text_encoder", **kwargs)
        components = self.artifact_config.get("components", {})
        for name in ("tokenizer", "scheduler"):
            source = self.artifact_dir / components.get(name, name)
            target = output / name
            if source.resolve() != target.resolve():
                shutil.copytree(source, target, dirs_exist_ok=True)
        stats_source = self.artifact_dir / self.artifact_config.get("stats_file", "motion_stats.json")
        if stats_source.resolve() != (output / "motion_stats.json").resolve():
            shutil.copy2(stats_source, output / "motion_stats.json")
        config = dict(self.artifact_config)
        config.update(
            artifact_format="motius-prism-v1",
            stats_file="motion_stats.json",
            runtime_options=dict(
                transformer_dtype=str(self.transformer_dtype).removeprefix("torch."),
                text_dtype=str(self.text_dtype).removeprefix("torch."),
                default_kafs_mode=self.default_kafs_mode,
                latent_sample_method=self.latent_sample_method,
            ),
            components={
                "transformer": "transformer",
                "vae": "vae",
                "text_encoder": "text_encoder",
                "tokenizer": "tokenizer",
                "scheduler": "scheduler",
            },
        )
        (output / "motius_config.json").write_text(json.dumps(config, indent=2) + "\n")
        from motius.configuration.prism import write_pipeline_index
        write_pipeline_index(output)
        return output
