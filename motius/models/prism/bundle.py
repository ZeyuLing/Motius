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


def _get_sigmas(scheduler, timesteps: torch.Tensor, n_dim: int) -> torch.Tensor:
    sigmas = scheduler.sigmas.to(device=timesteps.device, dtype=torch.float32)
    schedule = scheduler.timesteps.to(device=timesteps.device)
    indices = [(schedule == timestep).nonzero().item() for timestep in timesteps]
    sigma = sigmas[indices].flatten()
    while sigma.ndim < n_dim:
        sigma = sigma.unsqueeze(-1)
    return sigma


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
        self.backend = None
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
        if hasattr(self.scheduler, "set_timesteps"):
            self.scheduler.set_timesteps(self.scheduler.config.num_train_timesteps)

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
        self._configure_training_modules()
        return self.backend

    @torch.no_grad()
    def encode_motion(self, motion: torch.Tensor) -> torch.Tensor:
        """Normalize a 138D motion and encode it with the frozen fp32 VAE."""
        from motius.models.prism.gaussian_distribution import (
            DiagonalGaussianDistributionNd,
        )

        self.load_model()
        motion = motion.float()
        if motion.ndim == 2:
            motion = motion.unsqueeze(0)
        motion = self.processor.normalize(motion)
        motion = motion.reshape(*motion.shape[:2], 23, 6)
        with torch.autocast(motion.device.type, enabled=False):
            parameters = self.vae.encode(motion.float())
        posterior = DiagonalGaussianDistributionNd(parameters)
        latents = (
            posterior.sample()
            if self.latent_sample_method == "sample"
            else posterior.mode()
        )
        latents = (
            latents - self.backend.latents_mean.to(latents)
        ) / self.backend.latents_std.to(latents)
        stats = latents.detach().float()
        channel_mean = stats.mean(dim=(0, 2, 3))
        channel_std = stats.std(dim=(0, 2, 3), unbiased=False)
        self._last_latent_norm_stats = {
            "latent_norm_mean": stats.mean(),
            "latent_norm_std": stats.std(unbiased=False),
            "latent_norm_channel_mean_abs_max": channel_mean.abs().max(),
            "latent_norm_channel_std_min": channel_std.min(),
            "latent_norm_channel_std_max": channel_std.max(),
        }
        return latents

    @torch.no_grad()
    def encode_prompt_with_mask(
        self,
        prompt: Union[str, List[str]],
        max_sequence_length: int = 128,
        prompt_drop_rate: float = 0.0,
        dtype: Optional[torch.dtype] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.load_model()
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if prompt_drop_rate > 0:
            prompts = [
                "" if torch.rand(()).item() < prompt_drop_rate else value
                for value in prompts
            ]
        device = next(self.text_encoder.parameters()).device
        dtype = dtype or next(self.text_encoder.parameters()).dtype
        tokens = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        attention_mask = tokens.attention_mask.to(device)
        encoded = self.text_encoder(
            input_ids=tokens.input_ids.to(device),
            attention_mask=attention_mask,
        ).last_hidden_state
        return encoded.to(dtype=dtype), attention_mask

    def create_padding_mask(
        self,
        num_frames: Optional[torch.Tensor],
        batch_size: int,
        latent_frames: int,
        latent_joints: int,
        device: torch.device,
    ) -> torch.Tensor:
        if num_frames is None:
            return torch.ones(batch_size, latent_frames, latent_joints, device=device)
        lengths = torch.as_tensor(num_frames, device=device).long()
        scale = int(self.vae.config.scale_factor_temporal)
        lengths = ((lengths + scale - 1) // scale).clamp(0, latent_frames)
        valid = torch.arange(latent_frames, device=device)[None] < lengths[:, None]
        return valid.unsqueeze(-1).expand(-1, -1, latent_joints).float()

    def create_condition_mask(
        self,
        latents: torch.Tensor,
        frame_condition_rate: float = 0.1,
        condition_num_frames: Union[int, List[int]] = 1,
        num_frames: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, _, latent_frames, latent_joints = latents.shape
        device = latents.device
        mask = torch.ones(
            batch_size, 1, latent_frames, latent_joints,
            dtype=torch.bool,
            device=device,
        )
        if frame_condition_rate <= 0:
            return mask
        candidates = (
            [condition_num_frames]
            if isinstance(condition_num_frames, int)
            else list(condition_num_frames)
        )
        selected = torch.tensor(candidates, device=device)[
            torch.randint(len(candidates), (batch_size,), device=device)
        ]
        scale = int(self.vae.config.scale_factor_temporal)
        selected = ((selected + scale - 1) // scale).clamp(0, latent_frames)
        selected *= (torch.rand(batch_size, device=device) < frame_condition_rate)
        frames = torch.arange(latent_frames, device=device)[None]
        condition = frames < selected[:, None]
        mask = (~condition).unsqueeze(1).unsqueeze(-1).expand_as(mask).clone()
        if num_frames is not None:
            valid = torch.as_tensor(num_frames, device=device).long()
            valid = ((valid + scale - 1) // scale).clamp(0, latent_frames)
            padded = frames >= valid[:, None]
            mask |= padded.unsqueeze(1).unsqueeze(-1).expand_as(mask)
        return mask

    @staticmethod
    def create_sequence_ts(
        timesteps: torch.Tensor,
        condition_mask: torch.Tensor,
        patch_size=(1, 1),
    ) -> torch.Tensor:
        patch_t, patch_j = patch_size
        sampled_mask = condition_mask[:, 0, ::patch_t, ::patch_j]
        expanded = timesteps[:, None, None].expand_as(sampled_mask)
        return torch.where(sampled_mask, expanded, 0).flatten(1)

    def add_flow_noise(
        self, latents: torch.Tensor, timesteps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(latents)
        sigmas = _get_sigmas(self.scheduler, timesteps, latents.ndim).to(latents)
        return (1 - sigmas) * latents + sigmas * noise, noise - latents

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
