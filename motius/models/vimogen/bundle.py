"""ViMoGen ModelBundle.

The official ViMoGen evaluator entry is tightly coupled to distributed/FSDP
runtime state. This bundle keeps the released network and scheduler inside the
Motius model-zoo path while exposing a deterministic, upstream-checkout-free
T2M inference API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "vimogen" / "motius_1_3b"
_DEFAULT_WAN_DIR = _REPO_ROOT / "checkpoints" / "Wan2.1-T2V-1.3B"
_DEFAULT_WAN_REPO_ID = "Wan-AI/Wan2.1-T2V-1.3B"
_DEFAULT_CONTEXT_NULL = (
    Path(__file__).resolve().parent
    / "network"
    / "vimogen"
    / "models"
    / "transformer"
    / "wan"
    / "context_null_padded.pth"
)


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    """Resolve a Hugging Face Hub repo id to a local snapshot directory."""
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))
    except Exception:
        return local


def _read_model_index(artifact: Path) -> dict:
    path = artifact / "model_index.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _artifact_path(value: str | Path, artifact: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = artifact / path
    return path.resolve()


def _resolve_wan_dir(
    wan_dir: Optional[str],
    artifact: Path,
    index: dict,
) -> Path:
    if wan_dir:
        path = Path(wan_dir)
        if path.is_absolute():
            return path.resolve()
        for base in (artifact, _REPO_ROOT):
            candidate = (base / path).resolve()
            if (candidate / "config.json").exists():
                return candidate
        return (artifact / path).resolve()

    indexed = index.get("wan_dir") or index.get("wan_path")
    if indexed:
        return _resolve_wan_dir(str(indexed), artifact, {})

    for candidate in (
        artifact / "Wan2.1-T2V-1.3B",
        artifact / "wan",
        _DEFAULT_WAN_DIR,
    ):
        if (candidate / "config.json").exists():
            return candidate.resolve()

    repo_id = str(index.get("wan_repo_id") or _DEFAULT_WAN_REPO_ID)
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=repo_id, repo_type="model")).resolve()
    except Exception:
        return _DEFAULT_WAN_DIR.resolve()


def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    key = str(dtype).lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype!r}")
    return mapping[key]


def _resolve_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return device


@MODEL_BUNDLES.register_module()
class ViMoGenBundle(ModelBundle):
    """ViMoGen text-to-motion bundle for the released HumanML3D checkpoint.

    The model emits ViMoGen's 276D global DART-style motion representation.
    Returned tensors are denormalized, matching the official saved
    ``motion_gen_condition_on_text.pt`` semantics.
    """

    motion_dim = 276
    ref_motion_dim = 138

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        checkpoint: Optional[str] = None,
        wan_dir: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        context_null_path: Optional[str] = None,
        device: str = "cuda",
        dtype: str | torch.dtype = "bf16",
        text_dtype: str | torch.dtype | None = None,
        load_text_encoder: bool = True,
        text_len: int = 512,
        cfg_scale: float = 5.0,
        denoising_strength: float = 0.7,
        num_inference_steps: int = 50,
        min_length: int = 40,
        max_length: int = 200,
        **kwargs,
    ):
        super().__init__()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
        index = _read_model_index(artifact)
        checkpoint = _artifact_path(checkpoint or index.get("checkpoint") or "model.pt", artifact)
        wan_dir = _resolve_wan_dir(wan_dir, artifact, index)
        mean_path = _artifact_path(
            mean_path or index.get("mean_path") or "assets/meta/mean.npy",
            artifact,
        )
        std_path = _artifact_path(
            std_path or index.get("std_path") or "assets/meta/std.npy",
            artifact,
        )
        context_null_path = Path(context_null_path or _DEFAULT_CONTEXT_NULL).resolve()

        if not checkpoint.exists():
            raise FileNotFoundError(f"ViMoGen checkpoint not found: {checkpoint}")
        if not (wan_dir / "config.json").exists():
            raise FileNotFoundError(f"Wan base config not found: {wan_dir / 'config.json'}")
        if not mean_path.exists() or not std_path.exists():
            raise FileNotFoundError(f"ViMoGen mean/std not found: {mean_path}, {std_path}")

        self.artifact_dir = artifact
        self.checkpoint = checkpoint
        self.wan_dir = wan_dir
        self.context_null_path = context_null_path
        self.cfg_scale = float(cfg_scale)
        self.denoising_strength = float(denoising_strength)
        self.num_inference_steps = int(num_inference_steps)
        self.min_length = int(min_length)
        self.max_length = int(max_length)
        self.text_len = int(text_len)

        self._device = _resolve_device(device)
        self.dtype = _torch_dtype(dtype)
        if self._device.type == "cpu" and self.dtype in (torch.float16, torch.bfloat16):
            # CPU inference is not supported by the Wan blocks anyway; this
            # keeps construction/debug import paths predictable.
            self.dtype = torch.float32
        self.text_dtype = _torch_dtype(text_dtype or self.dtype)

        mean = np.load(str(mean_path)).astype(np.float32)
        std = np.load(str(std_path)).astype(np.float32)
        if mean.shape != (self.motion_dim,) or std.shape != (self.motion_dim,):
            raise ValueError(f"Expected 276D mean/std, got {mean.shape} and {std.shape}")
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

        context_null = torch.load(
            str(context_null_path),
            map_location="cpu",
            weights_only=True,
        )
        if context_null.ndim == 2:
            context_null = context_null.unsqueeze(0)
        self.register_buffer("prompt_emb_null", context_null.float(), persistent=True)

        from motius.models.vimogen.network.vimogen.models.transformer import (
            get_transformer3d,
        )

        model = get_transformer3d(
            model_name="wanvideotm2m_1.3b",
            load_pretrain=True,
            patch_size=2,
            in_channel=self.motion_dim,
            base_repo=str(wan_dir),
            strict=False,
            model_kwargs=dict(
                dense_interval=1,
                rope_mode="naive",
                force_no_sincos_embed=True,
                i2v_mode=0,
                in_channels=self.motion_dim,
                ref_motion_dim=self.ref_motion_dim,
                load_path=str(checkpoint),
            ),
        )
        model.eval().requires_grad_(False)
        self.model = model.to(device=self._device, dtype=self.dtype)
        self.text_encoder = None
        if load_text_encoder:
            self.load_text_encoder()
        self.to_device(self._device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / "model.pt").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        if path.is_dir() and (path / "model.pt").exists():
            return cls(artifact_dir=str(path), **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        index = {
            "model_type": "vimogen",
            "checkpoint": "model.pt",
            "mean_path": "assets/meta/mean.npy",
            "std_path": "assets/meta/std.npy",
            "motion_representation": "vimogen276",
            "cfg_scale": self.cfg_scale,
            "denoising_strength": self.denoising_strength,
            "num_inference_steps": self.num_inference_steps,
            "text_encoder": "Wan2.1-T2V-1.3B UMT5-XXL",
            "wan_repo_id": _DEFAULT_WAN_REPO_ID,
        }
        (save_dir / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")

    def to_device(self, device: str | torch.device):
        device = _resolve_device(device)
        self._device = device
        self.model.to(device=device, dtype=self.dtype)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.prompt_emb_null = self.prompt_emb_null.to(device=device, dtype=self.text_dtype)
        if self.text_encoder is not None:
            self.text_encoder.device = device
            self.text_encoder.model.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    def load_text_encoder(self):
        if self.text_encoder is not None:
            return self.text_encoder
        if self.device.type != "cuda":
            raise RuntimeError("ViMoGen text encoder inference requires a CUDA device.")
        from motius.models.vimogen.network.vimogen.models.transformer.wan.modules.t5 import (
            T5EncoderModel,
        )

        self.text_encoder = T5EncoderModel(
            text_len=self.text_len,
            dtype=self.text_dtype,
            device=self.device,
            checkpoint_path=str(self.wan_dir / "models_t5_umt5-xxl-enc-bf16.pth"),
            tokenizer_path=str(self.wan_dir / "google" / "umt5-xxl"),
            shard_fn=None,
        )
        return self.text_encoder

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.std.to(motion) + self.mean.to(motion)

    def clamp_length(self, n_frames: int) -> int:
        n_frames = int(n_frames)
        return max(self.min_length, min(self.max_length, n_frames))

    def _collate_contexts(self, contexts: List[torch.Tensor], min_len: int) -> torch.Tensor:
        max_len = max([min_len] + [int(x.shape[0]) for x in contexts])
        hidden = int(contexts[0].shape[-1])
        out = torch.zeros(
            len(contexts),
            max_len,
            hidden,
            device=self.device,
            dtype=self.text_dtype,
        )
        for i, ctx in enumerate(contexts):
            ctx = ctx.to(device=self.device, dtype=self.text_dtype)
            out[i, : ctx.shape[0]] = ctx
        return out

    def encode_texts(self, captions: Sequence[str]) -> torch.Tensor:
        encoder = self.load_text_encoder()
        contexts = encoder(list(captions), self.device)
        return self._collate_contexts(contexts, min_len=self.prompt_emb_null.shape[1])

    def null_context(self, batch_size: int, context_len: int) -> torch.Tensor:
        null = self.prompt_emb_null.to(device=self.device, dtype=self.text_dtype)
        if null.shape[1] < context_len:
            pad = torch.zeros(
                null.shape[0],
                context_len - null.shape[1],
                null.shape[2],
                device=self.device,
                dtype=self.text_dtype,
            )
            null = torch.cat([null, pad], dim=1)
        return null[:, :context_len].repeat(batch_size, 1, 1)

    @torch.no_grad()
    def generate_motion276_from_embeddings(
        self,
        prompt_emb: torch.Tensor,
        lengths: Sequence[int],
        seed: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        denoising_strength: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        if prompt_emb.ndim != 3:
            raise ValueError(f"prompt_emb must be (B,L,C), got {tuple(prompt_emb.shape)}")
        if int(prompt_emb.shape[0]) != len(lengths):
            raise ValueError("prompt_emb batch and lengths must have equal length")
        if self.device.type != "cuda":
            raise RuntimeError("ViMoGen generation requires CUDA.")

        from tqdm import tqdm

        from motius.models.vimogen.network.vimogen.trainer.scheduler import (
            FlowMatchScheduler,
        )
        from motius.models.vimogen.network.vimogen.utils import smooth_motion_rep

        lengths = [self.clamp_length(x) for x in lengths]
        batch_size = len(lengths)
        max_len = max(lengths)
        cfg_scale = float(self.cfg_scale if cfg_scale is None else cfg_scale)
        num_inference_steps = int(
            self.num_inference_steps if num_inference_steps is None else num_inference_steps
        )
        denoising_strength = float(
            self.denoising_strength if denoising_strength is None else denoising_strength
        )

        prompt_emb = prompt_emb.to(device=self.device, dtype=self.text_dtype)
        prompt_emb_null = self.null_context(batch_size, prompt_emb.shape[1])

        latents_mask = torch.zeros(batch_size, max_len, device=self.device, dtype=self.dtype)
        for i, length in enumerate(lengths):
            latents_mask[i, :length] = 1
        ref_latents = torch.zeros(
            batch_size,
            max_len,
            self.ref_motion_dim,
            device=self.device,
            dtype=self.dtype,
        )
        ref_latents_mask = latents_mask.clone()
        attend_to_text_mask = torch.ones(batch_size, device=self.device, dtype=torch.bool)

        generator = torch.Generator(device=self.device)
        if seed is None:
            seed = int(torch.randint(0, 2**31 - 1, (1,)).item())
        generator.manual_seed(int(seed))
        xt = torch.randn(
            batch_size,
            max_len,
            self.motion_dim,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        scheduler = FlowMatchScheduler()
        scheduler.set_timesteps(
            num_inference_steps,
            training=False,
            denoising_strength=denoising_strength,
        )
        timesteps = scheduler.timesteps.to(self.device)

        latents_mask_input = torch.cat([latents_mask] * 2, dim=0)
        ref_latents_input = torch.cat([ref_latents, torch.zeros_like(ref_latents)], dim=0)
        ref_latents_mask_input = torch.cat([ref_latents_mask] * 2, dim=0)
        attend_to_text_mask_input = torch.cat([attend_to_text_mask] * 2, dim=0)
        context_input = torch.cat([prompt_emb, prompt_emb_null], dim=0)

        self.model.eval()
        iterator = tqdm(timesteps, desc="ViMoGen", disable=not show_progress)
        for timestep in iterator:
            with torch.amp.autocast(dtype=self.dtype, device_type=self.device.type):
                latent_model_input = torch.cat([xt] * 2, dim=0)
                noise_pred = self.model(
                    x=latent_model_input,
                    timestep=timestep.unsqueeze(0),
                    context=context_input,
                    x_mask=latents_mask_input,
                    ref_motion=ref_latents_input,
                    ref_motion_mask=ref_latents_mask_input,
                    use_gradient_checkpointing=False,
                    attend_to_text_mask=attend_to_text_mask_input,
                )
                noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + cfg_scale * (
                    noise_pred_cond - noise_pred_uncond
                )
                xt = scheduler.step(noise_pred, timestep, xt)

        outputs = []
        for i, length in enumerate(lengths):
            motion = smooth_motion_rep(xt[i, :length], kernel_size=5, sigma=1.0)
            outputs.append(self.denormalize(motion).float().detach().cpu())
        return outputs

    @torch.no_grad()
    def generate_motion276(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        seed: Optional[int] = None,
        cfg_scale: Optional[float] = None,
        num_inference_steps: Optional[int] = None,
        denoising_strength: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[torch.Tensor]:
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        prompt_emb = self.encode_texts(captions)
        return self.generate_motion276_from_embeddings(
            prompt_emb=prompt_emb,
            lengths=lengths,
            seed=seed,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            denoising_strength=denoising_strength,
            show_progress=show_progress,
        )

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use ViMoGenPipeline.infer_t2m for inference.")
