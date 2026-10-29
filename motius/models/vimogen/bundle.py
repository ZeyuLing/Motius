"""ViMoGen ModelBundle.

The official ViMoGen evaluator entry is tightly coupled to distributed/FSDP
runtime state. This bundle keeps the released network and scheduler inside the
Motius model-zoo path while exposing a deterministic, upstream-checkout-free
T2M inference API.
"""
from __future__ import annotations
import json
from pathlib import Path
import shutil
from typing import List, Optional, Sequence
import numpy as np
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / 'checkpoints' / 'vimogen' / 'motius_1_3b'

def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    """Resolve a Hugging Face Hub repo id to a local snapshot directory."""
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(repo_id=name_or_path, repo_type='model'))
    except Exception:
        return local

def _read_model_index(artifact: Path) -> dict:
    path = artifact / 'model_index.json'
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

def _is_complete_wan_runtime(path: Path) -> bool:
    return (path / 'config.json').is_file() and (path / 'models_t5_umt5-xxl-enc-bf16.pth').is_file() and (path / 'google' / 'umt5-xxl' / 'tokenizer_config.json').is_file()

def _resolve_wan_dir(wan_dir: Optional[str], artifact: Path, index: dict) -> Path:
    if wan_dir:
        path = Path(wan_dir)
        if path.is_absolute():
            resolved = path.resolve()
            if _is_complete_wan_runtime(resolved):
                return resolved
            raise FileNotFoundError(f'Incomplete ViMoGen Wan runtime: {resolved}')
        for base in (artifact, _REPO_ROOT):
            candidate = (base / path).resolve()
            if _is_complete_wan_runtime(candidate):
                return candidate
        raise FileNotFoundError(f'Incomplete ViMoGen Wan runtime: {artifact / path}')
    indexed = index.get('wan_dir') or index.get('wan_path')
    if indexed:
        return _resolve_wan_dir(str(indexed), artifact, {})
    for candidate in (artifact / 'wan', artifact / 'Wan2.1-T2V-1.3B'):
        if _is_complete_wan_runtime(candidate):
            return candidate.resolve()
    raise FileNotFoundError("The ViMoGen artifact is incomplete: missing the bundled Wan config and UMT5 text encoder under 'wan/'. Pass wan_dir explicitly only for raw checkpoint conversion.")

def _torch_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    mapping = {'fp32': torch.float32, 'float32': torch.float32, 'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16, 'fp16': torch.float16, 'float16': torch.float16}
    key = str(dtype).lower()
    if key not in mapping:
        raise ValueError(f'Unsupported dtype: {dtype!r}')
    return mapping[key]

def _resolve_device(device: str | torch.device) -> torch.device:
    device = torch.device(device)
    if device.type == 'cuda' and (not torch.cuda.is_available()):
        return torch.device('cpu')
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

    def __init__(self, artifact_dir: Optional[str]=None, checkpoint: Optional[str]=None, wan_dir: Optional[str]=None, mean_path: Optional[str]=None, std_path: Optional[str]=None, context_null_path: Optional[str]=None, device: str='cuda', dtype: str | torch.dtype='bf16', text_dtype: str | torch.dtype | None=None, load_text_encoder: bool=True, text_len: int=512, cfg_scale: float=5.0, denoising_strength: float=0.7, num_inference_steps: int=50, min_length: int=40, max_length: int=200, **kwargs):
        super().__init__()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
        index = _read_model_index(artifact)
        checkpoint = _artifact_path(checkpoint or index.get('checkpoint') or 'model.pt', artifact)
        wan_dir = _resolve_wan_dir(wan_dir, artifact, index)
        mean_path = _artifact_path(mean_path or index.get('mean_path') or 'assets/meta/mean.npy', artifact)
        std_path = _artifact_path(std_path or index.get('std_path') or 'assets/meta/std.npy', artifact)
        context_null_path = _artifact_path(context_null_path or index.get('context_null_path') or 'context_null_padded.pth', artifact)
        if not checkpoint.exists():
            raise FileNotFoundError(f'ViMoGen checkpoint not found: {checkpoint}')
        if not (wan_dir / 'config.json').exists():
            raise FileNotFoundError(f"Wan base config not found: {wan_dir / 'config.json'}")
        if not mean_path.exists() or not std_path.exists():
            raise FileNotFoundError(f'ViMoGen mean/std not found: {mean_path}, {std_path}')
        if not context_null_path.exists():
            raise FileNotFoundError(f'ViMoGen null text context not found: {context_null_path}')
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
        if self._device.type == 'cpu' and self.dtype in (torch.float16, torch.bfloat16):
            self.dtype = torch.float32
        self.text_dtype = _torch_dtype(text_dtype or self.dtype)
        mean = np.load(str(mean_path)).astype(np.float32)
        std = np.load(str(std_path)).astype(np.float32)
        if mean.shape != (self.motion_dim,) or std.shape != (self.motion_dim,):
            raise ValueError(f'Expected 276D mean/std, got {mean.shape} and {std.shape}')
        self.register_buffer('mean', torch.from_numpy(mean), persistent=True)
        self.register_buffer('std', torch.from_numpy(std), persistent=True)
        context_null = torch.load(str(context_null_path), map_location='cpu', weights_only=True)
        if context_null.ndim == 2:
            context_null = context_null.unsqueeze(0)
        self.register_buffer('prompt_emb_null', context_null.float(), persistent=True)
        from .motion_network import WanModelTM2M
        architecture = json.loads((wan_dir / 'config.json').read_text())
        for key in ['_class_name', '_diffusers_version']:
            architecture.pop(key, None)
        architecture.update(in_dim=self.motion_dim, out_dim=self.motion_dim, ref_motion_dim=self.ref_motion_dim)
        model = WanModelTM2M(**architecture)
        model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True), strict=True)
        model.eval().requires_grad_(False)
        self.model = model.to(device=self._device, dtype=self.dtype)
        self.text_encoder = None
        if load_text_encoder:
            self.load_text_encoder()
        self.to_device(self._device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / 'model.pt').exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        if path.is_dir() and (path / 'model.pt').exists():
            return cls(artifact_dir=str(path), **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        include_wan = bool(kwargs.pop('include_wan', True))
        torch.save({key: value.detach().cpu() for key, value in self.unwrap_network(self.model).state_dict().items()}, save_dir / 'model.pt')
        stats_dir = save_dir / 'assets' / 'meta'
        stats_dir.mkdir(parents=True, exist_ok=True)
        np.save(stats_dir / 'mean.npy', self.mean.detach().cpu().numpy())
        np.save(stats_dir / 'std.npy', self.std.detach().cpu().numpy())
        torch.save(self.prompt_emb_null.detach().cpu(), save_dir / 'context_null_padded.pth')
        if include_wan:
            wan_dir = save_dir / 'wan'
            (wan_dir / 'google' / 'umt5-xxl').mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.wan_dir / 'config.json', wan_dir / 'config.json')
            shutil.copy2(self.wan_dir / 'models_t5_umt5-xxl-enc-bf16.pth', wan_dir / 'models_t5_umt5-xxl-enc-bf16.pth')
            for source in (self.wan_dir / 'google' / 'umt5-xxl').iterdir():
                if source.is_file():
                    shutil.copy2(source, wan_dir / 'google' / 'umt5-xxl' / source.name)
        index = {'model_type': 'vimogen', 'checkpoint': 'model.pt', 'mean_path': 'assets/meta/mean.npy', 'std_path': 'assets/meta/std.npy', 'context_null_path': 'context_null_padded.pth', 'motion_representation': 'vimogen276', 'cfg_scale': self.cfg_scale, 'denoising_strength': self.denoising_strength, 'num_inference_steps': self.num_inference_steps, 'text_encoder': 'Wan2.1-T2V-1.3B UMT5-XXL', 'wan_dir': 'wan' if include_wan else None, 'text_encoder_stored_in_artifact': include_wan}
        (save_dir / 'model_index.json').write_text(json.dumps(index, indent=2) + '\n')

    def to_device(self, device: str | torch.device):
        device = _resolve_device(device)
        self._device = device
        self.model.to(device=device, dtype=self.dtype)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        self.prompt_emb_null = self.prompt_emb_null.to(device=device, dtype=self.text_dtype)
        if self.text_encoder is not None:
            self.text_encoder.to(device=device, dtype=self.text_dtype)
        return self

    @property
    def device(self) -> torch.device:
        return self._device

    def load_text_encoder(self):
        if self.text_encoder is not None:
            return self.text_encoder
        from .text_network import T5Encoder
        with torch.device(self.device):
            model = T5Encoder(vocab=256384, dim=4096, dim_attn=4096, dim_ffn=10240, num_heads=64, num_layers=24, num_buckets=32, shared_pos=False, dropout=0.1)
        model = model.to(device=self.device, dtype=self.text_dtype).eval().requires_grad_(False)
        model.load_state_dict(torch.load(self.wan_dir / 'models_t5_umt5-xxl-enc-bf16.pth', map_location='cpu', weights_only=True, mmap=True), strict=True)
        self.text_encoder = model
        return model

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @staticmethod
    def unwrap_network(model):
        while hasattr(model, 'module'):
            model = model.module
        return model

    def predict_velocity(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def encode_text_tokens(self, *args, **kwargs):
        if self.text_encoder is None:
            self.load_text_encoder()
        return self.text_encoder(*args, **kwargs)
