"""FlowMDM neural bundle and artifact loading."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.models.clip_artifact import artifact_clip_load
from motius.registry import MODEL_BUNDLES

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "flowmdm"


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    """Resolve a Hugging Face Hub model repo id to a local snapshot directory."""
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))
    except Exception:
        return local


def _load_tensor(path: Path) -> np.ndarray:
    """Load tensor-only statistics across supported PyTorch versions."""

    try:
        value = torch.load(str(path), map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch < 2.0 does not expose weights_only.
        value = torch.load(str(path), map_location="cpu")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _load_args(
    artifact_dir: Path,
    model_path: Path,
    device: str | int,
    seed: int,
    guidance_param: float,
    bpe_denoising_step: int,
    use_chunked_att: bool,
) -> Namespace:
    model_args = json.loads((artifact_dir / "args.json").read_text())
    model_args.update(
        {
            "model_path": str(model_path),
            "device": device,
            "seed": seed,
            "guidance_param": guidance_param,
            "bpe_denoising_step": bpe_denoising_step,
            "use_chunked_att": use_chunked_att,
        }
    )
    model_args.setdefault("dataset", "humanml")
    model_args.setdefault("unconstrained", False)
    model_args.setdefault("lambda_fc", 0.0)
    model_args.setdefault("lambda_rcxyz", 0.0)
    model_args.setdefault("lambda_vel", 0.0)
    model_args.setdefault("lambda_vel_rcxyz", 0.0)
    model_args.setdefault("sigma_small", True)
    return Namespace(**model_args)


@MODEL_BUNDLES.register_module()
class FlowMDMBundle(ModelBundle):
    """FlowMDM text-to-motion bundle for HumanML3D-263 generation."""

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        model_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        clip_weights_path: Optional[str] = None,
        device: str | int = 0,
        seed: int = 42,
        guidance_param: Optional[float] = None,
        bpe_denoising_step: Optional[int] = None,
        use_chunked_att: bool = False,
        **kwargs,
    ):
        super().__init__()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
        model_args = json.loads((artifact / "args.json").read_text())
        dataset = str(model_args.get("dataset", "humanml")).lower()
        if dataset not in {"humanml", "babel"}:
            raise ValueError(f"FlowMDM artifact dataset must be humanml or babel, got {dataset!r}")
        default_model = "model001300000.pt" if dataset == "babel" else "model000500000.pt"
        model_path = Path(model_path or artifact / default_model).resolve()
        if guidance_param is None:
            guidance_param = 1.5 if dataset == "babel" else 2.5
        if bpe_denoising_step is None:
            bpe_denoising_step = 125 if dataset == "babel" else 60

        if dataset == "babel":
            mean_path = Path(mean_path or artifact / "rfeats_mean.pt").resolve()
            std_path = Path(std_path or artifact / "rfeats_std.pt").resolve()
        else:
            mean_path = Path(mean_path or artifact / "Mean.npy").resolve()
            std_path = Path(std_path or artifact / "Std.npy").resolve()

        from .network import FlowMDM
        from motius.configuration.flowmdm import get_model_args

        flow_args = _load_args(
            artifact,
            model_path,
            device=device,
            seed=seed,
            guidance_param=guidance_param,
            bpe_denoising_step=bpe_denoising_step,
            use_chunked_att=use_chunked_att,
        )
        torch.manual_seed(seed)
        resolved_device = torch.device(
            f"cuda:{int(device)}" if torch.cuda.is_available() and str(device).isdigit()
            else ("cuda" if torch.cuda.is_available() and str(device) == "cuda" else "cpu")
        )
        if not clip_weights_path:
            raise ValueError("FlowMDM requires artifact-local CLIP weights")
        with artifact_clip_load(clip_weights_path, expected_name="ViT-B/32"):
            model = FlowMDM(**get_model_args(flow_args))
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        # The released checkpoint retains an unused pre-BPE-refactor frequency
        # buffer, also ignored by the original loader. Current RoPE has only
        # relative frequencies; absolute positions belong to seqTransEncoder.pos_emb.
        state.pop("seqTransEncoder.attn_layers.rotary_pos_emb.absolute.inv_freq", None)
        missing, unexpected = model.load_state_dict(state, strict=False)
        optional = {"t_pos_encoder.pe", "seqTransEncoder.attn_layers.rotary_pos_emb.last_timestep"}
        bad_missing = [key for key in missing if not key.startswith("clip_model.") and key not in optional]
        bad_unexpected = [key for key in unexpected if key not in optional]
        if bad_missing or bad_unexpected:
            raise RuntimeError(f"FlowMDM checkpoint mismatch: missing={bad_missing}, unexpected={bad_unexpected}")
        self.model = model.to(resolved_device).eval()
        self._trainable_modules = ["model"]
        self._save_ckpt_modules = ["model"]
        self.flow_args = flow_args
        self.guidance_param = float(guidance_param)
        self.dataset = dataset
        self.representation = "babel135" if dataset == "babel" else "hml263"
        self.fps = 30.0 if dataset == "babel" else 20.0

        if dataset == "babel":
            mean = _load_tensor(mean_path)
            std = _load_tensor(std_path)
            expected_dim = 135
        else:
            mean = np.load(str(mean_path))
            std = np.load(str(std_path))
            expected_dim = 263
        mean = np.asarray(mean, dtype=np.float32).reshape(-1)
        std = np.asarray(std, dtype=np.float32).reshape(-1)
        if mean.shape != (expected_dim,) or std.shape != (expected_dim,):
            raise ValueError(
                f"expected {expected_dim}-dim {dataset} FlowMDM stats, "
                f"got {mean.shape} and {std.shape}"
            )
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

    def to_device(self, device):
        device = torch.device(device)
        self.model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device


    def save_pretrained(self, save_directory, **kwargs):
        """Export current neural weights and architecture in the released layout."""
        from safetensors.torch import save_file
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        args = dict(vars(self.flow_args))
        args.pop("model_path", None)
        (destination / "args.json").write_text(json.dumps(args, indent=2) + "\n")
        filename = "model001300000.pt" if self.dataset == "babel" else "model000500000.pt"
        torch.save({key: value.detach().cpu().contiguous()
                    for key, value in self.network.state_dict().items()
                    if not key.startswith("clip_model.")}, destination / filename)
        save_file({key: value.detach().cpu().contiguous()
                   for key, value in self.network.clip_model.state_dict().items()},
                  str(destination / "clip.safetensors"))
        if self.dataset == "babel":
            torch.save(self.mean.detach().cpu(), destination / "rfeats_mean.pt")
            torch.save(self.std.detach().cpu(), destination / "rfeats_std.pt")
        else:
            np.save(destination / "Mean.npy", self.mean.detach().cpu().numpy())
            np.save(destination / "Std.npy", self.std.detach().cpu().numpy())
        (destination / "model_index.json").write_text(json.dumps({
            "_class_name": "FlowMDMPipeline", "_library_name": "motius", "model_type": "flowmdm",
            "bundle_class": "motius.models.flowmdm.FlowMDMBundle",
            "pipeline_class": "motius.pipelines.flowmdm.FlowMDMPipeline"}, indent=2) + "\n")
        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / "args.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        if path.is_dir() and (path / "args.json").exists():
            clip_weights = path / "clip.safetensors"
            if not clip_weights.exists():
                clip_weights = path / "clip.pt"
            if not clip_weights.exists():
                raise FileNotFoundError(
                    "The FlowMDM artifact is incomplete: missing bundled "
                    f"CLIP weights at {path / 'clip.safetensors'}."
                )
            return cls(
                artifact_dir=str(path),
                clip_weights_path=str(clip_weights),
                **kwargs,
            )
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    @property
    def network(self):
        model = self.model
        while hasattr(model, "module"):
            model = model.module
        return model

    def predict_motion(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return self.predict_motion(*args, **kwargs)
