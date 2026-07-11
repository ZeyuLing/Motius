"""FlowMDM ModelBundle.

This bundle uses the vendored FlowMDM runtime under
``motius.models.flowmdm.network`` and self-contained artifacts under
``checkpoints/flowmdm``. It never imports from an external checkout at
runtime.
"""

from __future__ import annotations

import json
import sys
import types
from argparse import Namespace
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
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


class _UnusedRotation2XYZ(torch.nn.Module):
    """FlowMDM predicts HML263; the SMPL visualizer is unused for T2M eval."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.smpl_model = torch.nn.Identity()


def _install_unused_rotation2xyz_module() -> None:
    module_name = "motius.models.flowmdm.network.model.rotation2xyz"
    module = types.ModuleType(module_name)
    module.Rotation2xyz = _UnusedRotation2XYZ
    module.JOINTSTYPES = ["a2m", "a2mpl", "smpl", "vibe", "vertices", "smplx"]
    sys.modules[module_name] = module


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
        device: str | int = 0,
        seed: int = 42,
        guidance_param: float = 2.5,
        bpe_denoising_step: int = 60,
        use_chunked_att: bool = False,
        **kwargs,
    ):
        super().__init__()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
        model_path = Path(model_path or artifact / "model000500000.pt").resolve()
        mean_path = Path(mean_path or artifact / "Mean.npy").resolve()
        std_path = Path(std_path or artifact / "Std.npy").resolve()

        _install_unused_rotation2xyz_module()
        from motius.models.flowmdm.network.diffusion.diffusion_wrappers import (
            DiffusionWrapper_FlowMDM,
        )
        import motius.models.flowmdm.network.model.FlowMDM as flowmdm_module
        import motius.models.flowmdm.network.model.MDM as mdm_module
        from motius.models.flowmdm.network.utils.model_util import load_model

        # Patch the visualization-only SMPL converter before the network is built.
        mdm_module.Rotation2xyz = _UnusedRotation2XYZ
        flowmdm_module.Rotation2xyz = _UnusedRotation2XYZ

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
        model, diffusion = load_model(flow_args, resolved_device)
        self.sampler = DiffusionWrapper_FlowMDM(flow_args, diffusion, model)
        self.flow_args = flow_args
        self.guidance_param = float(guidance_param)

        mean = np.load(str(mean_path)).astype(np.float32)
        std = np.load(str(std_path)).astype(np.float32)
        if mean.shape != (263,) or std.shape != (263,):
            raise ValueError(f"expected 263-dim FlowMDM stats, got {mean.shape} and {std.shape}")
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

    def to_device(self, device):
        device = torch.device(device)
        self.sampler.model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.sampler.model.parameters()).device

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        return motion_263 * self.std.to(motion_263) + self.mean.to(motion_263)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / "args.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        if path.is_dir() and (path / "args.json").exists():
            return cls(artifact_dir=str(path), **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use FlowMDMPipeline.infer_t2m for inference.")
