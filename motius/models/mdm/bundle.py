"""MDM ModelBundle.

Wraps the Motius-native MDM network + Gaussian diffusion implementation (see
``motius.models.mdm.network``) behind a clean ``ModelBundle``
interface. Runtime loading is artifact-based; raw upstream checkpoints are
handled by converter/debug code.
"""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

# Repo root: motius/models/mdm/bundle.py -> parents[3]
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    """Resolve a HuggingFace Hub repo id to a local snapshot dir.

    Returns ``local`` unchanged if it is already a directory; otherwise tries
    ``snapshot_download(name_or_path)`` and returns the cached path.
    """
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local

# HumanML3D-263 normalization stats used to denormalize the MDM diffusion output.
#
# IMPORTANT: this MUST be the *training* normalization (HumanML3D ``data_root``
# ``Mean.npy`` / ``Std.npy``), NOT the T2M evaluator stats (``t2m_mean.npy`` /
# ``t2m_std.npy``). MDM's data loader normalizes the diffusion model with
# ``data_root`` stats (mode='train'/'eval'), while the *evaluator* (mode='gt')
# uses the ``t2m_*`` stats. The two differ: the evaluator stats shrink the root
# channels by ~30x (feat_bias), so using them to denormalize the model output
# collapses root velocity to ~0 and leaves a constant per-frame forward bias
# (= dataset mean) -> visible forward drift.
_DEFAULT_MEAN = _REPO_ROOT / "data/statistic/humanml3d_263/Mean.npy"
_DEFAULT_STD = _REPO_ROOT / "data/statistic/humanml3d_263/Std.npy"


# Stats file names looked up next to a checkpoint (sidecar). Keeping the
# normalization next to the weights makes each reproduced checkpoint
# self-contained and prevents the recurring "wrong Mean/Std" class of bugs.
_SIDECAR_MEAN_NAMES = ("Mean.npy", "mean.npy")
_SIDECAR_STD_NAMES = ("Std.npy", "std.npy")


def _resolve_stats(model_path: Path, mean_path, std_path):
    """Resolve the (mean, std) files used to denormalize the MDM output.

    Priority (first hit wins):

    1. Explicit ``mean_path`` / ``std_path`` arguments.
    2. A ``Mean.npy`` / ``Std.npy`` *sidecar* sitting next to the checkpoint
       (``model_path``'s directory). Drop the correct training stats here to
       make a checkpoint fully self-contained.
    3. The repo-level default (``data/statistic/humanml3d_263``).

    Returns ``(mean_file, std_file, source_tag)``.
    """
    if mean_path and std_path:
        return Path(mean_path), Path(std_path), "explicit"

    ckpt_dir = Path(model_path).resolve().parent
    side_mean = next((ckpt_dir / n for n in _SIDECAR_MEAN_NAMES if (ckpt_dir / n).exists()), None)
    side_std = next((ckpt_dir / n for n in _SIDECAR_STD_NAMES if (ckpt_dir / n).exists()), None)
    if side_mean is not None and side_std is not None:
        return side_mean, side_std, f"sidecar:{ckpt_dir}"

    return Path(_DEFAULT_MEAN), Path(_DEFAULT_STD), "repo_default"


# Defaults for attributes upstream helpers may read but that old (2022)
# checkpoints such as ``humanml-encoder-512`` predate.
_MDM_ARG_DEFAULTS = {
    "pred_len": 0,
    "context_len": 0,
    "emb_policy": "add",
    "multi_target_cond": False,
    "multi_encoder_type": "multi",
    "target_enc_layers": 1,
    "mask_frames": False,
    "lambda_vel": 0.0,
    "lambda_rcxyz": 0.0,
    "lambda_fc": 0.0,
    "unconstrained": False,
    "text_encoder_type": "clip",
    "pos_embed_max_len": 5000,
}

# The MDM config fields we persist into the self-contained Motius artifact.
# (Everything ``create_model_and_diffusion`` / the network reads.)
_MDM_CONFIG_KEYS = (
    "dataset", "arch", "latent_dim", "layers", "cond_mode", "cond_mask_prob",
    "emb_trans_dec", "noise_schedule", "diffusion_steps", "sigma_small",
    "text_encoder_type", "pos_embed_max_len", "mask_frames", "emb_policy",
    "pred_len", "context_len", "multi_target_cond", "multi_encoder_type",
    "target_enc_layers", "lambda_vel", "lambda_rcxyz", "lambda_fc", "unconstrained",
)


def _args_from_dict(cfg: dict) -> Namespace:
    """Build an MDM args ``Namespace`` from a plain dict, filling defaults."""
    args = Namespace(**dict(cfg))
    for key, val in _MDM_ARG_DEFAULTS.items():
        if not hasattr(args, key):
            setattr(args, key, val)
    return args


def _load_mdm_args(model_path: Path) -> Namespace:
    """Load the ``args.json`` that sits next to a raw MDM checkpoint."""
    args_file = model_path.parent / "args.json"
    if not args_file.exists():
        raise FileNotFoundError(
            f"MDM args.json not found next to checkpoint: {args_file}"
        )
    return _args_from_dict(json.loads(args_file.read_text()))


class _DummyDataset:
    num_actions = 1


class _DummyData:
    dataset = _DummyDataset()


@MODEL_BUNDLES.register_module()
class MDMBundle(ModelBundle):
    """MDM text-to-motion bundle (HumanML3D-263, CLIP text encoder)."""

    def __init__(
        self,
        model_path: Optional[str] = None,
        guidance_param: float = 2.5,
        config: Optional[dict] = None,
        weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        use_ema: bool = False,
        **kwargs,
    ):
        """Construct the MDM bundle.

        Two weight sources are supported:

        * **Raw upstream checkpoint** — pass ``model_path`` (a ``.pt`` with a
          sibling ``args.json``). Backward compatible with the released MDM.
        * **Self-contained Motius artifact** — pass ``config`` (the MDM
          arch/diffusion fields) and ``weights_path`` (a ``model.safetensors``
          holding the network weights, no CLIP). This is what
          :meth:`from_pretrained` / :meth:`save_pretrained` use.
        """
        super().__init__()
        from .network import (
            ClassifierFreeSampleModel,
            create_model_and_diffusion,
            load_saved_model,
        )

        self.model_path = str(model_path) if model_path else None
        self.guidance_param = float(guidance_param)

        if config is not None:
            args = _args_from_dict(config)
        elif model_path is not None:
            args = _load_mdm_args(Path(self.model_path))
        else:
            raise ValueError("MDMBundle needs either `model_path` or `config`.")
        if args.dataset != "humanml":
            raise ValueError(
                f"MDMBundle currently supports the HumanML3D checkpoint; "
                f"got dataset={args.dataset!r}"
            )
        self._args = args

        net, diffusion = create_model_and_diffusion(args, _DummyData())
        if weights_path is not None:
            self._load_net_weights(net, weights_path)
        elif model_path is not None:
            load_saved_model(net, self.model_path, use_avg=use_ema)
        else:
            raise ValueError("MDMBundle needs either `weights_path` or `model_path`.")
        if self.guidance_param != 1.0:
            net = ClassifierFreeSampleModel(net)
        net.eval()

        # Register the network as a child module; diffusion is a plain helper.
        self.net = net
        self.diffusion = diffusion

        self.njoints = 263
        self.nfeats = 1

        # Resolve stats: explicit > sidecar next to weights/checkpoint > repo default.
        stats_anchor = Path(weights_path) if weights_path else Path(self.model_path)
        mean_p, std_p, src = _resolve_stats(stats_anchor, mean_path, std_path)
        self.stats_source = src
        mean = np.load(str(mean_p)).astype(np.float32)
        std = np.load(str(std_p)).astype(np.float32)
        if mean.shape != (263,) or std.shape != (263,):
            raise ValueError(
                f"expected 263-dim mean/std, got {mean.shape} and {std.shape}"
            )
        # persistent=True so the stats travel inside any Motius checkpoint that
        # serializes this bundle's state_dict -- the denorm stats are part of the
        # model contract, not an external dependency. (Mismatched / external stats
        # have repeatedly caused root-velocity collapse + forward drift.)
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

    # ------------------------------------------------------------------
    # diffusers-style artifact I/O (self-contained, raw-checkout-independent)
    # ------------------------------------------------------------------
    @staticmethod
    def _inner_mdm(net):
        """Return the underlying MDM network, unwrapping the CFG sampler."""
        return getattr(net, "model", net)

    @staticmethod
    def _net_state_dict_no_clip(mdm) -> dict:
        """MDM weights excluding the (frozen, reloadable) CLIP backbone + the
        fixed positional-encoding buffers."""
        return {
            k: v.detach().cpu()
            for k, v in mdm.state_dict().items()
            if not k.startswith("clip_model.") and "sequence_pos_encoder.pe" not in k
        }

    def _load_net_weights(self, net, weights_path: str) -> None:
        """Load a saved (CLIP-free) MDM state dict into a freshly-built network."""
        wp = str(weights_path)
        if wp.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(wp)
        else:
            sd = torch.load(wp, map_location="cpu")
            if isinstance(sd, dict) and "model" in sd and "state_dict" not in sd:
                sd = sd["model"]
        # strict=False: CLIP backbone + pos-encoding buffers are not in the file.
        missing, unexpected = net.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys loading MDM weights: {unexpected[:5]}"
        assert all(
            k.startswith("clip_model.") or "sequence_pos_encoder" in k for k in missing
        ), f"unexpected missing keys loading MDM weights: {missing[:5]}"

    def config_dict(self) -> dict:
        """The MDM arch/diffusion config persisted into the artifact."""
        return {k: getattr(self._args, k) for k in _MDM_CONFIG_KEYS if hasattr(self._args, k)}

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        """Export a self-contained Motius MDM artifact.

        Layout::

            <dir>/mdm_config.json     # arch + diffusion config + guidance_param
            <dir>/model.safetensors   # network weights (no CLIP, no pe buffers)
            <dir>/Mean.npy, Std.npy   # 263-dim denorm stats

        Reloadable with :meth:`from_pretrained` and fully independent of the
        original ``.pt`` + ``args.json`` format.
        """
        import os

        os.makedirs(save_directory, exist_ok=True)
        save_dir = Path(save_directory)

        cfg = {
            "model_type": "mdm",
            "guidance_param": self.guidance_param,
            "config": self.config_dict(),
        }
        (save_dir / "mdm_config.json").write_text(json.dumps(cfg, indent=2))

        mdm = self._inner_mdm(self.net)
        state = self._net_state_dict_no_clip(mdm)
        if safe_serialization:
            from safetensors.torch import save_file

            save_file({k: v.contiguous() for k, v in state.items()}, str(save_dir / "model.safetensors"))
        else:
            torch.save(state, str(save_dir / "model.pt"))

        np.save(str(save_dir / "Mean.npy"), self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(str(save_dir / "Std.npy"), self.std.detach().cpu().numpy().astype(np.float32))
        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a self-contained Motius MDM artifact (see :meth:`save_pretrained`).

        ``pretrained_model_name_or_path`` may be a local directory **or** a
        Hugging Face Hub repo id, which is fetched via ``snapshot_download``.
        """
        path = Path(pretrained_model_name_or_path)
        if not (path / "mdm_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "mdm_config.json"
        if not cfg_file.exists():
            # Fall back to the generic spec-based path (raises a helpful error
            # if the artifact is not a Motius MDM directory).
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        meta = json.loads(cfg_file.read_text())
        weights = path / "model.safetensors"
        if not weights.exists():
            weights = path / "model.pt"
        guidance_param = kwargs.pop("guidance_param", meta.get("guidance_param", 2.5))
        return cls(
            config=meta["config"],
            weights_path=str(weights),
            mean_path=str(path / "Mean.npy"),
            std_path=str(path / "Std.npy"),
            guidance_param=guidance_param,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Atomic forward helpers (shared between pipeline and any future trainer)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_text(self, texts: List[str]):
        return self.net.encode_text(texts)

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        """Un-standardize HumanML3D-263 features back to physical scale."""
        return motion_263 * self.std + self.mean

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def forward(self, *args, **kwargs):  # pragma: no cover - use pipeline
        raise NotImplementedError(
            "Use MDMPipeline.infer_t2m for inference."
        )
