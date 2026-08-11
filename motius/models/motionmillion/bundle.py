"""MotionMillion / "Go to Zero" ModelBundle.

Wraps the Motius-native ICCV'25 "Go to Zero: Towards Zero-shot Motion
Generation with Million-scale Data" (MotionMillion) text-to-motion
implementation behind a clean ``ModelBundle`` interface.

Architecture (implemented in ``motius.models.motionmillion.network``):

* **HumanVQVAE** — non-causal 1D-conv encoder/decoder + **FSQ** tokenizer
  (Finite Scalar Quantization, 65536-code codebook, levels ``[8,8,8,5,5,5]``).
* **LLaMAHF** — 7B (or 3B) autoregressive transformer with RoPE and a
  length-causal text cross-attention mask; greedy AR decoding with EOS stop.
* **Flan-T5-XL** text encoder (frozen, reloaded by name; not stored in the
  Motius artifact, exactly like CLIP in MDM / SentenceT5 in MotionStreamer).

Representation: the released model emits the **same 272-dim, 30 fps
``humanml3d_272``** representation used by MotionStreamer (root xz-vel + heading
6D + 22 joint pos/vel + 22 local 6D rotations). After de-normalising with the
MotionMillion ``vector_272`` mean/std, the raw 272 vectors feed *directly* into
the MotionStreamer-272 evaluator (and, via ``motion272_to_hml263``, the
HumanML3D-263 evaluator) — no rotation re-encoding required.
"""

from __future__ import annotations

import json
import pickle
import shutil
import types
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

# Repo root: motius/models/motionmillion/bundle.py -> repository root.
_REPO_ROOT = Path(__file__).resolve().parents[3]

_CKPT_DIR = _REPO_ROOT / "checkpoints/motionmillion"
_DEFAULT_MEAN = _CKPT_DIR / "mean_std/vector_272/mean.npy"
_DEFAULT_STD = _CKPT_DIR / "mean_std/vector_272/std.npy"

# Flan-T5-XL text encoder (clip_dim = 2048). Resolved by HF name.
_DEFAULT_TEXT_MODEL = "google/flan-t5-xl"

# VQVAE (FSQ) defaults — released ``test_t2m_7B.sh`` / ``train_tokenizer`` config.
_VQVAE_DEFAULTS = {
    "nb_code": 65536,
    "code_dim": 512,
    "output_emb_width": 512,
    "down_t": 1,
    "stride_t": 2,
    "width": 512,
    "depth": 3,
    "dilation_growth_rate": 3,
    "activation": "relu",
    "norm": "LN",
    "kernel_size": 3,
    "use_patcher": True,
    "patch_size": 1,
    "patch_method": "haar",
    "input_dim": 272,
    "quantizer": "FSQ",
}
# LLaMA AR defaults — released 7B config.
_AR_DEFAULTS = {
    "config_name": "7B",   # n_layer=36, n_head=32, n_embd=4096
    "block_size": 301,
    "clip_dim": 2048,      # flan-t5-xl hidden size
    "tie_weights": False,
}


def _tolerant_pickle_module():
    """A ``pickle`` shim whose Unpickler tolerates unimportable classes.

    The released ``t2m_*_all.zip`` checkpoints bundle a DeepSpeed/optimizer state
    alongside the ``trans`` weights; importing those classes fails in a clean
    env. We only read the tensor ``state_dict``, so unknown classes are replaced
    with a no-op stub — keeping the loader independent of the original repo.
    """

    class _Stub:
        def __init__(self, *a, **k):
            pass

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            try:
                return super().find_class(module, name)
            except Exception:
                return _Stub

    shim = types.ModuleType("motius_tolerant_pickle_motionmillion")
    shim.Unpickler = _Unpickler
    shim.load = lambda f, **kw: _Unpickler(f, **kw).load()
    shim.loads = pickle.loads
    shim.Pickler = pickle.Pickler
    shim.dump = pickle.dump
    shim.dumps = pickle.dumps
    return shim


def _strip_module_prefix(state: dict) -> dict:
    return {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in state.items()}


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path))
    except Exception:
        return local


def _copy_pretrained_tree(src: Path, dst: Path, ignore_patterns: Optional[tuple[str, ...]] = None) -> None:
    if not src.exists() or not any(src.iterdir()):
        raise FileNotFoundError(f"pretrained component not found or empty: {src}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ignore = shutil.ignore_patterns(*(ignore_patterns or ())) if ignore_patterns else None
    shutil.copytree(src, dst, symlinks=False, ignore=ignore)


def _resolve_repo_path(value: Union[str, Path]) -> Path:
    path = Path(value)
    if path.exists() or path.is_absolute():
        return path
    repo_path = _REPO_ROOT / path
    if repo_path.exists():
        return repo_path
    return path


@MODEL_BUNDLES.register_module()
class MotionMillionBundle(ModelBundle):
    """MotionMillion / Go-to-Zero text-to-motion bundle (272-dim, Flan-T5-XL text)."""

    def __init__(
        self,
        fsq_path: Optional[str] = None,
        ar_path: Optional[str] = None,
        text_model_name: str = _DEFAULT_TEXT_MODEL,
        config: Optional[dict] = None,
        fsq_weights_path: Optional[str] = None,
        ar_weights_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        load_ar_model: bool = True,
        load_text_model: bool = True,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Construct the bundle.

        Two weight sources are supported:

        * **Self-contained Motius artifact** — ``config`` plus
          ``fsq_weights_path`` / ``ar_weights_path`` (safetensors), as produced
          by :meth:`save_pretrained` / consumed by :meth:`from_pretrained`.
        * **Explicit raw upstream checkpoints** — ``fsq_path`` (``.zip``/``.pth``
          with a ``net`` key) and ``ar_path`` (with a ``trans`` key). These are
          intended for converter/debug code only; the bundle never guesses a raw
          checkpoint location.
        """
        super().__init__()
        if fsq_weights_path is None and fsq_path is None:
            raise ValueError(
                "MotionMillionBundle requires fsq_weights_path from an "
                "Motius artifact or an explicit raw fsq_path for conversion."
            )
        if load_ar_model and ar_weights_path is None and ar_path is None:
            raise ValueError(
                "MotionMillionBundle requires ar_weights_path from an "
                "Motius artifact or an explicit raw ar_path for conversion."
            )

        from .network import HumanVQVAE, LLaMAHF, LLaMAHFConfig

        self.text_model_name = text_model_name
        cfg = dict(config) if config is not None else {}
        vq_cfg = {**_VQVAE_DEFAULTS, **cfg.get("vqvae", {})}
        ar_cfg = {**_AR_DEFAULTS, **cfg.get("ar", {})}
        self._vq_cfg = vq_cfg
        self._ar_cfg = ar_cfg

        # --- tokenizer (HumanVQVAE + FSQ) ------------------------------- #
        vqvae = HumanVQVAE(
            nb_code=vq_cfg["nb_code"], code_dim=vq_cfg["code_dim"],
            output_emb_width=vq_cfg["output_emb_width"], down_t=vq_cfg["down_t"],
            stride_t=vq_cfg["stride_t"], width=vq_cfg["width"], depth=vq_cfg["depth"],
            dilation_growth_rate=vq_cfg["dilation_growth_rate"], activation=vq_cfg["activation"],
            norm=vq_cfg["norm"], kernel_size=vq_cfg["kernel_size"], use_patcher=vq_cfg["use_patcher"],
            patch_size=vq_cfg["patch_size"], patch_method=vq_cfg["patch_method"],
            input_dim=vq_cfg["input_dim"], quantizer=vq_cfg["quantizer"],
        )

        # --- AR transformer --------------------------------------------- #
        # vocab = codebook_size + 2 (PAD/EOS). FSQ codebook_size = prod(levels).
        codebook_size = vqvae.vqvae.quantizer.codebook_size
        ar = None
        if load_ar_model:
            llama_cfg = LLaMAHFConfig.from_name(
                ar_cfg["config_name"],
                block_size=ar_cfg["block_size"],
                vocab_size=codebook_size + 2,
                clip_dim=ar_cfg["clip_dim"],
                tie_weights=ar_cfg["tie_weights"],
            )
            ar = LLaMAHF(llama_cfg)

        # --- load weights ----------------------------------------------- #
        if fsq_weights_path is not None:
            self._load_weights(vqvae, fsq_weights_path, key=None)
        else:
            self._load_weights(vqvae, str(fsq_path), key="net")
        if ar is not None:
            if ar_weights_path is not None:
                self._load_weights(ar, ar_weights_path, key=None)
            else:
                self._load_weights(ar, str(ar_path), key="trans")

        vqvae.eval()
        if ar is not None:
            ar.eval()
        self.vqvae = vqvae
        self.ar = ar
        self.nfeats = 272

        # --- text encoder (frozen, reloadable; not stored in artifact) -- #
        self.tokenizer = None
        self.text_model = None
        if load_text_model:
            self.tokenizer, self.text_model = self._build_text_model(text_model_name)

        # --- normalization buffers (272-dim) ---------------------------- #
        mean = np.load(str(mean_path or _DEFAULT_MEAN)).astype(np.float32)
        std = np.load(str(std_path or _DEFAULT_STD)).astype(np.float32)
        if mean.shape != (272,) or std.shape != (272,):
            raise ValueError(f"expected 272-dim mean/std, got {mean.shape} and {std.shape}")
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer("std", torch.from_numpy(std), persistent=True)

        if device is not None:
            self.to_device(device)

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------
    @staticmethod
    def _load_weights(module, path: str, key: Optional[str]) -> None:
        p = str(path)
        if p.endswith(".safetensors"):
            from safetensors.torch import load_file

            sd = load_file(p)
        else:
            try:
                ckpt = torch.load(p, map_location="cpu", weights_only=False)
            except Exception:
                ckpt = torch.load(
                    p, map_location="cpu", pickle_module=_tolerant_pickle_module(),
                    weights_only=False,
                )
            sd = ckpt[key] if (key is not None and isinstance(ckpt, dict) and key in ckpt) else ckpt
        sd = _strip_module_prefix(sd)
        module.load_state_dict(sd, strict=True)

    @staticmethod
    def _build_text_model(name: str):
        from transformers import T5EncoderModel, T5Tokenizer

        tokenizer = T5Tokenizer.from_pretrained(name)
        text_model = T5EncoderModel.from_pretrained(name)
        text_model.eval()
        for p in text_model.parameters():
            p.requires_grad = False
        return tokenizer, text_model

    # ------------------------------------------------------------------
    # diffusers-style artifact I/O (self-contained, repo-independent)
    # ------------------------------------------------------------------
    def config_dict(self) -> dict:
        return {"vqvae": dict(self._vq_cfg), "ar": dict(self._ar_cfg)}

    def save_pretrained(self, save_directory: str, safe_serialization: bool = True, **kwargs):
        """Export a self-contained Motius MotionMillion artifact.

        Layout::

            <dir>/mm_config.json        # vqvae + ar arch config, text model name
            <dir>/fsq.safetensors       # HumanVQVAE (encoder/decoder/FSQ) weights
            <dir>/ar.safetensors        # LLaMA AR weights
            <dir>/mean.npy, std.npy     # 272-dim denorm stats
            <dir>/text_encoder/         # Flan-T5-XL tokenizer + encoder files

        The default export is complete and stores the Flan-T5-XL text encoder in
        the artifact. Pass ``include_text_encoder=False`` only for legacy
        lightweight artifacts.
        """
        import os

        include_text_encoder = bool(kwargs.pop("include_text_encoder", True))
        text_encoder_subdir = str(kwargs.pop("text_encoder_subdir", "text_encoder"))
        text_model_source = kwargs.pop("text_model_source", None)
        os.makedirs(save_directory, exist_ok=True)
        save_dir = Path(save_directory)

        text_model_name = self.text_model_name
        if include_text_encoder:
            src_name = text_model_source or self.text_model_name
            src = _resolve_repo_path(src_name)
            if not src.exists() or not any(src.iterdir()):
                from huggingface_hub import snapshot_download

                src = Path(snapshot_download(repo_id=str(src_name)))
            _copy_pretrained_tree(
                src,
                save_dir / text_encoder_subdir,
                ignore_patterns=(
                    ".cache",
                    "pytorch_model*.bin",
                    "pytorch_model.bin.index.json",
                    "tf_model*.h5",
                    "tf_model.h5.index.json",
                    "flax_model*.msgpack",
                    "flax_model.msgpack.index.json",
                    "rust_model.ot",
                ),
            )
            text_model_name = text_encoder_subdir

        cfg = {
            "model_type": "motionmillion",
            "text_model_name": self.text_model_name,
            "artifact_format": "motius-motionmillion-v1",
            "text_encoder": {
                "stored_in_artifact": include_text_encoder,
                "path": text_model_name if include_text_encoder else None,
                "source": self.text_model_name,
                "type": "google/flan-t5-xl",
            },
            "config": self.config_dict(),
        }
        (save_dir / "mm_config.json").write_text(json.dumps(cfg, indent=2))

        def _cpu_state(m):
            return {k: v.detach().cpu().contiguous() for k, v in m.state_dict().items()}

        if safe_serialization:
            from safetensors.torch import save_file

            save_file(_cpu_state(self.vqvae), str(save_dir / "fsq.safetensors"))
            save_file(_cpu_state(self.ar), str(save_dir / "ar.safetensors"))
        else:
            torch.save(_cpu_state(self.vqvae), str(save_dir / "fsq.pt"))
            torch.save(_cpu_state(self.ar), str(save_dir / "ar.pt"))

        np.save(str(save_dir / "mean.npy"), self.mean.detach().cpu().numpy().astype(np.float32))
        np.save(str(save_dir / "std.npy"), self.std.detach().cpu().numpy().astype(np.float32))
        model_index = {
            "_class_name": "MotionMillionPipeline",
            "_diffusers_version": "motius",
            "pipeline": {
                "library": "motius",
                "class_name": "MotionMillionPipeline",
                "module": "motius.pipelines.motionmillion",
            },
            "bundle": {
                "library": "motius",
                "class_name": "MotionMillionBundle",
                "module": "motius.models.motionmillion",
            },
            "weights": {
                "fsq": "fsq.safetensors" if safe_serialization else "fsq.pt",
                "ar": "ar.safetensors" if safe_serialization else "ar.pt",
                "text_encoder": text_encoder_subdir if include_text_encoder else None,
            },
        }
        (save_dir / "model_index.json").write_text(json.dumps(model_index, indent=2))
        readme = save_dir / "README.md"
        if not readme.exists():
            readme.write_text(
                "---\n"
                "library_name: motius\n"
                "pipeline_tag: other\n"
                "---\n\n"
                "# Motius GoToZero / MotionMillion HumanML3D-272\n\n"
                "Load with `MotionMillionPipeline.from_pretrained(...)` from motius.\n",
            )
        return save_directory

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Load a self-contained Motius MotionMillion artifact (local dir or HF Hub id)."""
        path = Path(pretrained_model_name_or_path)
        if not (path / "mm_config.json").exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg_file = path / "mm_config.json"
        if not cfg_file.exists():
            return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        meta = json.loads(cfg_file.read_text())
        fsq_w = path / "fsq.safetensors"
        if not fsq_w.exists():
            fsq_w = path / "fsq.pt"
        ar_w = path / "ar.safetensors"
        if not ar_w.exists():
            ar_w = path / "ar.pt"
        text_model_name = kwargs.pop("text_model_name", None)
        if text_model_name is None:
            text_meta = meta.get("text_encoder", {}) if isinstance(meta, dict) else {}
            rel_text = text_meta.get("path") if isinstance(text_meta, dict) else None
            if rel_text and (path / rel_text).exists():
                text_model_name = str(path / rel_text)
            else:
                text_model_name = meta.get("text_model_name", _DEFAULT_TEXT_MODEL)
        return cls(
            config=meta["config"],
            fsq_weights_path=str(fsq_w),
            ar_weights_path=str(ar_w),
            mean_path=str(path / "mean.npy"),
            std_path=str(path / "std.npy"),
            text_model_name=text_model_name,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # device / forward helpers
    # ------------------------------------------------------------------
    def to_device(self, device):
        device = torch.device(device)
        self.vqvae.to(device)
        if self.ar is not None:
            self.ar.to(device)
        if self.text_model is not None:
            self.text_model.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @torch.no_grad()
    def encode_text(self, captions: List[str], max_tokens: int = 150):
        """Flan-T5-XL text features ``(B, L, 2048)`` and attention mask ``(B, L)``."""
        if self.text_model is None:
            raise RuntimeError("text encoder not loaded (load_text_model=False).")
        device = self.text_model.device
        inputs = self.tokenizer(list(captions), padding=True, truncation=True, return_tensors="pt")
        y_mask = inputs.attention_mask.to(device)
        feat = self.text_model(
            input_ids=inputs.input_ids.to(device),
            attention_mask=y_mask, output_hidden_states=False,
        ).last_hidden_state
        if feat.shape[1] > max_tokens:
            feat = feat[:, :max_tokens, :]
            y_mask = y_mask[:, :max_tokens]
        return feat, y_mask

    def denormalize(self, motion_272: torch.Tensor) -> torch.Tensor:
        """Un-standardize MotionMillion-272 features back to physical (raw) scale."""
        return motion_272 * self.std + self.mean

    def forward(self, *args, **kwargs):  # pragma: no cover - use pipeline
        if self.ar is None:
            raise RuntimeError(
                "MotionMillionBundle was loaded with load_ar_model=False; "
                "only tokenizer reconstruction is available."
            )
        raise NotImplementedError("Use MotionMillionPipeline.infer_t2m for inference.")
