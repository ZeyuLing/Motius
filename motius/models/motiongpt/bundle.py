"""MotionGPT ModelBundle.

This bundle wraps the original MotionGPT (NeurIPS 2023) inference components in
a self-contained runtime:

* ``MLM``: FLAN-T5-base language model extended with 515 motion tokens.
* ``VQVae``: 263-dim HumanML3D motion tokenizer / decoder.

The released checkpoint stores a distinct T5 ``lm_head`` while sharing
``shared`` / encoder / decoder input embeddings. Loading with ordinary
``t5-base`` is wrong because its FFN dimension is 3072; MotionGPT uses the
FLAN-T5-base / T5-v1.1 gated FFN shape (2048).
"""

from __future__ import annotations

import contextlib
import importlib
import json
import random
import shutil
import sys
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "motiongpt"
_DEFAULT_CHECKPOINT = "motiongpt_s3_h3d.tar"
_DEFAULT_T5_NAME = "google/flan-t5-base"


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))
    except Exception:
        return local


def _install_legacy_aliases() -> None:
    """Expose the vendored package as ``mGPT`` for upstream absolute imports."""
    pkg = importlib.import_module("motius.models.motiongpt.network.mGPT")
    sys.modules.setdefault("mGPT", pkg)


@contextlib.contextmanager
def _t5_from_config_only(local_files_only: bool = False):
    """Make upstream ``MLM`` build T5 from config, not base pretrained weights."""
    from transformers import AutoConfig, T5ForConditionalGeneration

    original = T5ForConditionalGeneration.from_pretrained

    def from_pretrained_config(model_path, *args, **kwargs):
        cfg = kwargs.pop("config", None)
        if cfg is None:
            cfg = AutoConfig.from_pretrained(
                model_path,
                local_files_only=local_files_only,
            )
        cfg.tie_word_embeddings = False
        return T5ForConditionalGeneration(cfg)

    T5ForConditionalGeneration.from_pretrained = from_pretrained_config
    try:
        yield
    finally:
        T5ForConditionalGeneration.from_pretrained = original


def _strip_prefix(state: dict, prefix: str) -> dict:
    n = len(prefix)
    return {key[n:]: value for key, value in state.items() if key.startswith(prefix)}


def _load_npy(path: Path) -> torch.Tensor:
    return torch.from_numpy(np.load(str(path)).astype(np.float32))


@MODEL_BUNDLES.register_module()
class MotionGPTBundle(ModelBundle):
    """MotionGPT text-to-motion bundle for HumanML3D-263 generation."""

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        checkpoint: Optional[str] = None,
        t5_model_path: Optional[str] = None,
        mean_path: Optional[str] = None,
        std_path: Optional[str] = None,
        prompt_mode: str = "official_nolen",
        max_new_tokens: int = 128,
        device: str = "cuda",
        local_files_only: bool = False,
        **kwargs,
    ):
        super().__init__()
        _install_legacy_aliases()

        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
        if checkpoint is None:
            checkpoint_path = (artifact / _DEFAULT_CHECKPOINT).resolve()
        else:
            checkpoint_path = Path(checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"MotionGPT checkpoint not found: {checkpoint_path}")

        if t5_model_path is None:
            local_t5 = artifact / "deps" / "flan-t5-base"
            t5_model_path = str(local_t5 if local_t5.exists() else _DEFAULT_T5_NAME)
        if mean_path is None:
            mean_path = artifact / "assets" / "meta" / "mean.npy"
        if std_path is None:
            std_path = artifact / "assets" / "meta" / "std.npy"
        if not Path(mean_path).exists():
            raise FileNotFoundError(f"MotionGPT mean file not found: {mean_path}")
        if not Path(std_path).exists():
            raise FileNotFoundError(f"MotionGPT std file not found: {std_path}")

        self.checkpoint_path = str(checkpoint_path)
        self.t5_model_path = str(t5_model_path)
        self.prompt_mode = str(prompt_mode)
        self.max_new_tokens = int(max_new_tokens)
        self.codebook_size = 512

        self.lm = self._build_lm(self.t5_model_path, local_files_only=local_files_only)
        self.vae = self._build_vae()
        self._load_motiongpt_state(checkpoint_path)

        self.register_buffer("mean", _load_npy(Path(mean_path)), persistent=True)
        self.register_buffer("std", _load_npy(Path(std_path)), persistent=True)
        self.load_report = {
            "checkpoint": self.checkpoint_path,
            "t5_model_path": self.t5_model_path,
            "prompt_mode": self.prompt_mode,
            "max_new_tokens": self.max_new_tokens,
            "embedding_policy": "shared_encoder_decoder_untied_lm_head",
        }

        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to_device(resolved_device)
        self.eval()

    @staticmethod
    def _build_lm(model_path: str, local_files_only: bool):
        from motius.models.motiongpt.network.mGPT.archs.mgpt_lm import MLM

        with _t5_from_config_only(local_files_only=local_files_only):
            lm = MLM(
                model_path=model_path,
                model_type="t5",
                stage="lm_instruct",
                motion_codebook_size=512,
                max_length=256,
            )
        return lm

    @staticmethod
    def _build_vae():
        from motius.models.motiongpt.network.mGPT.archs.mgpt_vq import VQVae

        return VQVae(
            nfeats=263,
            quantizer="ema_reset",
            code_num=512,
            code_dim=512,
            output_emb_width=512,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            norm=None,
            activation="relu",
        )

    def _load_motiongpt_state(self, checkpoint_path: Path) -> None:
        state = torch.load(str(checkpoint_path), map_location="cpu")["state_dict"]
        lm_state = _strip_prefix(state, "lm.")
        vae_state = _strip_prefix(state, "vae.")
        self.lm.load_state_dict(lm_state, strict=True)
        self.vae.load_state_dict(vae_state, strict=True)

        language_model = self.lm.language_model
        shared = language_model.shared
        language_model.encoder.embed_tokens = shared
        language_model.decoder.embed_tokens = shared
        language_model.config.tie_word_embeddings = False

    def to_device(self, device):
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.lm.to(device)
        self.vae.to(device)
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.mean.device

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        return motion_263 * self.std.to(motion_263) + self.mean.to(motion_263)

    def generate_motion_tokens(
        self,
        captions: Sequence[str],
        lengths: Optional[Sequence[int]] = None,
        prompt_mode: Optional[str] = None,
        do_sample: bool = True,
    ) -> List[torch.Tensor]:
        mode = prompt_mode or self.prompt_mode
        captions = list(captions)
        lengths = [int(x) for x in (lengths or [0] * len(captions))]

        if mode == "official_nolen":
            return self.lm.generate_conditional(
                texts=captions,
                lengths=lengths,
                task="t2m",
                with_len=False,
                stage="test",
                tasks=None,
            )
        if mode == "official_len":
            return self.lm.generate_conditional(
                texts=captions,
                lengths=lengths,
                task="t2m",
                with_len=True,
                stage="test",
                tasks=None,
            )
        if mode == "direct":
            outputs, _ = self.lm.generate_direct(
                captions,
                max_length=self.max_new_tokens,
                num_beams=1,
                do_sample=do_sample,
            )
            return outputs
        raise ValueError(f"unsupported MotionGPT prompt_mode: {mode}")

    def decode_tokens(self, tokens: torch.Tensor, fallback_length: int) -> torch.Tensor:
        tokens = torch.clamp(tokens.to(self.device), 0, self.codebook_size - 1)
        if int(tokens.numel()) > 1:
            motion = self.vae.decode(tokens)[0]
        else:
            motion = torch.zeros((max(1, int(fallback_length)), 263), device=self.device)
        return self.denormalize(motion.float())

    @torch.no_grad()
    def infer_hml263(
        self,
        captions: Sequence[str],
        lengths: Optional[Sequence[int]] = None,
        prompt_mode: Optional[str] = None,
        seed: Optional[int] = None,
        do_sample: bool = True,
    ) -> List[np.ndarray]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        target_lengths = [int(x) for x in (lengths or [0] * len(captions))]
        tokens = self.generate_motion_tokens(
            captions,
            target_lengths,
            prompt_mode=prompt_mode,
            do_sample=do_sample,
        )
        outputs: List[np.ndarray] = []
        for token_ids, target_len in zip(tokens, target_lengths):
            motion = self.decode_tokens(token_ids, fallback_length=target_len or 1)
            if target_len:
                motion = motion[: max(1, min(int(motion.shape[0]), target_len))]
            outputs.append(motion.detach().cpu().numpy().astype(np.float32))
        return outputs

    @classmethod
    def _bundle_config_from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        if not (path / _DEFAULT_CHECKPOINT).exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        cfg = {"artifact_dir": str(path)}
        cfg.update(kwargs)
        return cfg

    def save_pretrained(self, save_directory: str, **kwargs):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        assets = save_dir / "assets" / "meta"
        deps = save_dir / "deps" / "flan-t5-base"
        assets.mkdir(parents=True, exist_ok=True)
        deps.mkdir(parents=True, exist_ok=True)

        checkpoint_dst = save_dir / _DEFAULT_CHECKPOINT
        if not checkpoint_dst.exists():
            try:
                checkpoint_dst.hardlink_to(Path(self.checkpoint_path))
            except OSError:
                shutil.copy2(self.checkpoint_path, checkpoint_dst)
        np.save(assets / "mean.npy", self.mean.detach().cpu().numpy())
        np.save(assets / "std.npy", self.std.detach().cpu().numpy())

        src_t5 = Path(self.t5_model_path)
        if src_t5.exists():
            for name in (
                "config.json",
                "tokenizer.json",
                "spiece.model",
                "tokenizer_config.json",
                "special_tokens_map.json",
                "generation_config.json",
            ):
                src = src_t5 / name
                if src.exists():
                    shutil.copy2(src, deps / name)
        (save_dir / "motius_config.json").write_text(
            json.dumps(
                {
                    "model": "MotionGPT",
                    "checkpoint": _DEFAULT_CHECKPOINT,
                    "representation": "hml263",
                    "prompt_mode": self.prompt_mode,
                    "embedding_policy": "shared_encoder_decoder_untied_lm_head",
                },
                indent=2,
            )
        )
        (save_dir / "model_index.json").write_text(
            json.dumps(
                {
                    "_class_name": "MotionGPTPipeline",
                    "_library_name": "motius",
                    "model_type": "motiongpt",
                    "format": "motius-motiongpt-artifact-v1",
                    "bundle_class": "motius.models.motiongpt.bundle.MotionGPTBundle",
                    "pipeline_class": "motius.pipelines.motiongpt.pipeline.MotionGPTPipeline",
                    "artifacts": {
                        "checkpoint": _DEFAULT_CHECKPOINT,
                        "stats": "assets/meta",
                        "flan_t5_base": "deps/flan-t5-base",
                    },
                    "api": {
                        "from_pretrained": "motius.pipelines.motiongpt.MotionGPTPipeline.from_pretrained",
                    },
                },
                indent=2,
            )
        )

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use MotionGPTPipeline.infer_t2m for inference.")
