"""Self-contained TM2T motion-captioning model bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.models.tm2t.network import Quantizer, TransformerV2, VQEncoderV3
from motius.registry import MODEL_BUNDLES
from motius.configuration.tm2t import (
    _first_existing,
    _download_artifact,
    vocabulary_metadata,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "tm2t"


@MODEL_BUNDLES.register_module()
class TM2TBundle(ModelBundle):
    """TM2T HumanML3D tokenizer and motion-to-text Transformer."""

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        *,
        device: str = "cuda",
        beam_size: Optional[int] = None,
        max_text_length: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
        portable = (artifact / "tm2t_config.json").is_file()
        metadata = {}
        if portable:
            import json

            metadata = json.loads((artifact / "tm2t_config.json").read_text())
            tokenizer_checkpoint = artifact / "vq_encoder.safetensors"
            caption_checkpoint = artifact / "transformer.safetensors"
        else:
            tokenizer_checkpoint = _first_existing(
                artifact,
                [
                    "tokenizer.tar",
                    "VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar",
                    "t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar",
                    "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar",
                ],
            )
            caption_checkpoint = _first_existing(
                artifact,
                [
                    "m2t_transformer.tar",
                    "M2T_EL4_DL4_NH8_PS/model/finest.tar",
                    "t2m/M2T_EL4_DL4_NH8_PS/model/finest.tar",
                    "checkpoints/t2m/M2T_EL4_DL4_NH8_PS/model/finest.tar",
                ],
            )
        beam_size = int(
            beam_size if beam_size is not None else metadata.get("beam_size", 2)
        )
        max_text_length = int(
            max_text_length
            if max_text_length is not None
            else metadata.get("max_text_length", 30)
        )
        mean_path = _first_existing(
            artifact,
            [
                "stats/mean.npy",
                "VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy",
                "t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy",
                "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/mean.npy",
            ],
        )
        std_path = _first_existing(
            artifact,
            [
                "stats/std.npy",
                "VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy",
                "t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy",
                "checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta/std.npy",
            ],
        )

        vocabulary_size, text_start, text_end = vocabulary_metadata(artifact)
        self.vq_encoder = VQEncoderV3(259, [1024, 1024], 2)
        self.quantizer = Quantizer(1024, 1024, 1.0)
        if portable:
            from safetensors.torch import load_file

            tokenizer_state = {
                "vq_encoder": load_file(str(tokenizer_checkpoint)),
                "quantizer": load_file(str(artifact / "quantizer.safetensors")),
            }
        else:
            tokenizer_state = torch.load(
                str(tokenizer_checkpoint), map_location="cpu", weights_only=False
            )
        self.vq_encoder.load_state_dict(tokenizer_state["vq_encoder"], strict=True)
        self.quantizer.load_state_dict(tokenizer_state["quantizer"], strict=True)

        self.motion_start_index = 1024
        self.motion_end_index = 1025
        self.motion_pad_index = 1026
        self.text_start_index = text_start
        self.text_end_index = text_end
        self.text_pad_index = vocabulary_size
        self.transformer = TransformerV2(
            1027,
            self.motion_pad_index,
            vocabulary_size + 1,
            self.text_pad_index,
            d_src_word_vec=512,
            d_trg_word_vec=512,
            d_model=512,
            d_inner=2048,
            n_enc_layers=4,
            n_dec_layers=4,
            n_head=8,
            d_k=64,
            d_v=64,
            dropout=0.1,
            n_src_position=100,
            n_trg_position=50,
            trg_emb_prj_weight_sharing=True,
        )
        caption_state = (
            {"m2t_transformer": load_file(str(caption_checkpoint))}
            if portable
            else torch.load(
                str(caption_checkpoint), map_location="cpu", weights_only=False
            )
        )
        self.transformer.load_state_dict(caption_state["m2t_transformer"], strict=True)
        self.beam_size = int(beam_size)
        self.max_text_length = int(max_text_length)
        self._save_ckpt_modules = ["vq_encoder", "quantizer", "transformer"]
        self._trainable_modules = ["transformer"]
        self._frozen_modules = ["vq_encoder", "quantizer"]
        self.register_buffer("mean", torch.from_numpy(np.load(mean_path)))
        self.register_buffer("std", torch.from_numpy(np.load(std_path)))
        self.artifact_dir = artifact
        self.load_report = {
            "tokenizer_checkpoint": str(tokenizer_checkpoint),
            "caption_checkpoint": str(caption_checkpoint),
            "beam_size": int(beam_size),
            "max_text_length": int(max_text_length),
        }
        resolved_device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.to(resolved_device).eval()

    @property
    def device(self) -> torch.device:
        return self.mean.device

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        return cls(
            artifact_dir=str(_download_artifact(pretrained_model_name_or_path)),
            **kwargs,
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        from motius.configuration.tm2t import save_bundle

        return save_bundle(self, save_directory)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use TM2TPipeline.infer_m2t for inference.")


__all__ = ["TM2TBundle"]
