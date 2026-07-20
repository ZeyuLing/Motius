"""Self-contained TM2T motion-captioning model bundle."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.models.tm2t.network import Quantizer, TransformerV2, Translator, VQEncoderV3
from motius.registry import MODEL_BUNDLES


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "tm2t"


def _download_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.is_dir():
        return path
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


def _first_existing(root: Path, candidates) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"TM2T artifact is missing all expected paths: {', '.join(map(str, candidates))}"
    )


class TM2TVocabulary:
    """Index mapping from TM2T's released ``our_vab`` GloVe assets."""

    def __init__(self, root: Path) -> None:
        words_path = _first_existing(root, ["vocab/our_vab_words.pkl", "glove/our_vab_words.pkl"])
        index_path = _first_existing(root, ["vocab/our_vab_idx.pkl", "glove/our_vab_idx.pkl"])
        with words_path.open("rb") as handle:
            words = pickle.load(handle)
        with index_path.open("rb") as handle:
            self.word_to_index = pickle.load(handle)
        self.index_to_word = {
            int(self.word_to_index[word]): str(word) for word in words
        }

    def __len__(self) -> int:
        return len(self.word_to_index)

    def index(self, token: str) -> int:
        return int(self.word_to_index[token])

    def token(self, index: int) -> str:
        if int(index) == len(self):
            return "pad"
        return self.index_to_word.get(int(index), "unk")


@MODEL_BUNDLES.register_module()
class TM2TBundle(ModelBundle):
    """TM2T HumanML3D tokenizer and motion-to-text Transformer."""

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        *,
        device: str = "cuda",
        beam_size: int = 2,
        max_text_length: int = 30,
        **kwargs,
    ) -> None:
        super().__init__()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
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

        self.vocabulary = TM2TVocabulary(artifact)
        self.vq_encoder = VQEncoderV3(259, [1024, 1024], 2)
        self.quantizer = Quantizer(1024, 1024, 1.0)
        tokenizer_state = torch.load(str(tokenizer_checkpoint), map_location="cpu")
        self.vq_encoder.load_state_dict(tokenizer_state["vq_encoder"], strict=True)
        self.quantizer.load_state_dict(tokenizer_state["quantizer"], strict=True)

        self.motion_start_index = 1024
        self.motion_end_index = 1025
        self.motion_pad_index = 1026
        self.text_start_index = self.vocabulary.index("sos")
        self.text_end_index = self.vocabulary.index("eos")
        self.text_pad_index = len(self.vocabulary)
        self.transformer = TransformerV2(
            1027,
            self.motion_pad_index,
            len(self.vocabulary) + 1,
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
        caption_state = torch.load(str(caption_checkpoint), map_location="cpu")
        self.transformer.load_state_dict(caption_state["m2t_transformer"], strict=True)
        self.translator = Translator(
            self.transformer,
            beam_size=int(beam_size),
            max_seq_len=int(max_text_length),
            src_pad_idx=self.motion_pad_index,
            trg_pad_idx=self.text_pad_index,
            trg_sos_idx=self.text_start_index,
            trg_eos_idx=self.text_end_index,
        )
        self.register_buffer("mean", torch.from_numpy(np.load(mean_path)).float())
        self.register_buffer("std", torch.from_numpy(np.load(std_path)).float())
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
        return cls(artifact_dir=str(_download_artifact(pretrained_model_name_or_path)), **kwargs)

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use TM2TPipeline.infer_m2t for inference.")


__all__ = ["TM2TBundle", "TM2TVocabulary"]
