"""Self-contained official HumanML3D text-motion matching evaluator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

from .network import MotionEncoderBiGRUCo, MovementConvEncoder, TextEncoderBiGRUCo
from .word_vectorizer import HumanML3DWordVectorizer, POS_ENUMERATOR


_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "evaluators" / "humanml3d_263"


def _download_artifact(name_or_path: str) -> Path:
    path = Path(name_or_path).expanduser()
    if path.is_dir():
        return path.resolve()
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))


def _first_existing(root: Path, candidates: Sequence[str]) -> Path:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    raise FileNotFoundError(
        f"HumanML3D evaluator artifact is missing: {', '.join(candidates)}"
    )


@MODEL_BUNDLES.register_module()
class HumanML3DMatchingBundle(ModelBundle):
    """Official Guo et al. HumanML3D-263 matching network.

    Public inputs are unnormalized HumanML3D-263 motions and raw English text.
    Both embedding methods preserve input order, including variable-length
    batches, so downstream candidate batching cannot silently break pairs.
    """

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        *,
        device: str = "cuda",
        spacy_model: str = "en_core_web_sm",
        max_text_length: int = 20,
        max_motion_length: int = 196,
        unit_length: int = 4,
    ) -> None:
        super().__init__()
        artifact = Path(artifact_dir or _DEFAULT_ARTIFACT).expanduser().resolve()
        checkpoint_path = _first_existing(
            artifact,
            (
                "text_mot_match.tar",
                "text_mot_match/model/finest.tar",
                "t2m/text_mot_match/model/finest.tar",
                "checkpoints/t2m/text_mot_match/model/finest.tar",
            ),
        )
        mean_path = _first_existing(
            artifact,
            (
                "meta/mean.npy",
                "t2m/Comp_v6_KLD005/meta/mean.npy",
                "checkpoints/t2m/Comp_v6_KLD005/meta/mean.npy",
            ),
        )
        std_path = _first_existing(
            artifact,
            (
                "meta/std.npy",
                "t2m/Comp_v6_KLD005/meta/std.npy",
                "checkpoints/t2m/Comp_v6_KLD005/meta/std.npy",
            ),
        )
        glove_dir = _first_existing(artifact, ("glove", "vocab"))

        self.movement_encoder = MovementConvEncoder(259, 512, 512)
        self.text_encoder = TextEncoderBiGRUCo(
            word_size=300,
            pos_size=len(POS_ENUMERATOR),
            hidden_size=512,
            output_size=512,
        )
        self.motion_encoder = MotionEncoderBiGRUCo(512, 1024, 512)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        self.movement_encoder.load_state_dict(checkpoint["movement_encoder"], strict=True)
        self.text_encoder.load_state_dict(checkpoint["text_encoder"], strict=True)
        self.motion_encoder.load_state_dict(checkpoint["motion_encoder"], strict=True)
        self.register_buffer("mean", torch.from_numpy(np.load(mean_path)).float())
        self.register_buffer("std", torch.from_numpy(np.load(std_path)).float())
        self.word_vectorizer = HumanML3DWordVectorizer(glove_dir)
        try:
            import spacy

            self.nlp = spacy.load(spacy_model)
        except Exception as exc:
            raise RuntimeError(
                f"HumanML3D matching evaluation requires spaCy model {spacy_model!r}."
            ) from exc
        self.max_text_length = int(max_text_length)
        self.max_motion_length = int(max_motion_length)
        self.unit_length = int(unit_length)
        self.artifact_dir = artifact
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

    def train(self, mode: bool = True):
        return super().train(False)

    def _text_tokens_from_doc(self, doc) -> tuple[list[str], int]:
        tokens = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if token.pos_ in {"NOUN", "VERB"} and word != "left":
                word = token.lemma_
            tokens.append(f"{word.lower()}/{token.pos_}")
        tokens = tokens or ["unk/OTHER"]
        if len(tokens) < self.max_text_length:
            tokens = ["sos/OTHER", *tokens, "eos/OTHER"]
            length = len(tokens)
            tokens.extend(
                ["unk/OTHER"] * (self.max_text_length + 2 - len(tokens))
            )
        else:
            tokens = ["sos/OTHER", *tokens[: self.max_text_length], "eos/OTHER"]
            length = len(tokens)
        return tokens, length

    def _text_tokens(self, sentence: str) -> tuple[list[str], int]:
        return self._text_tokens_from_doc(
            self.nlp(str(sentence).replace("-", ""))
        )

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 512), dtype=np.float32)
        word_rows, pos_rows, lengths = [], [], []
        documents = self.nlp.pipe(
            (str(text).replace("-", "") for text in texts),
            batch_size=512,
        )
        for document in documents:
            tokens, length = self._text_tokens_from_doc(document)
            word_values, pos_values = zip(
                *(self.word_vectorizer[token] for token in tokens)
            )
            word_rows.append(np.stack(word_values))
            pos_rows.append(np.stack(pos_values))
            lengths.append(length)

        order = np.argsort(lengths)[::-1].copy()
        inverse = np.argsort(order)
        word_embeddings = torch.from_numpy(np.stack(word_rows)[order]).to(
            self.device, dtype=torch.float32
        )
        pos_one_hot = torch.from_numpy(np.stack(pos_rows)[order]).to(
            self.device, dtype=torch.float32
        )
        caption_lengths = torch.tensor(
            np.asarray(lengths)[order], device=self.device, dtype=torch.long
        )
        embeddings = self.text_encoder(
            word_embeddings, pos_one_hot, caption_lengths
        )
        return embeddings[inverse].cpu().numpy()

    @torch.no_grad()
    def encode_motions(
        self, motions: Sequence[Union[np.ndarray, torch.Tensor]]
    ) -> np.ndarray:
        if not motions:
            return np.empty((0, 512), dtype=np.float32)
        rows, lengths = [], []
        for motion in motions:
            value = torch.as_tensor(motion, dtype=torch.float32)
            if value.ndim != 2 or value.shape[-1] != 263:
                raise ValueError(
                    f"Expected HumanML3D-263 shape (T,263), got {tuple(value.shape)}"
                )
            length = min(int(value.shape[0]), self.max_motion_length)
            length = length // self.unit_length * self.unit_length
            if length < self.unit_length:
                raise ValueError("Motion is too short for the HumanML3D evaluator.")
            normalized = (value[:length] - self.mean.cpu()) / self.std.cpu().clamp_min(1e-8)
            padded = torch.zeros(self.max_motion_length, 263, dtype=torch.float32)
            padded[:length] = normalized
            rows.append(padded)
            lengths.append(length)

        order = np.argsort(lengths)[::-1].copy()
        inverse = np.argsort(order)
        batch = torch.stack(rows)[order].to(self.device)
        motion_lengths = torch.tensor(
            np.asarray(lengths)[order] // self.unit_length,
            device=self.device,
            dtype=torch.long,
        )
        movements = self.movement_encoder(batch[..., :-4]).detach()
        embeddings = self.motion_encoder(movements, motion_lengths)
        return embeddings[inverse].cpu().numpy()

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use encode_texts() or encode_motions().")


__all__ = ["HumanML3DMatchingBundle"]
