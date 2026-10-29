"""Official HumanML3D matching evaluator preprocessing and neural inference."""

from typing import Sequence, Union
import numpy as np
import torch
from motius.datasets.motion.humanml3d_word_vectorizer import HumanML3DWordVectorizer
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class HumanML3DMatchingPipeline(BasePipeline):
    BUNDLE_CLS = "motius.models.humanml3d_evaluator.HumanML3DMatchingBundle"

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, **kwargs)
        artifact = bundle.artifact_dir
        glove = next(
            (
                artifact / name
                for name in ("glove", "vocab")
                if (artifact / name).exists()
            ),
            None,
        )
        if glove is None:
            raise FileNotFoundError(
                f"HumanML3D evaluator is missing glove/vocab: {artifact}"
            )
        self.word_vectorizer = HumanML3DWordVectorizer(glove)
        try:
            import spacy

            self.nlp = spacy.load(bundle.spacy_model)
        except Exception as exc:
            raise RuntimeError(
                f"HumanML3D matching requires spaCy model {bundle.spacy_model!r}"
            ) from exc

    def _text_tokens_from_doc(self, doc) -> tuple[list[str], int]:
        tokens = []
        for token in doc:
            word = token.text
            if not word.isalpha():
                continue
            if token.pos_ in {"NOUN", "VERB"} and word != "left":
                word = token.lemma_
            tokens.append(f"{word}/{token.pos_}")
        if len(tokens) < self.bundle.max_text_length:
            tokens = ["sos/OTHER", *tokens, "eos/OTHER"]
            length = len(tokens)
            tokens.extend(
                ["unk/OTHER"] * (self.bundle.max_text_length + 2 - len(tokens))
            )
        else:
            tokens = ["sos/OTHER", *tokens[: self.bundle.max_text_length], "eos/OTHER"]
            length = len(tokens)
        return tokens, length

    def _text_tokens(self, sentence: str) -> tuple[list[str], int]:
        return self._text_tokens_from_doc(self.nlp(str(sentence).replace("-", "")))

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
            self.bundle.device, dtype=torch.float32
        )
        pos_one_hot = torch.from_numpy(np.stack(pos_rows)[order]).to(
            self.bundle.device, dtype=torch.float32
        )
        caption_lengths = torch.tensor(
            np.asarray(lengths)[order], device=self.bundle.device, dtype=torch.long
        )
        embeddings = self.bundle.encode_text_tokens(
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
            value = torch.as_tensor(motion).detach().cpu()
            if value.ndim != 2 or value.shape[-1] != 263:
                raise ValueError(
                    f"Expected HumanML3D-263 shape (T,263), got {tuple(value.shape)}"
                )
            length = min(int(value.shape[0]), self.bundle.max_motion_length)
            length = length // self.bundle.unit_length * self.bundle.unit_length
            if length < self.bundle.unit_length:
                raise ValueError("Motion is too short for the HumanML3D evaluator.")
            normalized = (
                value[:length] - self.bundle.mean.cpu()
            ) / self.bundle.std.cpu()
            padded = torch.zeros(
                self.bundle.max_motion_length, 263, dtype=torch.float32
            )
            padded[:length] = normalized
            rows.append(padded)
            lengths.append(length)

        order = np.argsort(lengths)[::-1].copy()
        inverse = np.argsort(order)
        batch = torch.stack(rows)[order].to(self.bundle.device)
        motion_lengths = torch.tensor(
            np.asarray(lengths)[order] // self.bundle.unit_length,
            device=self.bundle.device,
            dtype=torch.long,
        )
        embeddings = self.bundle.encode_motion_features(batch, motion_lengths)
        return embeddings[inverse].cpu().numpy()
