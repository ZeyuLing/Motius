"""Text preprocessing for the released TM2D vocabulary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


class TM2DTokenizer:
    """Convert captions to TM2D word ids and motion-length indicators."""

    def __init__(
        self,
        vocabulary: Mapping[str, int],
        *,
        max_text_tokens: int = 20,
        length_id: int = 4199,
        pad_id: int = 4200,
        nlp=None,
    ):
        self.vocabulary = {str(key): int(value) for key, value in vocabulary.items()}
        self.max_text_tokens = int(max_text_tokens)
        self.length_id = int(length_id)
        self.pad_id = int(pad_id)
        self._nlp = nlp

    def _load_nlp(self):
        if self._nlp is None:
            import spacy

            try:
                self._nlp = spacy.load("en_core_web_sm")
            except OSError as error:
                raise RuntimeError(
                    "TM2D caption tokenization requires en_core_web_sm. Install it "
                    "with `python -m spacy download en_core_web_sm`."
                ) from error
        return self._nlp

    def _document_words(self, document) -> list[str]:
        words = []
        for token in document:
            word = token.text
            if not word.isalpha():
                continue
            if token.pos_ in {"NOUN", "VERB"} and word != "left":
                word = token.lemma_
            words.append(word)
        return words[: self.max_text_tokens]

    def tokenize(self, caption: str) -> list[str]:
        document = self._load_nlp()(caption.replace("-", ""))
        return self._document_words(document)

    def tokenize_batch(
        self, captions: Sequence[str], *, batch_size: int = 256
    ) -> list[list[str]]:
        texts = [caption.replace("-", "") for caption in captions]
        return [
            self._document_words(document)
            for document in self._load_nlp().pipe(texts, batch_size=batch_size)
        ]

    def encode(
        self,
        captions: str | Sequence[str],
        motion_token_lengths: int | Sequence[int],
        *,
        pretokenized: Sequence[str] | Sequence[Sequence[str]] | None = None,
    ) -> np.ndarray:
        if isinstance(captions, str):
            caption_list = [captions]
        else:
            caption_list = list(captions)
        if isinstance(motion_token_lengths, int):
            length_list = [motion_token_lengths] * len(caption_list)
        else:
            length_list = [int(value) for value in motion_token_lengths]
        if len(length_list) != len(caption_list):
            raise ValueError("captions and motion_token_lengths must have equal length")

        if pretokenized is None:
            words_list = [self.tokenize(caption) for caption in caption_list]
        else:
            tokenized_values = list(pretokenized)
            is_flat = not tokenized_values or isinstance(tokenized_values[0], str)
            if is_flat:
                if len(caption_list) != 1:
                    raise ValueError("A flat pretokenized sequence requires one caption")
                words_list = [tokenized_values]
            else:
                words_list = [list(words) for words in tokenized_values]
            if len(words_list) != len(caption_list):
                raise ValueError("captions and pretokenized must have equal length")

        sequence_length = self.max_text_tokens + 2
        encoded = np.full((len(caption_list), sequence_length), self.pad_id, dtype=np.int64)
        unknown = self.vocabulary["unk"]
        for row, (words, motion_length) in enumerate(zip(words_list, length_list)):
            lexical = ["sos", *words[: self.max_text_tokens], "eos"]
            sent_length = len(lexical)
            encoded[row, :sent_length] = [
                self.vocabulary.get(word, unknown) for word in lexical
            ]
            indicator_end = min(max(int(motion_length), sent_length), sequence_length)
            encoded[row, sent_length:indicator_end] = self.length_id
        return encoded


__all__ = ["TM2DTokenizer"]
