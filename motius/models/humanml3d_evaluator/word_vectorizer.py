"""GloVe vectorizer used by the released HumanML3D matching network."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


POS_ENUMERATOR = {
    "VERB": 0,
    "NOUN": 1,
    "DET": 2,
    "ADP": 3,
    "NUM": 4,
    "AUX": 5,
    "PRON": 6,
    "ADJ": 7,
    "ADV": 8,
    "Loc_VIP": 9,
    "Body_VIP": 10,
    "Obj_VIP": 11,
    "Act_VIP": 12,
    "Desc_VIP": 13,
    "OTHER": 14,
}

VIP_WORDS = {
    "Loc_VIP": (
        "left", "right", "clockwise", "counterclockwise", "anticlockwise",
        "forward", "back", "backward", "up", "down", "straight", "curve",
    ),
    "Body_VIP": (
        "arm", "chin", "foot", "feet", "face", "hand", "mouth", "leg",
        "waist", "eye", "knee", "shoulder", "thigh",
    ),
    "Obj_VIP": (
        "stair", "dumbbell", "chair", "window", "floor", "car", "ball",
        "handrail", "baseball", "basketball",
    ),
    "Act_VIP": (
        "walk", "run", "swing", "pick", "bring", "kick", "put", "squat",
        "throw", "hop", "dance", "jump", "turn", "stumble", "stop", "sit",
        "lift", "lower", "raise", "wash", "stand", "kneel", "stroll", "rub",
        "bend", "balance", "flap", "jog", "shuffle", "lean", "rotate", "spin",
        "spread", "climb",
    ),
    "Desc_VIP": (
        "slowly", "carefully", "fast", "careful", "slow", "quickly", "happy",
        "angry", "sad", "happily", "angrily", "sadly",
    ),
}


class HumanML3DWordVectorizer:
    def __init__(self, root: Path) -> None:
        with (root / "our_vab_words.pkl").open("rb") as handle:
            words = pickle.load(handle)
        with (root / "our_vab_idx.pkl").open("rb") as handle:
            word_to_index = pickle.load(handle)
        vectors = np.load(root / "our_vab_data.npy")
        self.word_to_vector = {
            str(word): vectors[int(word_to_index[word])] for word in words
        }

    @staticmethod
    def _pos_vector(pos: str) -> np.ndarray:
        value = np.zeros(len(POS_ENUMERATOR), dtype=np.float32)
        value[POS_ENUMERATOR.get(pos, POS_ENUMERATOR["OTHER"])] = 1.0
        return value

    def __getitem__(self, token: str):
        word, pos = token.rsplit("/", 1)
        if word not in self.word_to_vector:
            return self.word_to_vector["unk"], self._pos_vector("OTHER")
        vip_pos = next(
            (name for name, values in VIP_WORDS.items() if word in values),
            pos,
        )
        return self.word_to_vector[word], self._pos_vector(vip_pos)


__all__ = ["HumanML3DWordVectorizer", "POS_ENUMERATOR"]
