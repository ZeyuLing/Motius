"""TM2T released vocabulary lookup and token-to-word decoding."""

from pathlib import Path
import pickle
from motius.configuration.tm2t import _first_existing


class TM2TVocabulary:
    """Index mapping from TM2T's released ``our_vab`` GloVe assets."""

    def __init__(self, root: Path) -> None:
        words_path = _first_existing(
            root, ["vocab/our_vab_words.pkl", "glove/our_vab_words.pkl"]
        )
        index_path = _first_existing(
            root, ["vocab/our_vab_idx.pkl", "glove/our_vab_idx.pkl"]
        )
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


def normalize_features(motion, mean, std, *, device):
    """Original dataset NumPy normalization, then the original trainer's FP32 cast."""
    import numpy as np
    import torch
    values = motion.detach().cpu().numpy() if isinstance(motion, torch.Tensor) else np.asarray(motion)
    normalized = (values - mean.detach().cpu().numpy()) / std.detach().cpu().numpy()
    return torch.as_tensor(normalized, dtype=torch.float32, device=device)
