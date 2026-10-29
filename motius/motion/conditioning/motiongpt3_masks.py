from torch import Tensor
import torch
from typing import List

def lengths_to_mask(lengths: List[int], device: torch.device, max_len: int=None) -> Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
