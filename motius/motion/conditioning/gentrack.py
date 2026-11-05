from typing import List
import torch
from torch import Tensor

def _len_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)

def pack_context(self, text_ctxt: List[Tensor], ctxt_len: Tensor, device, dtype):
    """List of (seq_i, 4096) -> padded (B, max_seq, 4096) + bool mask."""
    B = len(text_ctxt)
    max_seq = max((int(c.shape[0]) for c in text_ctxt))
    ctxt = torch.zeros(B, max_seq, self._ctxt_input_dim, dtype=dtype, device=device)
    for i, c in enumerate(text_ctxt):
        ctxt[i, :c.shape[0]] = c.to(device, dtype)
    return (ctxt, _len_to_mask(ctxt_len.to(device), max_seq))
