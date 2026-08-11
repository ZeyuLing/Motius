from collections.abc import Mapping, Sequence
from typing import Any, Sequence as SeqType
import torch
import numpy as np
from torch.utils.data._utils.collate import default_collate as torch_default_collate

from mmengine.structures import BaseDataElement


def flexible_collate(data_batch: SeqType[Any]) -> Any:
    """
    A “flexible” collate_fn for PyTorch DataLoader with full support for nested structures
    and graceful handling of unstackable types.

    Behavior:
      1) torch.Tensor → if all elements have identical shape, returns torch.stack(...);
         otherwise returns the raw list.
      2) numpy.ndarray → if all elements have identical shape, converts to Tensor and stacks;
         otherwise returns the raw list.
      3) torch-friendly types (int, float) → stacked into a Tensor of shape (batch_size,).
      4) dict-like mappings → collated per-key, recursively.
      5) list/tuple sequences (excluding str/bytes):
         • If all elements have the same length, transposes and collates element-wise.
         • If they differ in length, returns the raw list.
      6) str, bytes, BaseDataElement, or any other unsupported types → returned as a Python list.
      7) On any unexpected error in the final fallback, returns a Python list.

    Args:
        data_batch (Sequence[Any]): A batch of samples from the Dataset.

    Returns:
        Any: A batched structure matching the input format.
    """
    # If batch contains None values, return as raw list (e.g. from PackInputs dummy_value=None)
    if any(x is None for x in data_batch):
        return list(data_batch)

    first = data_batch[0]
    first_type = type(first)

    # -------------------------------------------------------------------------
    # 1) Explicitly handle torch.Tensor
    # -------------------------------------------------------------------------
    if isinstance(first, torch.Tensor):
        # Check that all tensors have the same shape
        all_shapes = [
            tuple(x.shape) if isinstance(x, torch.Tensor) else None for x in data_batch
        ]
        if len(set(all_shapes)) == 1:
            return torch.stack(data_batch, dim=0)
        else:
            # fallback: return raw list if shapes differ
            return list(data_batch)

    # -------------------------------------------------------------------------
    # 2) Explicitly handle numpy.ndarray
    # -------------------------------------------------------------------------
    if isinstance(first, np.ndarray):
        all_shapes = [x.shape for x in data_batch]
        if len(set(all_shapes)) == 1:
            # convert each to Tensor, then stack
            tensors = [torch.as_tensor(x) for x in data_batch]
            return torch.stack(tensors, dim=0)
        else:
            return list(data_batch)

    # -------------------------------------------------------------------------
    # 3) Treat BaseDataElement, str, bytes as atomic → list
    # -------------------------------------------------------------------------
    if (
        BaseDataElement is not None and isinstance(first, BaseDataElement)
    ) or isinstance(first, (str, bytes)):
        return list(data_batch)

    # -------------------------------------------------------------------------
    # 4) Namedtuple: preserve type, collate field-wise
    # -------------------------------------------------------------------------
    if isinstance(first, tuple) and hasattr(first, "_fields"):
        return first_type(*(flexible_collate(samples) for samples in zip(*data_batch)))

    # -------------------------------------------------------------------------
    # 5) General Sequence (list/tuple but not str/bytes)
    # -------------------------------------------------------------------------
    if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
        if all(
            isinstance(x, Sequence)
            and not isinstance(x, (str, bytes))
            and all(isinstance(y, str) for y in x)
            for x in data_batch
        ):
            return list(data_batch)
        lengths = [len(x) for x in data_batch]
        if len(set(lengths)) == 1:
            # Transpose and collate element-wise
            transposed = list(zip(*data_batch))
            if isinstance(first, tuple):
                return tuple(flexible_collate(samples) for samples in transposed)
            else:
                return [flexible_collate(samples) for samples in transposed]
        else:
            # Mixed-length sequences: return raw list
            return list(data_batch)

    # -------------------------------------------------------------------------
    # 6) Mapping (dict-like): collate each value under the same key
    # -------------------------------------------------------------------------
    if isinstance(first, Mapping):
        return first_type(
            {key: flexible_collate([d[key] for d in data_batch]) for key in first}
        )

    # -------------------------------------------------------------------------
    # 7) Fallback: try PyTorch default_collate, else list
    # -------------------------------------------------------------------------
    try:
        return torch_default_collate(data_batch)
    except Exception:

        return list(data_batch)
