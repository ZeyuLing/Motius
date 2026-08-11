"""Filter samples to ensure correct motion dimensions."""

from typing import Optional
import logging
import torch
import numpy as np
from mmcv.transforms import BaseTransform
from motius.registry import TRANSFORMS


logger = logging.getLogger(__name__)


@TRANSFORMS.register_module()
class EnsureDimensionFilter(BaseTransform):
    """Filter and ensure motion tensors have the expected dimension.

    This transform checks if the motion tensor has the expected dimension.
    If not, it raises a ValueError to skip this sample during data loading.

    Args:
        key: Key to the motion tensor in the data dict
        expected_dim: Expected motion feature dimension (default: 151)
    """

    def __init__(self, key: str = 'motion', expected_dim: int = 151):
        self.key = key
        self.expected_dim = expected_dim

    def transform(self, results: dict) -> Optional[dict]:
        """Filter motion by dimension.

        Returns None to skip this sample if dimension doesn't match.
        """
        motion = results.get(self.key)
        if motion is None:
            return results

        # Get the feature dimension (last dimension)
        if isinstance(motion, (list, tuple)):
            # Check all elements
            dims = []
            for item in motion:
                if hasattr(item, 'shape'):
                    dims.append(item.shape[-1])
            if any(d != self.expected_dim for d in dims):
                logger.warning(f"Skipping sample due to mixed/wrong motion dimensions: {dims} != {self.expected_dim}")
                return None  # Skip this sample
        elif isinstance(motion, (torch.Tensor, np.ndarray)):
            if motion.shape[-1] != self.expected_dim:
                logger.warning(f"Skipping sample due to wrong motion dimension: {motion.shape[-1]} != {self.expected_dim}")
                return None  # Skip this sample

        return results
