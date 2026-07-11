"""Tensor-related dataset transforms."""

from __future__ import annotations

import os

import torch

from hftrainer.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadOptionalTorchTensor:
    """Load a `.pt` tensor if the path field is present."""

    def __init__(self, file_key: str, output_key: str, root_key: str = 'data_root'):
        self.file_key = file_key
        self.output_key = output_key
        self.root_key = root_key

    def __call__(self, results: dict) -> dict:
        maybe_path = results.get(self.file_key)
        if not maybe_path:
            results[self.output_key] = None
            return results

        full_path = maybe_path
        if not os.path.isabs(full_path):
            data_root = results.get(self.root_key, '')
            full_path = os.path.join(data_root, full_path)

        if not os.path.exists(full_path):
            results[self.output_key] = None
            return results

        results[self.output_key] = torch.load(full_path, map_location='cpu')
        return results
