"""Checkpoint utilities."""

import os
import glob
import re
from typing import Optional

import torch


def find_latest_checkpoint(work_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in work_dir.

    Looks for directories named ``checkpoint-iter_N``, ``checkpoint-epoch_N``,
    or the legacy ``checkpoint-N`` format.

    Sorting priority:
      1. If ``meta.pt`` exists inside the checkpoint dir, use ``global_step``
         from the saved metadata (most reliable across formats).
      2. Otherwise fall back to parsing the directory name.

    Returns the path with the highest step, or None if not found.
    """
    if not os.path.isdir(work_dir):
        return None

    # Look for checkpoint directories
    pattern = os.path.join(work_dir, 'checkpoint-*')
    candidates = [c for c in glob.glob(pattern) if os.path.isdir(c)]

    if not candidates:
        return None

    def extract_step(path):
        # Try to read meta.pt for accurate ordering
        meta_path = os.path.join(path, 'meta.pt')
        if os.path.exists(meta_path):
            try:
                meta = torch.load(meta_path, map_location='cpu', weights_only=False)
                return meta.get('global_step', -1)
            except Exception:
                pass
        # Fallback: parse directory name
        basename = os.path.basename(path)
        for pat in [
            r'checkpoint-iter_(\d+)$',
            r'checkpoint-epoch_(\d+)$',
            r'checkpoint-(\d+)$',
        ]:
            m = re.match(pat, basename)
            if m:
                return int(m.group(1))
        return -1

    latest = max(candidates, key=extract_step)
    if extract_step(latest) < 0:
        return None
    return latest


def _unwrap_legacy_checkpoint(data: dict) -> dict:
    """Unwrap legacy checkpoint formats to a flat/nested state_dict.

    Supported wrappers:
      - MMEngine / OpenMMLab: ``{state_dict: {...}, optimizer: {...}, meta: {...}}``
      - HunyuanMotion / PyTorch Lightning: ``{model_state_dict: {...}, epoch: ..., global_step: ...}``

    This function extracts just the model weights part so it can be consumed by
    ``ModelBundle.load_state_dict_selective()``.
    """
    if 'state_dict' in data and isinstance(data['state_dict'], dict):
        # Looks like MMEngine / OpenMMLab format — unwrap
        return data['state_dict']
    if 'model_state_dict' in data and isinstance(data['model_state_dict'], dict):
        # Looks like HunyuanMotion / PyTorch Lightning format — unwrap
        return data['model_state_dict']
    return data


def load_checkpoint(path: str, map_location='cpu') -> dict:
    """Load a checkpoint file (safetensors or pytorch).

    Handles multiple formats:
      - HF-Trainer ``model.pt`` (nested or flat state dict)
      - safetensors files
      - MMEngine ``.pth`` files (auto-unwraps ``ckpt['state_dict']``)
      - pytorch ``pytorch_model.bin``
    """
    import torch

    if os.path.isfile(path):
        if path.endswith('.safetensors'):
            from safetensors.torch import load_file
            return load_file(path, device=map_location)
        else:
            data = torch.load(path, map_location=map_location, weights_only=False)
            return _unwrap_legacy_checkpoint(data)
    elif os.path.isdir(path):
        # Prefer HF-Trainer selective model weights over accelerator state files.
        pt_path = os.path.join(path, 'model.pt')
        if os.path.exists(pt_path):
            import torch
            data = torch.load(pt_path, map_location=map_location, weights_only=False)
            return _unwrap_legacy_checkpoint(data)
        st_path = os.path.join(path, 'model.safetensors')
        if os.path.exists(st_path):
            from safetensors.torch import load_file
            return load_file(st_path, device=map_location)
        pt_path = os.path.join(path, 'pytorch_model.bin')
        if os.path.exists(pt_path):
            import torch
            data = torch.load(pt_path, map_location=map_location, weights_only=False)
            return _unwrap_legacy_checkpoint(data)
    raise FileNotFoundError(f"No checkpoint found at: {path}")


def save_checkpoint(state_dict: dict, path: str, use_safetensors: bool = True):
    """Save a state dict to path."""
    import torch
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)

    if use_safetensors and path.endswith('.safetensors'):
        from safetensors.torch import save_file
        # safetensors requires flat dict with tensor values only
        flat = {}
        for k, v in state_dict.items():
            if isinstance(v, dict):
                for kk, vv in v.items():
                    flat[f"{k}.{kk}"] = vv
            else:
                flat[k] = v
        save_file(flat, path)
    else:
        torch.save(state_dict, path)
