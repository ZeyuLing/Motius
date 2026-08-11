"""
Selective T2M-to-M2M v2 Checkpoint Loading.

This module implements selective checkpoint loading from HyMotion-T2M pretrained
weights into MotionCanvas v2 bundle, handling architecture differences:

- **Reusable modules** (exact shape match):
  - Text encoders (ctxt_encoder, vtxt_encoder, text_refiner)
  - Timestep encoder
  - Transformer blocks (double_blocks, single_blocks)

- **Adapted modules** (shape mismatch, deterministically remapped):
  - input_encoder: 201 -> 594 (M2M input is [x_t, edit_context, target_mask])
  - final_layer: 201 -> 198 (M2M drops pelvis RIC from position block)

- **Bundle-level condition parameters** (loaded when present):
  - null_vtxt_feat, null_ctxt_input
  - special_game_vtxt_feat, special_game_ctxt_feat
- **Bundle-level statistics** (NOT loaded from T2M):
  - mean, std: M2M uses strict 198-dim stats derived from T2M 201-dim stats

**Usage**:
  from motius.models.motioncanvas.checkpoint_loading import load_t2m_pretrained_selective

  stats = load_t2m_pretrained_selective(
      bundle=m2m_bundle,
      t2m_checkpoint_path='checkpoints/HY-Motion-1.0/HY-Motion-1.0-Lite/latest.ckpt',
      freeze_strategy='encoders'  # 'none', 'encoders', 'text_refiner', 'full'
  )
  print(f"Loaded {stats['modules_loaded']} modules, "
        f"skipped {stats['modules_skipped']}, "
        f"reinitialized {stats['modules_reinitialized']}")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ============================================================================
# Module Path Definitions
# ============================================================================

# Modules that exist in both T2M and M2M with identical structure/shape
REUSABLE_MODULES = {
    'motion_transformer.ctxt_encoder',
    'motion_transformer.vtxt_encoder',
    'motion_transformer.timestep_encoder',
    'motion_transformer.text_refiner',
    'motion_transformer.double_blocks',
    'motion_transformer.single_blocks',
}

# Modules with shape mismatches (skipped, reinitialized)
SHAPE_MISMATCH_MODULES = {
    'motion_transformer.input_encoder',    # 201 -> 594 conditioning input
    'motion_transformer.final_layer',       # 201 -> 198 output dimension
}

# Bundle-level parameters to exclude (use config-initialized values)
EXCLUDED_BUNDLE_PARAMS = {
    'mean',              # M2M config loads strict 198-dim mapped stats.
    'std',               # M2M config loads strict 198-dim mapped stats.
}

REUSABLE_BUNDLE_PARAMS = {
    'null_vtxt_feat',
    'null_ctxt_input',
    'special_game_vtxt_feat',
    'special_game_ctxt_feat',
}


# ============================================================================
# Loading Utilities
# ============================================================================

def _load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
    """
    Load checkpoint from .ckpt or .pt file.

    Returns:
        Checkpoint dictionary with keys like 'model', 'optimizer', etc.
        or direct state_dict if .pt format.
    """
    resolved = Path(checkpoint_path).expanduser()
    if not resolved.exists() and '/' in checkpoint_path:
        from huggingface_hub import snapshot_download

        resolved = Path(snapshot_download(repo_id=checkpoint_path))
    if resolved.is_dir():
        for candidate in (
            'motion_transformer.safetensors',
            'motion_transformer.pt',
            'model.safetensors',
            'model.pt',
        ):
            path = resolved / candidate
            if path.exists():
                resolved = path
                break
        else:
            raise FileNotFoundError(
                f'No supported model weights found under {resolved}'
            )
    checkpoint_path = str(resolved)

    if checkpoint_path.endswith('.ckpt'):
        # PyTorch Lightning checkpoint format
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # HyMotion T2M format: {'model_state_dict': {...}, 'epoch': ..., 'global_step': ...}
            state_dict = checkpoint['model_state_dict']
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            # .ckpt with model/optimizer/trainer_state
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and any(k.startswith('motion_transformer') or k.startswith('text_encoder') for k in checkpoint.keys()):
            # .ckpt is already a state_dict
            state_dict = checkpoint
        else:
            state_dict = checkpoint
    elif checkpoint_path.endswith('.safetensors'):
        from safetensors.torch import load_file

        state_dict = load_file(checkpoint_path)
    elif checkpoint_path.endswith('.pt'):
        # Direct state_dict
        state_dict = torch.load(checkpoint_path, map_location='cpu')
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}. Expected .ckpt or .pt")

    return state_dict


def _filter_reusable_params(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Extract only parameters from reusable modules.

    Args:
        state_dict: Checkpoint state_dict (possibly nested)

    Returns:
        Filtered state_dict containing only reusable module parameters
    """
    filtered = {}

    for key, value in state_dict.items():
        if key in REUSABLE_BUNDLE_PARAMS:
            filtered[key] = value
            continue
        # Check if key starts with any reusable module path
        for reusable_mod in REUSABLE_MODULES:
            if key.startswith(reusable_mod):
                filtered[key] = value
                break

    return filtered


def _motion201_to_m2m198_indices(device=None) -> torch.Tensor:
    """Map HYMotion-Lite 201-dim channels to strict M2M 198-dim channels.

    HYMotion-Lite O6DP-201 is [135 trans+rot, 66 RIC]. M2M keeps the shared
    135-dim trans+rot prefix and uses 21*3 position channels, so the pelvis RIC
    slice old[135:138] is dropped and body-joint positions old[138:201] map to
    new[135:198].
    """
    return torch.cat(
        [
            torch.arange(0, 135, device=device),
            torch.arange(138, 201, device=device),
        ],
        dim=0,
    )


def _adapt_t2m_io_params(
    state_dict: Dict[str, torch.Tensor],
    bundle,
) -> tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
    """Adapt T2M input/output projection tensors to the M2M architecture."""
    model_state = {}
    mt_state = bundle.motion_transformer.state_dict()
    for key, value in mt_state.items():
        model_state[f'motion_transformer.{key}'] = value

    adapted: Dict[str, torch.Tensor] = {}
    stats: Dict[str, Any] = {
        'modules_adapted': [],
        'num_params_adapted': 0,
        'warnings': [],
    }
    idx = _motion201_to_m2m198_indices()

    old_in_w = state_dict.get('motion_transformer.input_encoder.weight')
    new_in_w_ref = model_state.get('motion_transformer.input_encoder.weight')
    if old_in_w is not None and new_in_w_ref is not None:
        if tuple(old_in_w.shape) == (new_in_w_ref.shape[0], 201) and new_in_w_ref.shape[1] >= 198:
            new_in_w = old_in_w.new_zeros(tuple(new_in_w_ref.shape))
            new_in_w[:, :198] = old_in_w[:, idx]
            adapted['motion_transformer.input_encoder.weight'] = new_in_w
            stats['modules_adapted'].append('motion_transformer.input_encoder')
            stats['num_params_adapted'] += new_in_w.numel()
        else:
            stats['warnings'].append(
                'input_encoder.weight not adapted: '
                f'ckpt={tuple(old_in_w.shape)} model={tuple(new_in_w_ref.shape)}'
            )

    old_in_b = state_dict.get('motion_transformer.input_encoder.bias')
    new_in_b_ref = model_state.get('motion_transformer.input_encoder.bias')
    if old_in_b is not None and new_in_b_ref is not None:
        if tuple(old_in_b.shape) == tuple(new_in_b_ref.shape):
            adapted['motion_transformer.input_encoder.bias'] = old_in_b.clone()
            stats['num_params_adapted'] += old_in_b.numel()
        else:
            stats['warnings'].append(
                'input_encoder.bias not adapted: '
                f'ckpt={tuple(old_in_b.shape)} model={tuple(new_in_b_ref.shape)}'
            )

    old_out_w = state_dict.get('motion_transformer.final_layer.linear.weight')
    new_out_w_ref = model_state.get('motion_transformer.final_layer.linear.weight')
    if old_out_w is not None and new_out_w_ref is not None:
        if tuple(old_out_w.shape) == (201, new_out_w_ref.shape[1]) and new_out_w_ref.shape[0] == 198:
            adapted['motion_transformer.final_layer.linear.weight'] = old_out_w[idx, :].clone()
            stats['modules_adapted'].append('motion_transformer.final_layer')
            stats['num_params_adapted'] += adapted[
                'motion_transformer.final_layer.linear.weight'
            ].numel()
        else:
            stats['warnings'].append(
                'final_layer.linear.weight not adapted: '
                f'ckpt={tuple(old_out_w.shape)} model={tuple(new_out_w_ref.shape)}'
            )

    old_out_b = state_dict.get('motion_transformer.final_layer.linear.bias')
    new_out_b_ref = model_state.get('motion_transformer.final_layer.linear.bias')
    if old_out_b is not None and new_out_b_ref is not None:
        if tuple(old_out_b.shape) == (201,) and tuple(new_out_b_ref.shape) == (198,):
            adapted['motion_transformer.final_layer.linear.bias'] = old_out_b[idx].clone()
            stats['num_params_adapted'] += adapted[
                'motion_transformer.final_layer.linear.bias'
            ].numel()
        else:
            stats['warnings'].append(
                'final_layer.linear.bias not adapted: '
                f'ckpt={tuple(old_out_b.shape)} model={tuple(new_out_b_ref.shape)}'
            )

    # The final-layer AdaLN modulation does not depend on motion dimensionality.
    # It must be copied verbatim; leaving it randomly initialized breaks the
    # HYMotion-Lite warm start even if the 201->198 output projection is adapted.
    for suffix in (
        'adaLN_modulation.linear.weight',
        'adaLN_modulation.linear.bias',
    ):
        key = f'motion_transformer.final_layer.{suffix}'
        old = state_dict.get(key)
        new_ref = model_state.get(key)
        if old is None or new_ref is None:
            continue
        if tuple(old.shape) == tuple(new_ref.shape):
            adapted[key] = old.clone()
            stats['modules_adapted'].append('motion_transformer.final_layer')
            stats['num_params_adapted'] += old.numel()
        else:
            stats['warnings'].append(
                f'final_layer.{suffix} not adapted: '
                f'ckpt={tuple(old.shape)} model={tuple(new_ref.shape)}'
            )

    stats['modules_adapted'] = sorted(set(stats['modules_adapted']))
    return adapted, stats


def _get_shape_mismatches(state_dict: Dict[str, torch.Tensor], bundle) -> Dict[str, tuple]:
    """
    Identify parameters with shape mismatches between checkpoint and model.

    Returns:
        Dict mapping parameter name to (ckpt_shape, model_shape)
    """
    mismatches = {}
    model_state = {}
    for key, value in bundle.motion_transformer.state_dict().items():
        model_state[f'motion_transformer.{key}'] = value
    for key, value in bundle.named_parameters(recurse=False):
        model_state[key] = value
    for key, value in bundle.named_buffers(recurse=False):
        model_state[key] = value

    for key, ckpt_value in state_dict.items():
        if key in model_state:
            model_value = model_state[key]
            if ckpt_value.shape != model_value.shape:
                mismatches[key] = (tuple(ckpt_value.shape), tuple(model_value.shape))

    return mismatches


def _count_parameters(module: nn.Module) -> int:
    """Count total parameters in a module."""
    return sum(p.numel() for p in module.parameters())


def _freeze_module(module: nn.Module) -> None:
    """Freeze all parameters in a module."""
    module.requires_grad_(False)


def _unfreeze_module(module: nn.Module) -> None:
    """Unfreeze all parameters in a module."""
    module.requires_grad_(True)


# ============================================================================
# Main Loading Function
# ============================================================================

def load_t2m_pretrained_selective(
    bundle,
    t2m_checkpoint_path: str,
    freeze_strategy: str = 'none',
) -> Dict[str, Any]:
    """
    Selectively load T2M pretrained weights into M2M v2 bundle.

    Handles architecture differences between T2M (input_dim=201) and M2M v2
    (input_dim=594 with motion conditioning). Loads all reusable modules
    (encoders, blocks) and adapts shape-mismatched input/output projections
    (input_encoder, final_layer).

    Args:
        bundle: MotionCanvasBundle instance
        t2m_checkpoint_path: Path to T2M pretrained checkpoint (.ckpt or .pt)
        freeze_strategy: Which modules to freeze after loading:
            - 'none': Don't freeze anything (default)
            - 'encoders': Freeze text encoders only
            - 'text_refiner': Also freeze text_refiner
            - 'blocks': Also freeze all transformer blocks (double + single)
            - 'full': Freeze all reusable modules (encoders + blocks + text_refiner)

    Returns:
        Dict with statistics:
            - 'modules_loaded': List of modules successfully loaded
            - 'modules_skipped': List of modules skipped due to shape mismatch
            - 'modules_reinitialized': List of modules with random reinitialization
            - 'num_params_loaded': Total parameters loaded
            - 'num_params_skipped': Total parameters skipped (shape mismatch)
            - 'num_params_reinitialized': Total parameters reinitialized
            - 'frozen_modules': List of modules frozen per freeze_strategy

    Raises:
        ValueError: If checkpoint_path doesn't exist or freeze_strategy is invalid
        RuntimeError: If checkpoint loading fails
    """
    import os

    if not os.path.exists(t2m_checkpoint_path):
        raise ValueError(f"Checkpoint not found: {t2m_checkpoint_path}")

    freeze_strategies = {'none', 'encoders', 'text_refiner', 'blocks', 'full'}
    if freeze_strategy not in freeze_strategies:
        raise ValueError(
            f"freeze_strategy must be one of {freeze_strategies}, got {freeze_strategy!r}"
        )

    logger.info(f"Loading T2M pretrained checkpoint: {t2m_checkpoint_path}")
    logger.info(f"Freeze strategy: {freeze_strategy}")

    # Load checkpoint
    try:
        t2m_state = _load_checkpoint(t2m_checkpoint_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}") from e

    # Extract reusable module parameters and adapt the IO layers that differ in
    # shape but still have a well-defined T2M-to-M2M mapping.
    reusable_state = _filter_reusable_params(t2m_state)
    adapted_state, adapted_stats = _adapt_t2m_io_params(t2m_state, bundle)
    reusable_state.update(adapted_state)

    if not reusable_state:
        logger.warning("No reusable parameters found in checkpoint. "
                      "This may indicate a format or path issue.")
        return {
            'modules_loaded': [],
            'modules_skipped': [],
            'modules_reinitialized': [],
            'num_params_loaded': 0,
            'num_params_skipped': 0,
            'num_params_reinitialized': 0,
            'frozen_modules': [],
        }

    # Detect shape mismatches
    mismatches = _get_shape_mismatches(reusable_state, bundle)

    # Load reusable parameters (strict=False to skip mismatches).
    # Pass as flat dict — load_state_dict_selective auto-splits
    # "motion_transformer.ctxt_encoder.weight" → module="motion_transformer",
    # param="ctxt_encoder.weight" which is the correct format for load_state_dict.
    bundle.load_state_dict_selective(
        reusable_state,
        strict=False,
        exclude_bundle_keys=list(EXCLUDED_BUNDLE_PARAMS),
    )

    # Track statistics
    stats = {
        'modules_loaded': [],
        'modules_skipped': [],
        'modules_reinitialized': [],
        'modules_adapted': adapted_stats['modules_adapted'],
        'num_params_loaded': 0,
        'num_params_skipped': 0,
        'num_params_reinitialized': 0,
        'num_params_adapted': adapted_stats['num_params_adapted'],
        'frozen_modules': [],
    }

    # Count loaded parameters
    for key, value in reusable_state.items():
        if key not in mismatches:
            stats['num_params_loaded'] += value.numel()
            # Extract module name (e.g., "motion_transformer.double_blocks")
            parts = key.split('.')
            if len(parts) >= 2:
                mod_name = f"{parts[0]}.{parts[1]}"
                if mod_name not in stats['modules_loaded']:
                    stats['modules_loaded'].append(mod_name)
            elif key in REUSABLE_BUNDLE_PARAMS and key not in stats['modules_loaded']:
                stats['modules_loaded'].append(key)

    # Count skipped/mismatched parameters
    for key, value in reusable_state.items():
        if key in mismatches:
            stats['num_params_skipped'] += value.numel()
            parts = key.split('.')
            if len(parts) >= 2:
                mod_name = f"{parts[0]}.{parts[1]}"
                if mod_name not in stats['modules_skipped']:
                    stats['modules_skipped'].append(mod_name)

    # Reinitialize only modules that could not be adapted. A healthy T2M-only
    # warm-start should adapt both input_encoder and final_layer.
    adapted_modules = set(adapted_stats['modules_adapted'])
    for mod_path in SHAPE_MISMATCH_MODULES:
        if mod_path in adapted_modules:
            continue
        try:
            parts = mod_path.split('.')
            if len(parts) == 2:
                parent = getattr(bundle, parts[0], None)
                if parent and hasattr(parent, parts[1]):
                    mod = getattr(parent, parts[1])
                    _reinitialize_module(mod)
                    stats['modules_reinitialized'].append(mod_path)
                    stats['num_params_reinitialized'] += _count_parameters(mod)
        except Exception as e:
            logger.warning(f"Failed to reinitialize {mod_path}: {e}")

    # Apply freezing strategy
    frozen_modules = _apply_freeze_strategy(bundle, freeze_strategy)
    stats['frozen_modules'] = frozen_modules

    # Log summary
    logger.info(
        f"Loaded {len(stats['modules_loaded'])} module types "
        f"({stats['num_params_loaded']:,} params), "
        f"skipped {len(stats['modules_skipped'])} module types "
        f"({stats['num_params_skipped']:,} params), "
        f"reinitialized {len(stats['modules_reinitialized'])} module types "
        f"({stats['num_params_reinitialized']:,} params)"
    )

    if stats['modules_skipped']:
        logger.info(f"Skipped (shape mismatch): {', '.join(stats['modules_skipped'])}")

    if stats['modules_reinitialized']:
        logger.info(f"Reinitialized: {', '.join(stats['modules_reinitialized'])}")

    if stats['modules_adapted']:
        logger.info(
            f"Adapted T2M IO modules: {', '.join(stats['modules_adapted'])} "
            f"({stats['num_params_adapted']:,} params)"
        )

    for warning in adapted_stats.get('warnings', []):
        logger.warning(f"T2M IO adapter: {warning}")

    if stats['frozen_modules']:
        logger.info(f"Frozen (strategy={freeze_strategy}): {', '.join(stats['frozen_modules'])}")

    return stats


# ============================================================================
# Helper Functions
# ============================================================================

def _reinitialize_module(module: nn.Module) -> None:
    """
    Reinitialize all weights in a module using Xavier uniform initialization.

    This is used for layers that don't have direct equivalents in T2M
    (e.g., input_encoder for expanded conditioning input, final_layer for different output dim).
    """
    for param in module.parameters():
        if param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        else:
            # Bias or 1D params: zero initialization
            nn.init.zeros_(param)


def _apply_freeze_strategy(bundle, freeze_strategy: str) -> list:
    """
    Apply freezing strategy to specify which modules are frozen after loading.

    Args:
        bundle: MotionCanvasBundle instance
        freeze_strategy: 'none', 'encoders', 'text_refiner', 'blocks', 'full'

    Returns:
        List of frozen module names
    """
    frozen = []

    if freeze_strategy == 'none':
        return frozen

    # Build list of modules to freeze based on strategy
    modules_to_freeze = []

    if freeze_strategy in ('encoders', 'text_refiner', 'blocks', 'full'):
        # Freeze text-side input projections only. The timestep encoder remains
        # trainable because it is part of the flow/diffusion dynamics, not a
        # caption encoder.
        modules_to_freeze.extend([
            'motion_transformer.ctxt_encoder',
            'motion_transformer.vtxt_encoder',
        ])

    if freeze_strategy in ('text_refiner', 'blocks', 'full'):
        # Also freeze text refiner
        modules_to_freeze.append('motion_transformer.text_refiner')

    if freeze_strategy in ('blocks', 'full'):
        # Also freeze transformer blocks
        modules_to_freeze.extend([
            'motion_transformer.double_blocks',
            'motion_transformer.single_blocks',
        ])

    # Apply freezing
    for mod_path in modules_to_freeze:
        try:
            parts = mod_path.split('.')
            if len(parts) == 2:
                parent = getattr(bundle, parts[0], None)
                if parent and hasattr(parent, parts[1]):
                    mod = getattr(parent, parts[1])
                    _freeze_module(mod)
                    frozen.append(mod_path)
        except Exception as e:
            logger.warning(f"Failed to freeze {mod_path}: {e}")

    return frozen


def verify_loading(bundle, t2m_checkpoint_path: str) -> Dict[str, Any]:
    """
    Verify that T2M pretrained parameters were correctly loaded into M2M bundle.

    Compares weights before/after loading by reloading from checkpoint and
    comparing specific parameters.

    Args:
        bundle: MotionCanvasBundle instance (after loading)
        t2m_checkpoint_path: Path to T2M checkpoint for verification

    Returns:
        Dict with verification results:
            - 'reusable_params_match': bool, whether loaded params match checkpoint
            - 'num_verified_params': int, how many parameters were checked
            - 'mismatches': list of parameter names with mismatches
    """
    t2m_state = _load_checkpoint(t2m_checkpoint_path)
    reusable_state = _filter_reusable_params(t2m_state)

    model_state = bundle.motion_transformer.state_dict()

    mismatches = []
    verified_count = 0

    for key, ckpt_value in reusable_state.items():
        if key in model_state:
            model_value = model_state[key]
            if ckpt_value.shape == model_value.shape:
                # Compare values (allow small numerical differences)
                if not torch.allclose(ckpt_value, model_value, atol=1e-4):
                    mismatches.append(key)
                verified_count += 1

    return {
        'reusable_params_match': len(mismatches) == 0,
        'num_verified_params': verified_count,
        'mismatches': mismatches,
    }
