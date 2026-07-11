"""
Registry system for hftrainer.

Registries:
  HF_MODELS    — HuggingFace model classes (from_pretrained / from_config / from_single_file)
  MODEL_BUNDLES — ModelBundle subclasses
  TRAINERS      — Trainer subclasses
  PIPELINES     — Pipeline subclasses
  DATASETS      — Dataset subclasses
  TRANSFORMS    — Transform classes
  HOOKS         — Hook classes
  EVALUATORS    — Evaluator classes
  VISUALIZERS   — Visualizer classes
"""

import copy
from mmengine.registry import Registry, TRANSFORMS as MMENGINE_TRANSFORMS

# Map string shorthand to torch.dtype
_DTYPE_MAP = {
    'fp32': 'torch.float32',
    'fp16': 'torch.float16',
    'bf16': 'torch.bfloat16',
    'float32': 'torch.float32',
    'float16': 'torch.float16',
    'bfloat16': 'torch.bfloat16',
}


def _resolve_dtype(kwargs: dict) -> dict:
    """Convert 'torch_dtype' string shortcuts to actual torch.dtype objects."""
    if 'torch_dtype' in kwargs:
        val = kwargs['torch_dtype']
        if isinstance(val, str):
            import torch
            resolved = _DTYPE_MAP.get(val, val)
            # Support 'torch.bfloat16' style strings
            if isinstance(resolved, str) and resolved.startswith('torch.'):
                attr = resolved.split('.', 1)[1]
                kwargs['torch_dtype'] = getattr(torch, attr)
            else:
                kwargs['torch_dtype'] = resolved
    # Also handle 'dtype' (used by some transformers classes)
    if 'dtype' in kwargs:
        val = kwargs['dtype']
        if isinstance(val, str):
            import torch
            resolved = _DTYPE_MAP.get(val, val)
            if isinstance(resolved, str) and resolved.startswith('torch.'):
                attr = resolved.split('.', 1)[1]
                kwargs['dtype'] = getattr(torch, attr)
            else:
                kwargs['dtype'] = resolved
    return kwargs


def build_hf_model_from_cfg(cfg, registry):
    """
    Build a HuggingFace model (or any class) from config dict.

    Handles three loading patterns:
      - from_pretrained: cls.from_pretrained(**from_pretrained_kwargs)
      - from_config:     cls.from_config(**from_config_kwargs)
      - from_single_file: cls.from_single_file(**from_single_file_kwargs)
      - fallback:        cls(**remaining_kwargs)

    The 'type' key is used to look up the class in the registry.
    Supports 'torch_dtype' as string shorthand: 'fp32', 'fp16', 'bf16'.
    """
    cfg = copy.deepcopy(cfg)
    obj_type = cfg.pop('type')

    # Get the class from registry
    if isinstance(obj_type, str):
        cls = registry.get(obj_type)
        if cls is None:
            # Try importing from transformers/diffusers by name
            cls = _import_hf_class(obj_type)
        if cls is None:
            raise KeyError(f"Class '{obj_type}' not found in registry '{registry.name}' "
                           f"and could not be imported from transformers/diffusers.")
    else:
        cls = obj_type

    # Check for special loading patterns
    if 'from_pretrained' in cfg:
        kwargs = _resolve_dtype(cfg.pop('from_pretrained'))
        return cls.from_pretrained(**kwargs)
    elif 'from_config' in cfg:
        kwargs = _resolve_dtype(cfg.pop('from_config'))
        return cls.from_config(**kwargs)
    elif 'from_single_file' in cfg:
        kwargs = _resolve_dtype(cfg.pop('from_single_file'))
        return cls.from_single_file(**kwargs)
    else:
        return cls(**cfg)


def _import_hf_class(class_name: str):
    """Try to import a class by name from transformers, diffusers, or torch.optim."""
    import_modules = [
        'transformers',
        'diffusers',
        'diffusers.schedulers',
        'peft',
        'torch.optim',
        'torch.optim.lr_scheduler',
    ]
    for module_name in import_modules:
        try:
            import importlib
            module = importlib.import_module(module_name)
            if hasattr(module, class_name):
                return getattr(module, class_name)
        except ImportError:
            continue
    return None


# Core registries
HF_MODELS = Registry('hf_model', build_func=build_hf_model_from_cfg)
# Compatibility alias for vendored model code that expects a generic model
# registry. HF_MODELS can already build plain nn.Module classes, not only
# HuggingFace-native ones, so reusing it keeps the migration surface small.
MODELS = HF_MODELS
MODEL_BUNDLES = Registry('model_bundle')
TRAINERS = Registry('trainer')
PIPELINES = Registry('pipeline')
DATASETS = Registry('dataset')
TRANSFORMS = MMENGINE_TRANSFORMS
HOOKS = Registry('hook')
EVALUATORS = Registry('evaluator')
VISUALIZERS = Registry('visualizer')
