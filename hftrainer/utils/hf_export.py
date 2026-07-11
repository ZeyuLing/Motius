"""Helpers for exporting HuggingFace-native models more robustly."""

from contextlib import contextmanager

import torch.nn as nn


def _unwrap_module(module: nn.Module) -> nn.Module:
    while hasattr(module, 'module') and isinstance(module.module, nn.Module):
        module = module.module
    return module


def _is_broken_deepspeed_import(exc: ImportError) -> bool:
    text = str(exc)
    return (
        'deepspeed' in text
        or 'torch.distributed.elastic.agent.server.api' in text
    )


@contextmanager
def safe_hf_export():
    """
    Patch HuggingFace unwrapping helpers so export still works when the current
    environment has a broken DeepSpeed installation.

    In our common case the model is not wrapped in DeepSpeed/DDP at export
    time, so falling back to a simple ``.module`` unwrap is correct.
    """
    patches = []

    def _patch(module, attr_name: str):
        original = getattr(module, attr_name, None)
        if original is None:
            return

        def wrapped(model, *args, **kwargs):
            try:
                return original(model, *args, **kwargs)
            except ImportError as exc:
                if not _is_broken_deepspeed_import(exc):
                    raise
                return _unwrap_module(model)

        setattr(module, attr_name, wrapped)
        patches.append((module, attr_name, original))

    try:
        from accelerate.utils import other as accelerate_other
        _patch(accelerate_other, 'extract_model_from_parallel')
    except ImportError:
        pass

    try:
        import transformers.modeling_utils as modeling_utils
        _patch(modeling_utils, 'extract_model_from_parallel')
    except ImportError:
        pass

    try:
        yield
    finally:
        for module, attr_name, original in reversed(patches):
            setattr(module, attr_name, original)
