"""PEFT (LoRA / QLoRA) utilities for ModelBundle."""

from typing import Dict, Any, Optional

import torch.nn as nn


def _unwrap_module(module: nn.Module) -> nn.Module:
    while hasattr(module, 'module') and isinstance(module.module, nn.Module):
        module = module.module
    return module


def is_peft_model(module: nn.Module) -> bool:
    try:
        from peft import PeftModel
    except ImportError:
        return False
    module = _unwrap_module(module)
    return isinstance(module, PeftModel)


def apply_lora(module: nn.Module, lora_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Apply LoRA to a module using peft.

    Args:
        module: The nn.Module to apply LoRA to.
        lora_cfg: Configuration dict for LoraConfig. Supports:
            - r: int (LoRA rank, default 16)
            - lora_alpha: float (default 32)
            - target_modules: list[str] | str (default 'all-linear')
            - lora_dropout: float (default 0.1)
            - bias: str (default 'none')
            - task_type: str (optional, e.g. 'CAUSAL_LM')

    Returns:
        The module wrapped with LoRA (peft.PeftModel).
    """
    try:
        from peft import LoraConfig, get_peft_model, TaskType
    except ImportError:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    if lora_cfg is None:
        lora_cfg = {}
    else:
        lora_cfg = dict(lora_cfg)

    # Map task_type string to peft TaskType enum
    task_type_str = lora_cfg.pop('task_type', None)
    task_type = None
    if task_type_str is not None:
        task_type = getattr(TaskType, task_type_str, None)

    config = LoraConfig(
        r=lora_cfg.get('r', 16),
        lora_alpha=lora_cfg.get('lora_alpha', 32),
        target_modules=lora_cfg.get('target_modules', 'all-linear'),
        lora_dropout=lora_cfg.get('lora_dropout', 0.1),
        bias=lora_cfg.get('bias', 'none'),
        task_type=task_type,
    )

    return get_peft_model(module, config)


def get_lora_state_dict(
    module: nn.Module,
    state_dict: Optional[Dict[str, Any]] = None,
    adapter_name: str = 'default',
) -> Dict[str, Any]:
    try:
        from peft import get_peft_model_state_dict
    except ImportError:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    module = _unwrap_module(module)
    return get_peft_model_state_dict(
        module,
        state_dict=state_dict,
        adapter_name=adapter_name,
    )


def set_lora_state_dict(
    module: nn.Module,
    state_dict: Dict[str, Any],
    adapter_name: str = 'default',
):
    try:
        from peft import set_peft_model_state_dict
    except ImportError:
        raise ImportError("peft is required for LoRA. Install with: pip install peft")

    module = _unwrap_module(module)
    return set_peft_model_state_dict(
        module,
        peft_model_state_dict=state_dict,
        adapter_name=adapter_name,
    )


def looks_like_lora_state_dict(state_dict: Dict[str, Any]) -> bool:
    if not state_dict:
        return False
    return any(
        'lora_' in key or 'modules_to_save' in key
        for key in state_dict.keys()
    )


def merge_lora(module: nn.Module) -> nn.Module:
    module = _unwrap_module(module)
    if not hasattr(module, 'merge_and_unload'):
        raise TypeError(
            f"Module {type(module).__name__} does not expose merge_and_unload()."
        )
    return module.merge_and_unload()


def apply_qlora(module: nn.Module, lora_cfg: Optional[Dict[str, Any]] = None) -> nn.Module:
    """
    Apply QLoRA (4-bit quantization + LoRA) to a module.
    Requires bitsandbytes to be installed.
    """
    try:
        import bitsandbytes as bnb
        from peft import prepare_model_for_kbit_training
    except ImportError:
        raise ImportError(
            "bitsandbytes and peft are required for QLoRA. "
            "Install with: pip install bitsandbytes peft"
        )

    module = prepare_model_for_kbit_training(module)
    return apply_lora(module, lora_cfg)
