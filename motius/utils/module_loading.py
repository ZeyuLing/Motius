"""Instantiate explicit network classes without changing Python import state."""
import importlib
import torch


def load_object(module_name, module_args=None, **extra_args):
    target = module_name.replace('/', '.')
    if target.startswith('flowhmr.network.'):
        target = 'motius.models.' + target
    module, _, name = target.rpartition('.')
    args = dict(module_args or {})
    args.update(extra_args)
    return getattr(importlib.import_module(module), name)(**args)


def get_module_device(module):
    try:
        parameter = next(module.parameters())
    except StopIteration as exc:
        raise ValueError('The input module should contain parameters.') from exc
    return torch.device(parameter.get_device()) if parameter.is_cuda else torch.device('cpu')


def cast_to_module_dtype(x, module, device=None):
    if not torch.is_tensor(x):
        return x
    ref = next(module.parameters(), None)
    if ref is None:
        ref = next(module.buffers(), None)
    if ref is None:
        return x.to(device=device) if device is not None else x
    return x.to(device=device if device is not None else ref.device,
                dtype=ref.dtype if x.is_floating_point() else x.dtype)
