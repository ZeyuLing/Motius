"""Read original typed environment data using native configuration classes.

The public file pickles environment dataclasses and compute functions alongside
agent configuration metadata. Map environment definitions to their native
implementations; retain agent options as plain data without loading an agent.
No neural class, tensor or storage implementation is substituted.
"""

import importlib
import json
from pathlib import Path
import pickle
from types import ModuleType

import torch


_SCHEMA = json.loads((Path(__file__).parent / 'environment_modules.json').read_text())
_MODULES = _SCHEMA['modules']
_TRAINING_METADATA = {tuple(value) for value in _SCHEMA['training_metadata']}


class _ConfigurationData:
    def __setstate__(self, state):
        self.__dict__.update(state)


class _EnvironmentUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _TRAINING_METADATA:
            return _ConfigurationData
        if module in _MODULES:
            return getattr(importlib.import_module(_MODULES[module]), name)
        if module.startswith('protomotions.'):
            raise ValueError(
                'Unsupported original environment definition: ' + module + '.' + name
            )
        return super().find_class(module, name)


def load_environment_configuration(path, *, map_location='cpu'):
    path = Path(path).expanduser().resolve()
    reader = ModuleType('motius_protomotions_environment_data')
    reader.Unpickler = _EnvironmentUnpickler
    reader.load = lambda file, **kwargs: _EnvironmentUnpickler(file, **kwargs).load()
    with path.open('rb') as stream:
        is_zip = stream.read(4) == b'PK\x03\x04'
    return torch.load(
        path, map_location=map_location, weights_only=False, pickle_module=reader, mmap=is_zip
    )


def resolve_environment_target(path):
    module, name = path.rsplit('.', 1)
    if module in _MODULES:
        module = _MODULES[module]
    if not module.startswith('motius.'):
        raise ValueError(
            'Environment targets must resolve to native Motius implementations: ' + path
        )
    return getattr(importlib.import_module(module), name)
