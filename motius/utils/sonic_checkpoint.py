"""Read SONIC neural tensors without importing its serialized trainer objects.

The public checkpoint pickles TRL training arguments/state and enum values
beside plain neural state dictionaries. Only those explicit non-neural classes
are discarded. Neural classes, tensors and storage deserialization are never
substituted. This reader intentionally does not import training progress.
"""

from pathlib import Path
import pickle
from types import ModuleType
import torch


_TRAINING_METADATA = {
    ('accelerate.state', 'PartialState'),
    ('accelerate.utils.dataclasses', 'DistributedType'),
    ('transformers.trainer_pt_utils', 'AcceleratorConfig'),
    ('transformers.trainer_utils', 'HubStrategy'),
    ('transformers.trainer_utils', 'IntervalStrategy'),
    ('transformers.trainer_utils', 'SaveStrategy'),
    ('transformers.trainer_utils', 'SchedulerType'),
    ('transformers.training_args', 'OptimizerNames'),
    ('trl.trainer.ppo_config', 'PPOConfig'),
    ('trl.trainer.utils', 'OnlineTrainerState'),
}


class _DiscardedTrainingMetadata:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _NeuralStateUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _TRAINING_METADATA:
            return _DiscardedTrainingMetadata
        return super().find_class(module, name)


def load_sonic_neural_state(checkpoint):
    """Return strict plain-tensor actor/critic state from the public checkpoint."""
    path = Path(checkpoint).expanduser().resolve()
    module = ModuleType('motius_sonic_neural_checkpoint')
    module.Unpickler = _NeuralStateUnpickler
    module.load = lambda file, **kwargs: _NeuralStateUnpickler(file, **kwargs).load()
    with path.open('rb') as handle:
        is_zip = handle.read(4) == b'PK\x03\x04'
    source = torch.load(
        path, map_location='cpu', weights_only=False, pickle_module=module, mmap=is_zip
    )
    roles = {}
    for name in [
        'actor_model_state_dict',
        'critic_model_state_dict',
        'policy_state_dict',
        'value_state_dict',
    ]:
        if name not in source:
            continue
        state = source[name]
        if not isinstance(state, dict) or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor)
            for key, value in state.items()
        ):
            raise ValueError(
                'SONIC neural state must contain only named tensors: ' + name
            )
        roles[name] = state
    if not roles:
        raise ValueError('No published SONIC neural state dictionaries found')
    return roles
