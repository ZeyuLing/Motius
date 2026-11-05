"""Network-only native SONIC modality policy and value ModelBundle."""

from pathlib import Path
import json
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.models.onnx_policy import resolve_pretrained_directory
from motius.registry import MODEL_BUNDLES
from motius.utils.torch_checkpoint import load_torch_checkpoint
from motius.utils.sonic_checkpoint import load_sonic_neural_state
from .network import SonicPolicyNetwork, SonicValueNetwork


@MODEL_BUNDLES.register_module()
class SONICPolicyBundle(ModelBundle):
    METHOD_NAME = 'SONIC'

    def __init__(
        self, network_options, *, checkpoint=None, original=False, device='cpu'
    ):
        super().__init__()
        self.network_options = network_options
        self.artifact_root = (
            Path(checkpoint).expanduser().resolve().parent
            if checkpoint is not None
            else None
        )
        self.policy = SonicPolicyNetwork(**network_options['policy'])
        self.critic = SonicValueNetwork(**network_options['critic'])
        self._save_ckpt_modules = ['policy', 'critic']
        self._trainable_modules = ['policy', 'critic']
        if checkpoint is not None:
            if original:
                self.load_state_dict(
                    self.convert_original_state(load_sonic_neural_state(checkpoint)),
                    strict=True,
                )
            else:
                self.load_state_dict(
                    load_torch_checkpoint(checkpoint, weights_only=True), strict=True
                )
        self.to(device)

    @staticmethod
    def convert_original_state(states):
        converted = {}
        roles = [
            (
                states.get('policy_state_dict', states.get('actor_model_state_dict')),
                'policy',
            ),
            (
                states.get('value_state_dict', states.get('critic_model_state_dict')),
                'critic',
            ),
        ]
        for weights, role in roles:
            if weights is None:
                raise ValueError('Missing SONIC ' + role + ' state')
            for name, value in weights.items():
                if role == 'policy':
                    if name == 'std':
                        destination = 'policy.std'
                    elif name.startswith('actor_module.'):
                        destination = 'policy.' + name[len('actor_module.') :]
                    else:
                        raise ValueError('Unsupported original policy state: ' + name)
                elif name.startswith('critic_module.'):
                    destination = 'critic.network.' + name[len('critic_module.') :]
                elif name.startswith('running_mean_std.'):
                    destination = (
                        'critic.normalizer.' + name[len('running_mean_std.') :]
                    )
                else:
                    raise ValueError('Unsupported original value state: ' + name)
                if destination in converted:
                    raise ValueError('Duplicate neural tensor: ' + destination)
                converted[destination] = value
        return converted

    @property
    def device(self):
        return next(self.policy.parameters()).device

    def predict_policy(self, features, proprioception, encoder_masks, **kwargs):
        return self.policy(features, proprioception, encoder_masks, **kwargs)

    def predict_value(self, observations):
        return self.critic(observations)

    def encode_tokens(self, features, encoder_masks, batch, sequence=1):
        return self.policy.encode(features, encoder_masks, batch, sequence)[0]

    def decode_actions(self, tokens, proprioception):
        return self.policy.decode_actions(tokens, proprioception)

    def encode_g1_latents(self, features):
        return getattr(self.policy, 'module', self.policy).encoders['g1'](features)

    def policy_std_parameter(self):
        return getattr(self.policy, 'module', self.policy).std

    forward = predict_policy

    @classmethod
    def from_original_checkpoint(cls, checkpoint, *, configuration=None, device='cpu'):
        from motius.configuration.sonic import original_neural_options

        checkpoint = Path(checkpoint).expanduser().resolve()
        configuration = configuration or checkpoint.parent / 'config.yaml'
        states = load_sonic_neural_state(checkpoint)
        return cls(
            original_neural_options(states, configuration),
            checkpoint=checkpoint,
            original=True,
            device=device,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        hub = {
            name: kwargs.pop(name)
            for name in ['revision', 'cache_dir', 'token', 'local_files_only']
            if name in kwargs
        }
        root = resolve_pretrained_directory(pretrained_model_name_or_path, **hub)
        config = json.loads((root / 'sonic_neural.json').read_text())
        config.update(kwargs)
        return cls(**config, checkpoint=root / 'sonic_networks.pt').eval()

    def save_pretrained(self, save_directory, **kwargs):
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        state = {
            name + '.' + key: value.detach().cpu()
            for name in self._save_ckpt_modules
            for key, value in getattr(
                getattr(self, name), 'module', getattr(self, name)
            )
            .state_dict()
            .items()
        }
        torch.save(state, root / 'sonic_networks.pt')
        (root / 'sonic_neural.json').write_text(
            json.dumps(dict(network_options=self.network_options), indent=2) + '\n'
        )
        (root / 'model_index.json').write_text(
            json.dumps(
                dict(
                    model_type='sonic',
                    artifact_format='motius-sonic-neural-v2',
                    bundle_class='motius.models.sonic.SONICPolicyBundle',
                    pipeline_class='motius.pipelines.sonic.SONICPipeline',
                    tasks=['motion_tracking'],
                    required_files=['sonic_neural.json', 'sonic_networks.pt'],
                ),
                indent=2,
            )
            + '\n'
        )
        return str(root)
