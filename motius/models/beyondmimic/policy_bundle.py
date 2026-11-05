"""Trainable BeyondMimic neural roots and source checkpoint import/export."""

from pathlib import Path
import json
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.models.onnx_policy import resolve_pretrained_directory
from motius.registry import MODEL_BUNDLES
from motius.utils.torch_checkpoint import load_torch_checkpoint
from .actor_critic import BeyondMimicActorCritic
from .normalizer import EmpiricalNormalization


@MODEL_BUNDLES.register_module()
class BeyondMimicPolicyBundle(ModelBundle):
    METHOD_NAME = 'BeyondMimic'

    def __init__(
        self,
        network_options,
        *,
        checkpoint=None,
        empirical_normalization=True,
        device='cpu',
    ):
        super().__init__()
        self.artifact_root = (
            None
            if checkpoint is None
            else Path(checkpoint).expanduser().resolve().parent
        )
        self.network_options = dict(network_options)
        self.empirical_normalization = bool(empirical_normalization)
        self.network = BeyondMimicActorCritic(**self.network_options)
        options = self.network_options
        self.actor_normalizer = (
            EmpiricalNormalization([options['num_actor_obs']], until=1e8)
            if empirical_normalization
            else torch.nn.Identity()
        )
        self.critic_normalizer = (
            EmpiricalNormalization([options['num_critic_obs']], until=1e8)
            if empirical_normalization
            else torch.nn.Identity()
        )
        self._save_ckpt_modules = ['network', 'actor_normalizer', 'critic_normalizer']
        self._trainable_modules = ['network']
        if checkpoint is not None:
            state = load_torch_checkpoint(checkpoint, weights_only=False)
            if 'model_state_dict' in state:
                self.network.load_state_dict(state['model_state_dict'], strict=True)
                if empirical_normalization:
                    self.actor_normalizer.load_state_dict(
                        state['obs_norm_state_dict'], strict=True
                    )
                    key = (
                        'privileged_obs_norm_state_dict'
                        if 'privileged_obs_norm_state_dict' in state
                        else 'critic_obs_norm_state_dict'
                    )
                    self.critic_normalizer.load_state_dict(state[key], strict=True)
            else:
                self.load_state_dict(state, strict=True)
        self.to(device)

    @property
    def device(self):
        return next(self.network.parameters()).device

    def normalize_actor(self, observations):
        return self.actor_normalizer(observations)

    def normalize_critic(self, observations):
        return self.critic_normalizer(observations)

    def predict_policy(self, observations, critic_observations=None):
        return self.network(observations, critic_observations)

    def predict_value(self, critic_observations):
        return self.network(None, critic_observations)['value']

    forward = predict_policy

    @classmethod
    def from_original_checkpoint(cls, checkpoint, *, activation='elu', device='cpu'):
        state = load_torch_checkpoint(checkpoint, weights_only=False)
        weights = state['model_state_dict']
        layers = {}
        for name in ['actor', 'critic']:
            keys = sorted(
                (
                    key
                    for key in weights
                    if key.startswith(name + '.') and key.endswith('.weight')
                ),
                key=lambda key: int(key.split('.')[1]),
            )
            if not keys:
                raise ValueError('Missing ' + name + ' linear layers')
            layers[name] = [weights[key] for key in keys]
        options = dict(
            num_actor_obs=layers['actor'][0].shape[1],
            num_critic_obs=layers['critic'][0].shape[1],
            num_actions=layers['actor'][-1].shape[0],
            actor_hidden_dims=[value.shape[0] for value in layers['actor'][:-1]],
            critic_hidden_dims=[value.shape[0] for value in layers['critic'][:-1]],
            activation=activation,
            noise_std_type='log' if 'log_std' in weights else 'scalar',
        )
        return cls(
            options,
            checkpoint=checkpoint,
            empirical_normalization='obs_norm_state_dict' in state,
            device=device,
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        options = {
            name: kwargs.pop(name)
            for name in ['revision', 'cache_dir', 'token', 'local_files_only']
            if name in kwargs
        }
        root = resolve_pretrained_directory(pretrained_model_name_or_path, **options)
        config = json.loads((root / 'beyondmimic_neural.json').read_text())
        config.update(kwargs)
        return cls(**config, checkpoint=root / 'beyondmimic_networks.pt').eval()

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
        torch.save(state, root / 'beyondmimic_networks.pt')
        (root / 'beyondmimic_neural.json').write_text(
            json.dumps(
                dict(
                    network_options=self.network_options,
                    empirical_normalization=self.empirical_normalization,
                ),
                indent=2,
            )
            + '\n'
        )
        (root / 'model_index.json').write_text(
            json.dumps(
                dict(
                    model_type='beyondmimic',
                    artifact_format='motius-beyondmimic-neural-v2',
                    bundle_class='motius.models.beyondmimic.BeyondMimicPolicyBundle',
                    pipeline_class='motius.pipelines.beyondmimic.BeyondMimicPipeline',
                    tasks=['motion_tracking'],
                    required_files=[
                        'beyondmimic_neural.json',
                        'beyondmimic_networks.pt',
                    ],
                ),
                indent=2,
            )
            + '\n'
        )
        return str(root)
