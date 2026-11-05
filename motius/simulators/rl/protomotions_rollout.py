"""Original G1 PPO/AMP collection through native neural roots and env terms."""

import torch
from torch.distributions import Normal

from .protomotions_advantages import (
    ProtoMotionsAdvantageNormalization,
    task_style_advantages,
)


class ProtoMotionsRollout:
    """Preserve reset, sampling, termination and next-value ordering.

    The environment implements reset(indices) and step(actions), returning the
    original observation/reward/done/terminated contract. It owns robot state,
    reference motion and observation history; none of those live in the Bundle.
    """

    def __init__(
        self,
        bundle,
        environment,
        *,
        gamma=0.99,
        tau=0.95,
        task_weight=0.5,
        style_weight=2.0,
        advantage_normalization=None,
        reward_threshold=0.02,
        max_bad_transitions=10,
    ):
        self.bundle = bundle
        self.environment = environment
        self.gamma = gamma
        self.tau = tau
        self.task_weight = task_weight
        self.style_weight = style_weight
        self.normalization = ProtoMotionsAdvantageNormalization(
            bundle.device, **(advantage_normalization or {})
        )
        self.bad_transitions = torch.zeros(
            environment.num_envs, dtype=torch.int32, device=bundle.device
        )
        self.done_indices = torch.arange(
            environment.num_envs, device=bundle.device, dtype=torch.long
        )
        self.initialized = False
        self.reward_threshold = reward_threshold
        self.max_bad_transitions = max_bad_transitions

    def _forward(self, observations):
        prediction = self.bundle.predict_policy(observations)
        distribution = Normal(prediction['mean'], prediction['std'])
        action = distribution.sample()
        value = self.bundle.predict_value(observations)
        logits = self.bundle.predict_discriminator(observations)
        disc_value = self.bundle.predict_discriminator_value(observations)
        return dict(
            action=action,
            mean_action=prediction['mean'],
            neglogp=-distribution.log_prob(action).sum(-1),
            value=value,
            disc_logits=logits,
            disc_value=disc_value,
        )

    @torch.no_grad()
    def collect(self, steps):
        self.bundle.eval()
        if not self.initialized:
            observations, _ = self.environment.reset()
            # Original fit() makes one setup-only full neural forward to infer
            # buffer shapes. Its unused sampled action advances CUDA RNG.
            self._forward(observations)
            self.initialized = True
        buffers = {}
        for _ in range(int(steps)):
            observations, _ = self.environment.reset(self.done_indices)
            prediction = self._forward(observations)
            next_observations, rewards, dones, terminated, _ = self.environment.step(
                prediction['action']
            )
            discriminator_termination = self.bad_transitions >= self.max_bad_transitions
            terminated = terminated | discriminator_termination
            dones = dones | terminated
            self.done_indices = dones.nonzero(as_tuple=False).squeeze(-1)
            next_value = self.bundle.predict_value(next_observations) * (
                1 - terminated.float()
            ).unsqueeze(-1)
            next_logits = self.bundle.predict_discriminator(next_observations)
            probability = 1 / (1 + torch.exp(-next_logits))
            style_reward = -torch.log(torch.clamp(1 - probability, min=1e-4)).flatten()
            bad = style_reward < self.reward_threshold
            self.bad_transitions[bad] += 1
            self.bad_transitions[~bad] = 0
            if len(self.done_indices):
                self.bad_transitions[self.done_indices] = 0
            next_disc_value = self.bundle.predict_discriminator_value(next_observations)
            next_disc_value = next_disc_value * (1 - terminated.float()).unsqueeze(-1)
            current = {
                **observations,
                **prediction,
                'next_value': next_value,
                'next_disc_value': next_disc_value,
                'rewards': rewards,
                'amp_rewards': style_reward,
                'dones': dones.long(),
            }
            for name, value in current.items():
                buffers.setdefault(name, []).append(value.detach().clone())
        rollout = {name: torch.stack(values) for name, values in buffers.items()}
        rollout.update(
            task_style_advantages(
                rollout,
                self.normalization,
                gamma=self.gamma,
                tau=self.tau,
                task_weight=self.task_weight,
                style_weight=self.style_weight,
            )
        )
        return {
            name: value.transpose(0, 1).reshape(
                value.shape[0] * value.shape[1], *value.shape[2:]
            )
            for name, value in rollout.items()
        }

    def state_dict(self):
        return dict(advantage_normalization=self.normalization.state_dict())

    def load_state_dict(self, state):
        # Like original checkpoint loading, fresh physical environments and AMP
        # episode counters are initialized on process restart.
        self.normalization.load_state_dict(state['advantage_normalization'])


class ProtoMotionsOnlineData:
    def __init__(
        self,
        collector,
        replay,
        expert_provider,
        *,
        batch_size=16384,
        mini_epochs=2,
        discriminator_batch_size=4096,
        replay_keep_probability=0.01,
    ):
        self.collector = collector
        self.replay = replay
        self.expert_provider = expert_provider
        self.options = dict(
            batch_size=batch_size,
            mini_epochs=mini_epochs,
            discriminator_batch_size=discriminator_batch_size,
            replay_keep_probability=replay_keep_probability,
        )
        self.epoch = 0

    def collect(self, steps):
        from motius.datasets.protomotions_online import prepare_amp_minibatches

        rollout = self.collector.collect(steps)
        batches = prepare_amp_minibatches(
            rollout, self.replay, self.expert_provider, **self.options
        )
        result = dict(optimization_epoch=self.epoch, rollout_batches=batches)
        self.epoch += 1
        return result
