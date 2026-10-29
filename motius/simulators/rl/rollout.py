"""On-policy vector rollouts through the shared neural bundle."""

import torch
from torch.distributions import Normal
from .advantages import generalized_advantage_estimation


class VectorPolicyRollout:
    def __init__(
        self, bundle, environment, *, gamma=0.99, lam=0.95, normalize_advantage=True
    ):
        self.bundle = bundle
        self.environment = environment
        self.gamma = gamma
        self.lam = lam
        self.normalize_advantage = bool(normalize_advantage)
        observations, extras = environment.get_observations()
        # RSL-RL 2.3.1 feeds the initial observation directly, then normalizes
        # after each environment step. Retain that published ordering.
        self.observations = observations.to(bundle.device)
        self.critic_observations = (
            extras.get('observations', {}).get('critic', observations).to(bundle.device)
        )

    def _normalize(self, observations, extras):
        device = self.bundle.device
        self.observations = self.bundle.normalize_actor(observations.to(device))
        critic = extras.get('observations', {}).get('critic', observations)
        self.critic_observations = self.bundle.normalize_critic(critic.to(device))

    @torch.inference_mode()
    def collect(self, steps):
        buffers = {
            key: []
            for key in [
                'observations',
                'critic_observations',
                'actions',
                'values',
                'actions_log_prob',
                'mu',
                'sigma',
                'rewards',
                'dones',
            ]
        }
        for _ in range(int(steps)):
            prediction = self.bundle.predict_policy(
                self.observations, self.critic_observations
            )
            distribution = Normal(
                prediction['mean'], prediction['std'], validate_args=False
            )
            actions = distribution.sample()
            probabilities = distribution.log_prob(actions).sum(dim=-1)
            observations, rewards, dones, extras = self.environment.step(
                actions.to(self.environment.device)
            )
            rewards = rewards.to(self.bundle.device).clone()
            dones = dones.to(self.bundle.device)
            if 'time_outs' in extras:
                rewards += self.gamma * torch.squeeze(
                    prediction['value']
                    * extras['time_outs'].unsqueeze(1).to(self.bundle.device),
                    1,
                )
            values = dict(
                observations=self.observations,
                critic_observations=self.critic_observations,
                actions=actions,
                values=prediction['value'],
                actions_log_prob=probabilities.unsqueeze(1),
                mu=prediction['mean'],
                std=prediction['std'],
                rewards=rewards.unsqueeze(1),
                dones=dones.unsqueeze(1),
            )
            values['sigma'] = values.pop('std')
            for name in buffers:
                buffers[name].append(values[name].detach().clone())
            self._normalize(observations, extras)
        rollout = {key: torch.stack(value) for key, value in buffers.items()}
        last_values = self.bundle.predict_value(self.critic_observations).detach()
        returns, advantages = generalized_advantage_estimation(
            rollout['rewards'],
            rollout['dones'],
            rollout['values'],
            last_values,
            gamma=self.gamma,
            lam=self.lam,
            normalize=self.normalize_advantage,
        )
        rollout.update(returns=returns, advantages=advantages)
        return {
            key: value
            for key, value in rollout.items()
            if key not in {'rewards', 'dones'}
        }
