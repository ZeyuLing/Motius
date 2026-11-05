"""SONIC online environment interaction, bootstrap, GAE and PPO batch preparation."""

import torch
from torch.distributions import Normal
from motius.motion.conditioning.sonic_observations import SonicObservationProcessor


def unwrap(network):
    while hasattr(network, "module"):
        network = network.module
    return network


class SonicRollout:
    def __init__(
        self,
        bundle,
        environment,
        schema,
        *,
        gamma=0.99,
        lam=0.95,
        sync_advantage_normalization=True,
        use_padding_mask=False,
        value_chunk_size=1024,
        accelerator=None,
        train_with_evaluating_env=False,
        output_original_obs_dict=False,
    ):
        self.bundle, self.environment = bundle, environment
        self.processor = SonicObservationProcessor(schema)
        self.gamma, self.lam = float(gamma), float(lam)
        self.sync_advantage_normalization = bool(sync_advantage_normalization)
        self.use_padding_mask = bool(use_padding_mask)
        self.value_chunk_size = int(value_chunk_size)
        self.accelerator = accelerator
        self.train_with_evaluating_env = bool(train_with_evaluating_env)
        self.output_original_obs_dict = bool(output_original_obs_dict)
        self.observations = None
        self.last_trace = None

    def returns(self, values, last_values, rewards, dones):
        advantage = 0
        returns = torch.zeros_like(values)
        for step in reversed(range(len(values))):
            next_values = last_values if step == len(values) - 1 else values[step + 1]
            terminal = 1.0 - dones[step].float()
            delta = rewards[step] + terminal * self.gamma * next_values - values[step]
            advantage = delta + terminal * self.gamma * self.lam * advantage
            returns[step] = advantage + values[step]
        advantages = returns - values
        if self.sync_advantage_normalization and self.accelerator is not None:
            advantages = self.accelerator.gather(advantages)
            advantages = (advantages - advantages.mean(dim=(0, 1), keepdim=True)) / (
                advantages.std(dim=(0, 1), keepdim=True) + 1e-8
            )
            advantages = advantages.reshape(
                self.accelerator.num_processes, -1, *advantages.shape[1:]
            )[self.accelerator.process_index].to(values.device)
        else:
            advantages = (advantages - advantages.mean(dim=(0, 1), keepdim=True)) / (
                advantages.std(dim=(0, 1), keepdim=True) + 1e-8
            )
        return returns, advantages

    @torch.no_grad()
    def collect(self, steps):
        self.bundle.eval()
        if hasattr(self.environment, "set_is_evaluating"):
            self.environment.set_is_evaluating(
                is_evaluating=self.train_with_evaluating_env, log_info=False
            )
        device = self.bundle.device
        if self.observations is None:
            self.observations = self.environment.reset_all()
        observations = {
            key: value.to(device) for key, value in self.observations.items()
        }
        columns = {key: [] for key in observations}
        states = {
            key: []
            for key in [
                "actions",
                "actions_log_prob",
                "action_mean",
                "action_sigma",
                "rewards",
                "dones",
                "time_outs",
            ]
        }
        policy, critic = unwrap(self.bundle.policy), unwrap(self.bundle.critic)
        for _ in range(int(steps)):
            prepared = self.processor(observations)
            prediction = policy(
                prepared["features"],
                prepared["proprioception"],
                prepared["encoder_masks"],
                decode_reconstruction=False,
            )
            mean = prediction["mean"][:, -1]
            self.bundle.policy_std_parameter().data.clamp_(min=0.001, max=0.5)
            distribution = Normal(
                mean, (mean * 0.0 + self.bundle.policy_std_parameter()).clamp(min=1e-6)
            )
            actions = distribution.sample()
            state = dict(
                actions=actions,
                actions_log_prob=distribution.log_prob(actions).sum(-1).unsqueeze(1),
                action_mean=distribution.mean,
                action_sigma=distribution.stddev,
                obs_dict=(
                    {key: value[:, None] for key, value in observations.items()}
                    if self.output_original_obs_dict
                    else None
                ),
            )
            for key, value in observations.items():
                columns[key].append(value.detach().clone())
            for key in ["actions", "actions_log_prob", "action_mean", "action_sigma"]:
                states[key].append(state[key].detach().clone())
            observations, rewards, dones, infos = self.environment.step(state)
            observations = {
                key: value.to(device) for key, value in observations.items()
            }
            states["rewards"].append(
                rewards.to(device=device, dtype=torch.float32).reshape(-1, 1).clone()
            )
            states["dones"].append(dones.to(device).reshape(-1, 1).bool().clone())
            states["time_outs"].append(
                infos["time_outs"].to(device).reshape(-1, 1).bool().clone()
            )
        self.observations = observations
        stored = {key: torch.stack(value) for key, value in states.items()}
        inputs = {key: torch.stack(value) for key, value in columns.items()}
        all_observations = {
            key: torch.cat([value, observations[key].unsqueeze(0)], dim=0).transpose(
                0, 1
            )
            for key, value in inputs.items()
        }
        values = []
        environments = next(iter(all_observations.values())).shape[0]
        for start in range(0, environments, self.value_chunk_size):
            values.append(
                critic(
                    all_observations["critic_obs"][
                        start : start + self.value_chunk_size
                    ]
                )
            )
        values_all = torch.cat(values, dim=0).transpose(0, 1)
        values, last = values_all[:-1], values_all[-1]
        rewards = stored["rewards"] + self.gamma * stored["time_outs"] * values
        returns, advantages = self.returns(values, last, rewards, stored["dones"])
        dones = stored["dones"].squeeze(-1).transpose(0, 1)
        logprobs = stored["actions_log_prob"].transpose(0, 1).squeeze(-1)
        old_values = values.transpose(0, 1)
        if self.use_padding_mask:
            padding = dones.clone()
            padding_value = padding.clone()
            for index in range(len(padding)):
                locations = torch.where(padding[index])[0]
                if len(locations):
                    padding[index, locations[0]] = False
                    padding_value[
                        index, locations[0] : min(locations[0] + 2, padding.shape[1])
                    ] = False
            logprobs = torch.masked_fill(logprobs, padding, 1.0)
            old_values = torch.masked_fill(old_values, padding_value.unsqueeze(-1), 0.0)
        else:
            padding = torch.zeros_like(dones)
            padding_value = torch.zeros_like(dones)
        result = dict(
            observations={key: value.transpose(0, 1) for key, value in inputs.items()},
            actions=stored["actions"].transpose(0, 1),
            old_logprobs=logprobs,
            old_mean=stored["action_mean"].transpose(0, 1),
            old_sigma=stored["action_sigma"].transpose(0, 1),
            values=old_values,
            returns=returns.transpose(0, 1),
            advantages=advantages.transpose(0, 1),
            rewards=rewards.transpose(0, 1),
            dones=dones,
            padding_mask=padding,
            padding_mask_value=padding_value,
        )
        self.last_trace = dict(
            storage=stored,
            values=values,
            returns=returns,
            advantages=advantages,
            next_observations=observations,
        )
        return result

    def minibatch(self, rollout, indices):
        prepared = self.processor(
            {key: value[indices] for key, value in rollout["observations"].items()}
        )
        prepared.update(
            {
                key: value[indices]
                for key, value in rollout.items()
                if key not in {"observations", "rewards", "dones"}
            }
        )
        return prepared
