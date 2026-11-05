# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""BeyondMimic's native PPO update, shared neural bundle and runner optimizers."""
import torch
from torch import nn
from torch.distributions import Normal
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.simulators.rl.advantages import rollout_minibatches


@TRAINERS.register_module()
class BeyondMimicTrainer(BaseTrainer):
    trainer_controls_optimization = True
    upstream_commit = 'cd65172032893724b445448818c34165846d847d'

    @classmethod
    def launch(cls, arguments=None):
        from .train import main

        return main(arguments)

    @classmethod
    def prepare_motion(cls, arguments=None):
        from .prepare_motion import main

        return main(arguments)

    @classmethod
    def export_policy(cls, checkpoint, output_dir, **kwargs):
        from .export_policy import export_policy

        return export_policy(checkpoint, output_dir, **kwargs)

    def __init__(
        self,
        bundle,
        *,
        num_learning_epochs=5,
        num_mini_batches=4,
        clip_param=0.2,
        value_loss_coef=1.0,
        entropy_coef=0.005,
        learning_rate=1e-3,
        max_grad_norm=1.0,
        use_clipped_value_loss=True,
        schedule='adaptive',
        desired_kl=0.01,
        normalize_advantage_per_mini_batch=False,
        optimizer_name='policy',
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        if not hasattr(bundle, 'predict_policy'):
            raise TypeError(
                'PPO requires BeyondMimicPolicyBundle with trainable neural roots'
            )
        self.num_learning_epochs = int(num_learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.clip_param = float(clip_param)
        self.value_loss_coef = float(value_loss_coef)
        self.entropy_coef = float(entropy_coef)
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = float(max_grad_norm)
        self.use_clipped_value_loss = bool(use_clipped_value_loss)
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.normalize_advantage_per_mini_batch = bool(
            normalize_advantage_per_mini_batch
        )
        self.optimizer_name = optimizer_name

    def compute_loss(self, batch):
        output = self.bundle.predict_policy(
            batch['observations'], batch['critic_observations']
        )
        distribution = Normal(output['mean'], output['std'], validate_args=False)
        # The original update samples an unused action before reevaluating the
        # stored action. Preserve that RNG draw for subsequent online rollouts.
        distribution.sample()
        log_probability = distribution.log_prob(batch['actions']).sum(dim=-1)
        entropy = distribution.entropy().sum(dim=-1)
        ratio = torch.exp(log_probability - torch.squeeze(batch['actions_log_prob']))
        surrogate = -torch.squeeze(batch['advantages']) * ratio
        clipped = -torch.squeeze(batch['advantages']) * torch.clamp(
            ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
        )
        surrogate_loss = torch.max(surrogate, clipped).mean()
        values = output['value']
        if self.use_clipped_value_loss:
            value_clipped = batch['values'] + (values - batch['values']).clamp(
                -self.clip_param, self.clip_param
            )
            value_losses = (values - batch['returns']).pow(2)
            value_losses_clipped = (value_clipped - batch['returns']).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()
        else:
            value_loss = (batch['returns'] - values).pow(2).mean()
        loss = (
            surrogate_loss
            + self.value_loss_coef * value_loss
            - self.entropy_coef * entropy.mean()
        )
        return dict(
            loss=loss,
            surrogate=surrogate_loss,
            value=value_loss,
            entropy=entropy.mean(),
            mean=output['mean'],
            std=output['std'],
        )

    def adapt_learning_rate(self, result, batch, optimizer):
        if self.desired_kl is None or self.schedule != 'adaptive':
            return
        with torch.inference_mode():
            mean, std = result['mean'], result['std']
            kl = torch.sum(
                torch.log(std / batch['sigma'] + 1e-5)
                + (torch.square(batch['sigma']) + torch.square(batch['mu'] - mean))
                / (2.0 * torch.square(std))
                - 0.5,
                axis=-1,
            )
            kl_mean = torch.mean(kl)
            distributed = (
                torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
            if distributed:
                torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                kl_mean /= torch.distributed.get_world_size()
            if not distributed or torch.distributed.get_rank() == 0:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                    self.learning_rate = min(1e-2, self.learning_rate * 1.5)
            if distributed:
                rate = torch.tensor(self.learning_rate, device=self.bundle.device)
                torch.distributed.broadcast(rate, src=0)
                self.learning_rate = rate.item()
            for group in optimizer.param_groups:
                group['lr'] = self.learning_rate

    def train_step(self, rollout):
        optimizer = self.get_optimizer(self.optimizer_name)
        total = dict(value=0.0, surrogate=0.0, entropy=0.0)
        for batch in rollout_minibatches(
            rollout, self.num_mini_batches, self.num_learning_epochs
        ):
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch['advantages'] = (
                        batch['advantages'] - batch['advantages'].mean()
                    ) / (batch['advantages'].std() + 1e-8)
            result = self.compute_loss(batch)
            self.adapt_learning_rate(result, batch, optimizer)
            optimizer.zero_grad()
            if self.accelerator is None:
                result['loss'].backward()
                nn.utils.clip_grad_norm_(
                    self.bundle.trainable_parameters(), self.max_grad_norm
                )
            else:
                self.accelerator.backward(result['loss'])
                self.accelerator.clip_grad_norm_(
                    self.bundle.trainable_parameters(), self.max_grad_norm
                )
            optimizer.step()
            for key in total:
                total[key] += result[key].item()
        count = self.num_learning_epochs * self.num_mini_batches
        return dict(
            loss=None,
            **{'ppo_' + key: value / count for key, value in total.items()},
            learning_rate=self.learning_rate,
        )

    @torch.no_grad()
    def val_step(self, batch):
        result = self.compute_loss(batch)
        return {
            'val_' + key: value.detach()
            for key, value in result.items()
            if key not in {'mean', 'std'}
        }
