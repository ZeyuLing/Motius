"""Native SONIC PPO and modality-consistency objectives."""

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer


def masked_mean(values, mask):
    return (values * mask).sum() / mask.sum().clamp(min=1)


@TRAINERS.register_module()
class SONICPPOTrainer(BaseTrainer):
    """Assemble policy/value forwards and the released default training losses.

    Inputs are prepared observation minibatches. Environment stepping, history,
    GAE, epoch shuffling and checkpoint ownership belong to the online runner.
    """

    def __init__(
        self,
        bundle,
        *,
        clip_range=0.2,
        clip_range_value=0.2,
        value_coefficient=1.0,
        entropy_coefficient=0.01,
        auxiliary_coefficients=None,
        auxiliary_scale=1.0,
        learning_rate=2e-5,
        desired_kl=0.01,
        adaptive_lr_min=1e-5,
        adaptive_lr_max=2e-4,
    ):
        super().__init__(bundle)
        self.clip_range = clip_range
        self.clip_range_value = clip_range_value
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        defaults = dict(
            g1_recon=0.01,
            g1_smpl_latent=1.0,
            g1_teleop_latent=1.0,
            teleop_smpl_latent=1.0,
            reencoded_smpl_g1_latent=1.0,
        )
        self.auxiliary_coefficients = (
            defaults if auxiliary_coefficients is None else dict(auxiliary_coefficients)
        )
        self.auxiliary_scale = auxiliary_scale
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.adaptive_lr_min = adaptive_lr_min
        self.adaptive_lr_max = adaptive_lr_max

    def auxiliary_losses(self, output, features, masks):
        latents = output['encoded_latents']
        reconstruction = output['reconstruction']
        losses = dict(
            g1_recon=F.mse_loss(reconstruction, features['g1'], reduction='none').mean()
        )
        for name, left, right in [
            ('g1_smpl_latent', 'g1', 'smpl'),
            ('g1_teleop_latent', 'g1', 'teleop'),
            ('teleop_smpl_latent', 'teleop', 'smpl'),
        ]:
            left_matched = latents[left][masks[right][masks[left]]]
            if name == 'g1_smpl_latent':
                right_matched = latents[right]
            else:
                right_matched = latents[right][masks[left][masks[right]]]
            losses[name] = (
                F.mse_loss(left_matched, right_matched)
                if left_matched.shape[0]
                else reconstruction.new_tensor(0.0)
            )
        matched = latents['g1'][masks['smpl'][masks['g1']]]
        if matched.shape[0]:
            reencoded = self.bundle.encode_g1_latents(
                reconstruction.reshape(-1, *reconstruction.shape[2:])[masks['smpl']]
            )
            losses['reencoded_smpl_g1_latent'] = F.mse_loss(reencoded, matched)
        else:
            losses['reencoded_smpl_g1_latent'] = reconstruction.new_tensor(0.0)
        return losses

    def ppo_loss(self, policy, value, batch):
        mean = policy['mean']
        sigma = policy['std']
        entropy = policy['entropy']
        logprobs = policy['logprobs']
        with torch.no_grad():
            kl = torch.sum(
                torch.log(sigma / batch['old_sigma'] + 1e-5)
                + (
                    torch.square(batch['old_sigma'])
                    + torch.square(batch['old_mean'] - mean)
                )
                / (2.0 * torch.square(sigma))
                - 0.5,
                axis=-1,
            )
            local_kl = kl.mean()
            kl_mean = (
                self.accelerator.gather(local_kl).mean()
                if self.accelerator is not None
                else local_kl
            )
            if self.desired_kl is not None:
                if kl_mean > self.desired_kl * 2.0:
                    self.learning_rate = max(
                        self.adaptive_lr_min, self.learning_rate / 1.5
                    )
                elif 0.0 < kl_mean < self.desired_kl / 2.0:
                    self.learning_rate = min(
                        self.adaptive_lr_max, self.learning_rate * 1.5
                    )
                for optimizer in self.optimizers.values():
                    for group in optimizer.param_groups:
                        group['lr'] = self.learning_rate
        clipped = torch.clamp(
            value,
            batch['values'] - self.clip_range_value,
            batch['values'] + self.clip_range_value,
        )
        vf1 = torch.square(value - batch['returns'])
        vf2 = torch.square(clipped - batch['returns'])
        value_loss = masked_mean(
            torch.max(vf1, vf2).mean(dim=-1), ~batch['padding_mask_value']
        )
        value_clip = masked_mean(
            (vf2 > vf1).float(), ~batch['padding_mask_value'].unsqueeze(-1)
        )
        ratio = torch.exp(logprobs - batch['old_logprobs']).unsqueeze(-1)
        pg1 = -batch['advantages'] * ratio
        pg2 = -batch['advantages'] * torch.clamp(
            ratio, 1.0 - self.clip_range, 1.0 + self.clip_range
        )
        policy_loss = masked_mean(torch.max(pg1, pg2).sum(-1), ~batch['padding_mask'])
        entropy_loss = -masked_mean(entropy, ~batch['padding_mask'])
        loss = (
            policy_loss
            + self.value_coefficient * value_loss
            + self.entropy_coefficient * entropy_loss
        )
        return dict(
            ppo_loss=loss,
            local_kl_mean=local_kl,
            pg_losses=pg1,
            pg_losses2=pg2,
            pg_loss=policy_loss,
            vf_loss=value_loss,
            entropy_loss=entropy_loss,
            ratio=ratio,
            vf_clipfrac=value_clip,
        )

    def train_step(self, batch):
        features = batch['features']
        masks = batch['encoder_masks']
        output = self.bundle.predict_policy(features, batch['proprioception'], masks)
        auxiliary = self.auxiliary_losses(output, features, masks)
        # The released source clamps the stored parameter in-place immediately
        # after the backbone forward, before constructing its Gaussian.
        std = self.bundle.policy_std_parameter()
        with torch.no_grad():
            std.data.clamp_(min=0.001, max=0.5)
        distribution = Normal(
            output['mean'], (output['mean'] * 0.0 + std).clamp(min=1e-6)
        )
        distribution.sample()  # Preserve the original unused training action RNG draw.
        # Preserve source graph construction order. Building entropy before
        # log-prob changes accumulation order for the shared std parameter.
        logprobs = distribution.log_prob(batch['actions']).sum(-1)
        policy = dict(
            logprobs=logprobs,
            mean=distribution.mean,
            std=distribution.stddev,
            entropy=distribution.entropy().sum(-1),
        )
        value = self.bundle.predict_value(batch['critic_observations'])
        losses = self.ppo_loss(policy, value, batch)
        total = value.new_tensor(0.0)
        for name, loss in auxiliary.items():
            total += self.auxiliary_coefficients.get(name, 0.0) * loss
        loss = losses['ppo_loss'] + total * self.auxiliary_scale
        return dict(
            loss=loss,
            ppo=losses,
            auxiliary=auxiliary,
            auxiliary_loss=total * self.auxiliary_scale,
            policy=policy,
            value=value,
        )

    def val_step(self, batch):
        with torch.no_grad():
            return self.bundle.predict_policy(
                batch['features'], batch['proprioception'], batch['encoder_masks']
            )
