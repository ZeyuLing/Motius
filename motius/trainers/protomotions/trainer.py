"""Native PPO, L2C2 and AMP objectives for ProtoMotions' released G1 model."""

import torch
from torch import nn
from torch.distributions import Normal
from motius.trainers.base_trainer import BaseTrainer
from motius.registry import TRAINERS


@TRAINERS.register_module()
class ProtoMotionsPPOTrainer(BaseTrainer):
    trainer_controls_optimization = True
    upstream_commit = "49fe5ad69de67ebbc07ea2b25d41b0f622c15c3c"

    @classmethod
    def launch(cls, arguments=None):
        from .train import main

        main(arguments)

    @classmethod
    def export_policy(
        cls, checkpoint, output_dir=None, *, configuration=None, device="cpu"
    ):
        from pathlib import Path
        import json
        from motius.pipelines.protomotions import ProtoMotionsPipeline

        checkpoint = Path(checkpoint).expanduser().resolve()
        destination = (
            Path(output_dir)
            if output_dir is not None
            else checkpoint.parent / "exported"
        )
        if checkpoint.is_dir() or checkpoint.name == "model.pt":
            directory = checkpoint if checkpoint.is_dir() else checkpoint.parent
            metadata = json.loads(
                (directory.parent / "protomotions_bundle.json").read_text()
            )
            from motius.models.protomotions import ProtoMotionsPolicyBundle

            bundle = ProtoMotionsPolicyBundle(
                metadata["network_options"], device=device
            )
            weights = torch.load(
                directory / "model.pt", map_location="cpu", weights_only=True
            )
            state = {
                role + "." + key: value
                for role in bundle._save_ckpt_modules
                for key, value in weights[role].items()
            }
            bundle.load_state_dict(state, strict=True)
            pipeline = ProtoMotionsPipeline(
                bundle.eval(), deployment=metadata["deployment"]
            )
        else:
            pipeline = ProtoMotionsPipeline.from_original_checkpoint(
                checkpoint, configuration=configuration, device=device
            )
        return Path(pipeline.save_pretrained(destination))

    def __init__(
        self,
        bundle,
        *,
        clip=0.2,
        entropy_coef=0.005,
        bounds_loss_coef=0.0,
        clip_value_loss=True,
        l2c2_weight=1.0,
        l2c2_pairs=None,
        max_grad_norm=50.0,
        actor_clip_frac_threshold=0.6,
        discriminator_batch_size=4096,
        discriminator_grad_penalty=5.0,
        discriminator_weight_decay=0.0001,
        discriminator_logit_weight_decay=0.01,
        discriminator_optimization_ratio=1,
        check_grad_mag=True,
        fail_on_bad_grads=False,
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        self.clip = clip
        self.entropy_coef = entropy_coef
        self.bounds_loss_coef = bounds_loss_coef
        self.clip_value_loss = clip_value_loss
        self.l2c2_weight = l2c2_weight
        self.max_grad_norm = max_grad_norm
        self.l2c2_pairs = (
            l2c2_pairs
            if l2c2_pairs is not None
            else {
                "noisy_reduced_coords_obs": "clean_reduced_coords_obs",
                "noisy_mimic_reduced_coords_target_poses": "clean_mimic_reduced_coords_target_poses",
            }
        )
        self.actor_clip_frac_threshold = actor_clip_frac_threshold
        self.discriminator_batch_size = discriminator_batch_size
        self.discriminator_grad_penalty = discriminator_grad_penalty
        self.discriminator_weight_decay = discriminator_weight_decay
        self.discriminator_logit_weight_decay = discriminator_logit_weight_decay
        self.discriminator_optimization_ratio = discriminator_optimization_ratio
        self.check_grad_mag = check_grad_mag
        self.fail_on_bad_grads = fail_on_bad_grads
        self.optimization_epoch = None
        self.skip_actor_for_epoch = False

    def actor_loss(self, batch):
        result = self.bundle.predict_policy(batch)
        mean = result["mean"]
        distribution = Normal(mean, result["std"])
        distribution.sample()  # The original actor forward always draws an action.
        current = -distribution.log_prob(batch["action"]).sum(dim=-1)
        ratio = torch.exp(batch["neglogp"] - current)
        first = batch["advantages"] * ratio
        second = batch["advantages"] * torch.clamp(
            ratio, 1.0 - self.clip, 1.0 + self.clip
        )
        ppo = torch.max(-first, -second).mean()
        clipped = (torch.abs(ratio - 1.0) > self.clip).detach().float().mean()
        if self.bounds_loss_coef > 0:
            upper = torch.maximum(mean - 1.0, torch.tensor(0, device=mean.device)) ** 2
            lower = torch.minimum(mean + 1.0, torch.tensor(0, device=mean.device)) ** 2
            bounds = (lower + upper).sum(axis=-1) * self.bounds_loss_coef
        else:
            bounds = torch.zeros(batch["action"].shape[0], device=mean.device)
        bounds = bounds.mean()
        extra = torch.tensor(0.0, device=mean.device)
        logs = {}
        if self.l2c2_weight is not None:
            actor = getattr(self.bundle.actor, "module", self.bundle.actor)
            clean = {}
            input_ss = torch.tensor(0.0, device=mean.device)
            input_n = 0
            for key in actor.mean.input_keys:
                if key in self.l2c2_pairs:
                    clean[key] = batch[self.l2c2_pairs[key]]
                    diff = batch[key] - clean[key]
                    input_ss = input_ss + diff.pow(2).sum()
                    input_n += diff.numel()
                else:
                    clean[key] = batch[key]
            input_distance = (input_ss / input_n).detach()
            clean_result = self.bundle.predict_policy(clean)
            Normal(clean_result["mean"], clean_result["std"]).sample()
            output_distance = (mean - clean_result["mean"]).pow(2).mean()
            l2c2 = output_distance / (input_distance + 1e-8)
            extra = extra + self.l2c2_weight * l2c2
            logs.update(
                l2c2=l2c2, l2c2_input=input_distance, l2c2_output=output_distance
            )
        # The original actor and its MLP each contribute a zero auxiliary term.
        model_loss = ppo * 0.0 + ppo * 0.0
        loss = ppo + bounds + extra + model_loss
        actor = getattr(self.bundle.actor, "module", self.bundle.actor)
        entropy = (
            distribution.entropy().sum(dim=-1).mean()
            if actor.logstd.requires_grad
            else torch.tensor(0.0, device=mean.device)
        )
        if actor.logstd.requires_grad:
            loss = loss - self.entropy_coef * entropy
        return loss, dict(
            ppo=ppo,
            bounds=bounds,
            extra=extra,
            entropy=entropy,
            clip_fraction=clipped,
            **logs,
        )

    def value_loss(self, batch, *, discriminator=False):
        if discriminator:
            values = self.bundle.predict_discriminator_value(batch)
            old = batch["disc_value"]
            returns = batch["disc_returns"]
        else:
            values = self.bundle.predict_value(batch)
            old = batch["value"]
            returns = batch["returns"]
        if self.clip_value_loss:
            raw = (values - returns.unsqueeze(-1)).pow(2)
            clipped = old + torch.clamp(values - old, -self.clip, self.clip)
            loss = torch.max(raw, (clipped - returns.unsqueeze(-1)).pow(2)).mean()
        else:
            loss = (returns.unsqueeze(-1) - values).pow(2).mean()
        return loss if discriminator else loss + loss * 0.0

    def discriminator_loss(self, batch):
        groups = {
            prefix: {
                key.replace(prefix + "_", ""): value[: self.discriminator_batch_size]
                for key, value in batch.items()
                if prefix + "_" in key
            }
            for prefix in ["expert", "agent", "replay"]
        }
        expert = {
            name: (
                value if value.is_floating_point() else value.float()
            ).requires_grad_(True)
            for name, value in groups["expert"].items()
        }
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            expert_logits, normalized = self.bundle.predict_discriminator(
                expert, return_normalized=True
            )
        agent_logits = self.bundle.predict_discriminator(groups["agent"])
        replay_logits = self.bundle.predict_discriminator(groups["replay"])
        positive = -nn.functional.logsigmoid(expert_logits).mean()
        negative = 0.5 * (
            nn.functional.softplus(agent_logits).mean()
            + nn.functional.softplus(replay_logits).mean()
        )
        classification = 0.5 * (positive + negative)
        gradients = torch.autograd.grad(
            expert_logits,
            [normalized],
            grad_outputs=torch.ones_like(expert_logits),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
            allow_unused=True,
        )
        norms = sum(
            (
                g.flatten(start_dim=1).pow(2).sum(dim=-1)
                for g in gradients
                if g is not None
            ),
            torch.zeros(expert_logits.shape[0], device=expert_logits.device),
        )
        penalty = torch.mean(norms)
        network = getattr(
            self.bundle.discriminator, "module", self.bundle.discriminator
        )
        if self.discriminator_weight_decay > 0:
            matrices = [
                module.weight
                for module in network.modules()
                if isinstance(getattr(module, "weight", None), nn.Parameter)
            ]
            decay = (
                sum([value.pow(2).sum() for value in matrices])
                * self.discriminator_weight_decay
            )
        else:
            decay = torch.tensor(0.0, device=expert_logits.device)
        if self.discriminator_logit_weight_decay > 0:
            logit_decay = (
                sum([network.mlp[-1].weight.pow(2).sum()])
                * self.discriminator_logit_weight_decay
            )
        else:
            logit_decay = torch.tensor(0.0, device=expert_logits.device)
        loss = (
            self.discriminator_grad_penalty * penalty
            + classification
            + decay
            + logit_decay
        )
        return loss, dict(
            classification=classification,
            gradient_penalty=penalty,
            weight_decay=decay,
            logit_decay=logit_decay,
        )

    def _update(self, name, loss):
        optimizer = self.get_optimizer(name)
        optimizer.zero_grad(set_to_none=True)
        if self.accelerator is None:
            loss.backward()
        else:
            self.accelerator.backward(loss)
        parameters = list(getattr(self.bundle, name).parameters())
        total = 0.0
        for parameter in parameters:
            if parameter.grad is not None:
                total += torch.sum(parameter.grad**2)
        norm = torch.sqrt(total)
        bad = torch.isnan(norm) or (self.check_grad_mag and norm > 1000000.0)
        if bad:
            if self.fail_on_bad_grads:
                raise ValueError("Invalid or excessive gradient in " + name)
            for parameter in parameters:
                if parameter.grad is not None:
                    parameter.grad.zero_()
        if self.max_grad_norm > 0:
            if self.accelerator is None:
                nn.utils.clip_grad_norm_(
                    parameters, self.max_grad_norm, error_if_nonfinite=True
                )
            else:
                if not bad and not torch.isfinite(norm):
                    raise RuntimeError("Nonfinite gradient norm in " + name)
                self.accelerator.clip_grad_norm_(parameters, self.max_grad_norm)
        optimizer.step()

    def train_step(self, batch):
        if "rollout_batches" in batch:
            # Collection and replay augmentation happened in eval mode, just as
            # in the source. One runner iteration owns the entire PPO rollout.
            self.bundle.train()
            totals = {}
            try:
                for index, minibatch in batch["rollout_batches"]:
                    minibatch.update(
                        optimization_epoch=batch["optimization_epoch"],
                        optimization_batch_index=index,
                    )
                    metrics = self.train_step(minibatch)
                    for key, value in metrics.items():
                        if key == "loss":
                            continue
                        if key not in totals:
                            totals[key] = []
                        totals[key].append(torch.as_tensor(value).detach())
                return dict(
                    loss=None,
                    **{
                        key: torch.stack(values).mean()
                        for key, values in totals.items()
                    },
                )
            finally:
                self.bundle.eval()
        epoch = int(batch["optimization_epoch"])
        if self.optimization_epoch != epoch:
            self.optimization_epoch = epoch
            self.skip_actor_for_epoch = False
        actor, metrics = self.actor_loss(batch)
        fraction = metrics["clip_fraction"].detach()
        if (
            torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            count = torch.tensor(
                batch["action"].shape[0], device=fraction.device, dtype=torch.float32
            )
            fraction = fraction * count
            torch.distributed.all_reduce(fraction)
            torch.distributed.all_reduce(count)
            fraction = fraction / count
        if (
            self.actor_clip_frac_threshold is not None
            and fraction.item() > self.actor_clip_frac_threshold
        ):
            self.skip_actor_for_epoch = True
        if self.skip_actor_for_epoch:
            return dict(loss=None, actor_update_skipped=1.0)
        self._update("actor", actor)
        critic = self.value_loss(batch)
        self._update("critic", critic)
        result = dict(
            loss=None,
            actor_loss=actor.detach(),
            critic_loss=critic.detach(),
            actor_update_skipped=0.0,
        )
        if hasattr(self.bundle, "discriminator_critic"):
            value = self.value_loss(batch, discriminator=True)
            self._update("discriminator_critic", value)
            result["discriminator_value_loss"] = value.detach()
        if (
            hasattr(self.bundle, "discriminator")
            and int(batch["optimization_batch_index"])
            % self.discriminator_optimization_ratio
            == 0
        ):
            value, _ = self.discriminator_loss(batch)
            self._update("discriminator", value)
            result["discriminator_loss"] = value.detach()
        return result

    def val_step(self, batch):
        # AMP gradient penalties require autograd even during validation.
        with torch.enable_grad():
            actor, metrics = self.actor_loss(batch)
            return dict(
                loss=actor.detach() + self.value_loss(batch).detach(),
                **{key: value.detach() for key, value in metrics.items()},
            )
