"""Native SONIC PPO epochs, optimizer updates and original statistic synchronization."""

import torch
from motius.registry import TRAINERS
from .trainer import SONICPPOTrainer
from motius.simulators.rl.sonic_rollout import unwrap


@TRAINERS.register_module()
class SONICOnlinePPOTrainer(SONICPPOTrainer):
    trainer_controls_optimization = True

    def __init__(
        self,
        bundle,
        *,
        collector,
        learning_epochs=5,
        num_mini_batches=4,
        micro_batch_size=None,
        max_grad_norm=0.1,
        shuffle_every_epoch=True,
        sync_running_mean_std_freq=1,
        sync_adaptive_sampling_freq=200,
        **kwargs,
    ):
        super().__init__(bundle, **kwargs)
        self.collector = collector
        self.learning_epochs = int(learning_epochs)
        self.num_mini_batches = int(num_mini_batches)
        self.micro_batch_size = (
            None if micro_batch_size is None else int(micro_batch_size)
        )
        self.max_grad_norm = float(max_grad_norm)
        self.shuffle_every_epoch = bool(shuffle_every_epoch)
        self.sync_running_mean_std_freq = int(sync_running_mean_std_freq)
        self.sync_adaptive_sampling_freq = int(sync_adaptive_sampling_freq)
        self.last_updates = []

    def sync_normalizers(self, iteration):
        frequency = self.sync_running_mean_std_freq
        if self.accelerator.num_processes <= 1 or not (
            iteration < 200 or (frequency > 0 and (iteration + 1) % frequency == 0)
        ):
            return
        normalizer = unwrap(self.bundle.critic).normalizer
        if not hasattr(normalizer, "running_mean"):
            return
        self.accelerator.wait_for_everyone()
        # Preserve the source's mean-of-local-mean/variance policy and local count.
        stats = torch.cat(
            [normalizer.running_mean.flatten(), normalizer.running_var.flatten()]
        )
        gathered = self.accelerator.gather(stats[None])
        dimension = normalizer.running_mean.numel()
        normalizer.running_mean.copy_(
            gathered[:, :dimension]
            .reshape(self.accelerator.num_processes, *normalizer.running_mean.shape)
            .mean(dim=0)
        )
        normalizer.running_var.copy_(
            gathered[:, dimension:]
            .reshape(self.accelerator.num_processes, *normalizer.running_var.shape)
            .mean(dim=0)
        )

    def sync_sampling(self, iteration):
        environment = self.collector.environment
        if hasattr(environment, "sync_and_compute_adaptive_sampling"):
            frequency = self.sync_adaptive_sampling_freq
            sync = frequency > 0 and (iteration + 1) % frequency == 0
            environment.sync_and_compute_adaptive_sampling(
                self.accelerator, sync_across_gpus=sync
            )

    def train_step(self, rollout):
        if self.accelerator is None or not self.optimizers:
            raise RuntimeError(
                "Online SONIC updates require the shared Runner optimizer/Accelerator binding"
            )
        if len(self.optimizers) != 1:
            raise ValueError("The original TRL release uses one joint AdamW optimizer")
        self.bundle.train()
        environment = self.collector.environment
        if hasattr(environment, "set_is_evaluating"):
            environment.set_is_evaluating(
                is_evaluating=self.collector.train_with_evaluating_env, log_info=False
            )
        optimizer = next(iter(self.optimizers.values()))
        environments = rollout["actions"].shape[0]
        if environments % self.num_mini_batches:
            raise ValueError(
                "SONIC environment count must be divisible by num_mini_batches"
            )
        mini = environments // self.num_mini_batches
        micro = self.micro_batch_size or mini
        if mini % micro:
            raise ValueError("SONIC minibatch must be divisible by micro_batch_size")
        parameters = list(self.bundle.policy.parameters()) + list(
            self.bundle.critic.parameters()
        )
        self.last_updates = []
        total = rollout["actions"].new_tensor(0.0)
        count = 0
        for epoch in range(self.learning_epochs):
            if self.shuffle_every_epoch or epoch == 0:
                order = torch.randperm(environments, device=self.bundle.device)
            for start in range(0, environments, mini):
                indices = order[start : start + mini]
                for offset in range(0, mini, micro):
                    chosen = indices[offset : offset + micro]
                    with self.accelerator.accumulate(
                        self.bundle.policy, self.bundle.critic
                    ):
                        output = super().train_step(
                            self.collector.minibatch(rollout, chosen)
                        )
                        self.accelerator.backward(output["loss"])
                        finite = all(
                            parameter.grad is None
                            or torch.isfinite(parameter.grad).all()
                            for parameter in parameters
                        )
                        norm = None
                        if finite:
                            norm = self.accelerator.clip_grad_norm_(
                                parameters, self.max_grad_norm
                            )
                            optimizer.step()
                        optimizer.zero_grad()
                    record = {
                        "loss": output["loss"].detach(),
                        "learning_rate": self.learning_rate,
                        "indices": chosen.detach().clone(),
                        "gradient_norm": (
                            None if norm is None else torch.as_tensor(norm).detach()
                        ),
                    }
                    record.update(
                        {key: value.detach() for key, value in output["ppo"].items()}
                    )
                    record["auxiliary"] = {
                        key: value.detach()
                        for key, value in output["auxiliary"].items()
                    }
                    self.last_updates.append(record)
                    total += output["loss"].detach()
                    count += 1
        iteration = self.runner.global_step if self.runner is not None else 0
        self.sync_normalizers(iteration)
        self.sync_sampling(iteration)
        return dict(
            loss=None,
            sonic_loss=total / count,
            sonic_updates=count,
            sonic_learning_rate=self.learning_rate,
        )
