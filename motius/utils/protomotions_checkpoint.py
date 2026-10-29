"""Source-compatible neural checkpoint interchange, outside PPO/model code."""

from pathlib import Path
import os
import torch
from motius.models.protomotions.policy_bundle import SOURCE_NEURAL_PREFIXES


def source_neural_state(bundle):
    result = {}
    for role in bundle._save_ckpt_modules:
        module = getattr(bundle, role)
        module = getattr(module, "module", module)
        for key, value in module.state_dict().items():
            native = role + "." + key
            for original, prefix in SOURCE_NEURAL_PREFIXES.items():
                if native.startswith(prefix):
                    result[original + native[len(prefix) :]] = value.detach().cpu()
                    break
            else:
                raise ValueError("Unsupported native neural state: " + native)
    return result


def save_source_checkpoint(runner, path, *, step_count):
    collector = runner.train_dataloader.dataset.collector.collector
    state = dict(
        model=source_neural_state(runner.bundle),
        epoch=runner.global_step,
        step_count=int(step_count),
        native_format="motius-protomotions-interchange-v1",
        adv_mean_ema=collector.normalization.mean.detach().cpu(),
        adv_std_ema=collector.normalization.std.detach().cpu(),
    )
    names = dict(
        actor="actor_optimizer",
        critic="critic_optimizer",
        discriminator="discriminator_optimizer",
        discriminator_critic="disc_critic_optimizer",
    )
    state.update(
        {
            names[role]: optimizer.state_dict()
            for role, optimizer in runner.optimizers.items()
        }
    )
    path = Path(path)
    temporary = path.with_name("." + path.name + "." + str(os.getpid()) + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
