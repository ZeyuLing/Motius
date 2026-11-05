"""PPO/AMP composition in the shared AccelerateRunner."""

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator

from motius.models.protomotions import ProtoMotionsPolicyBundle
from motius.trainers.protomotions import ProtoMotionsPPOTrainer
from motius.datasets.online_tracking import OnlineTrackingRollouts
from motius.datasets.protomotions_online import AMPReplay
from motius.simulators.rl.protomotions_rollout import (
    ProtoMotionsRollout,
    ProtoMotionsOnlineData,
)
from .accelerate_runner import AccelerateRunner, _BundleOrphanCheckpoint


def build_protomotions_runner(
    environment,
    work_dir,
    *,
    checkpoint,
    configuration=None,
    expert_provider,
    options=None,
    accelerator=None,
    max_iterations=None,
    auto_resume=False,
    resume_original=False,
):
    """Each iteration collects fresh data and performs the complete PPO epoch.

    Expert motion observations are supplied by the environment's motion/data
    adapter. They remain outside the neural Bundle and the PPO objective.
    """
    from motius.configuration.protomotions import original_training_options
    from motius.hooks.checkpoint_hook import CheckpointHook
    from motius.hooks.logger_hook import LoggerHook

    configuration = configuration or Path(checkpoint).parent / "resolved_configs.yaml"
    options = options or original_training_options(configuration)
    accelerator = accelerator or Accelerator(mixed_precision="no")
    bundle = ProtoMotionsPolicyBundle.from_original_checkpoint(
        checkpoint, configuration=configuration, device=accelerator.device
    )
    trainer = ProtoMotionsPPOTrainer(bundle, **options["trainer"])
    optimizers = {}
    for role, parameters in options["optimizers"].items():
        module = getattr(bundle, role)
        optimizer = torch.optim.Adam(module.parameters(), **parameters)
        module, optimizer = accelerator.prepare(module, optimizer)
        setattr(bundle, role, module)
        optimizers[role] = optimizer
    accelerator.register_for_checkpointing(_BundleOrphanCheckpoint(bundle))
    original_state = None
    if resume_original:
        original_state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        names = dict(
            actor="actor_optimizer",
            critic="critic_optimizer",
            discriminator="discriminator_optimizer",
            discriminator_critic="disc_critic_optimizer",
        )
        for role, optimizer in optimizers.items():
            optimizer.load_state_dict(original_state[names[role]])
    collector = ProtoMotionsRollout(bundle, environment, **options["rollout"])
    if original_state is not None:
        if "adv_mean_ema" in original_state:
            collector.normalization.mean.copy_(original_state["adv_mean_ema"])
        if "adv_std_ema" in original_state:
            collector.normalization.std.copy_(original_state["adv_std_ema"])
    accelerator.register_for_checkpointing(collector)
    replay = AMPReplay(options["replay"]["capacity"], accelerator.device)
    data = ProtoMotionsOnlineData(
        collector,
        replay,
        expert_provider,
        batch_size=options["batch_size"],
        mini_epochs=options["mini_epochs"],
        discriminator_batch_size=options["trainer"]["discriminator_batch_size"],
        replay_keep_probability=options["replay"]["keep_probability"],
    )
    loader = DataLoader(
        OnlineTrackingRollouts(data, options["num_steps"]),
        batch_size=None,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    iterations = options["max_steps"] // environment.num_envs // options["num_steps"]
    if max_iterations is not None:
        iterations = int(max_iterations)
    hooks = [
        LoggerHook(interval=50),
        CheckpointHook(interval=options["save_interval"], max_keep_ckpts=3),
    ]
    runner = AccelerateRunner(
        bundle,
        trainer,
        loader,
        None,
        optimizers,
        {},
        accelerator,
        hooks,
        [],
        [],
        dict(by_epoch=False, max_iters=iterations, resume_skip_batches=False),
        str(Path(work_dir).expanduser().resolve()),
        auto_resume=auto_resume,
    )
    trainer.set_optimizers(runner.optimizers, runner.lr_schedulers)
    runner._handle_load()
    if original_state is not None and not runner.global_step:
        runner.global_step = int(original_state.get("epoch", 0))
    data.epoch = runner.global_step
    if accelerator.is_main_process:
        path = Path(work_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "training_options.json").write_text(
            json.dumps(options, indent=2) + "\n"
        )
        from motius.configuration.protomotions import original_deployment_options

        inference = Path(configuration).parent / "resolved_configs_inference.yaml"
        metadata = dict(
            network_options=bundle.network_options,
            deployment=original_deployment_options(
                inference if inference.is_file() else configuration
            ),
        )
        (path / "protomotions_bundle.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    return runner
