"""Compose native SONIC PPO and online rollouts through the common Runner."""

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from motius.models.sonic import SONICPolicyBundle
from motius.trainers.sonic.online import SONICOnlinePPOTrainer
from motius.simulators.rl.sonic_rollout import SonicRollout
from motius.datasets.online_tracking import OnlineTrackingRollouts
from .accelerate_runner import AccelerateRunner, _BundleOrphanCheckpoint


class SonicTrainingState:
    def __init__(self, trainer):
        self.trainer = trainer

    def state_dict(self):
        state = dict(learning_rate=self.trainer.learning_rate)
        environment = self.trainer.collector.environment
        if hasattr(environment, "get_env_state_dict"):
            state["environment"] = environment.get_env_state_dict()
        return state

    def load_state_dict(self, state):
        self.trainer.learning_rate = float(state["learning_rate"])
        if "environment" in state:
            self.trainer.collector.environment.load_env_state_dict(state["environment"])
            self.trainer.collector.observations = None
        for optimizer in self.trainer.optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = self.trainer.learning_rate


class SonicMotionResampling:
    def __init__(self, environment, frequency, skip_frequency=None):
        self.environment, self.frequency = environment, int(frequency)
        self.skip_frequency = skip_frequency

    def after_train_iter(self, global_step, output=None):
        # Original callback runs after TRL increments global_step and tests +1.
        step = self.runner.global_step + 1
        skip = self.skip_frequency is not None and step % self.skip_frequency == 0
        if self.frequency > 0 and step % self.frequency == 0 and not skip:
            self.environment.resample_motion()


def build_sonic_runner(
    environment,
    work_dir,
    *,
    checkpoint,
    schema,
    configuration=None,
    options=None,
    accelerator=None,
    max_iterations=None,
    auto_resume=False,
):
    from motius.configuration.sonic import original_training_options
    from motius.hooks.checkpoint_hook import CheckpointHook
    from motius.hooks.logger_hook import LoggerHook

    configuration = configuration or Path(checkpoint).parent / "config.yaml"
    options = options or original_training_options(configuration)
    accelerator = accelerator or Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=options.get("gradient_accumulation_steps", 1),
    )
    bundle = SONICPolicyBundle.from_original_checkpoint(
        checkpoint, configuration=configuration, device=accelerator.device
    )
    # Same joint actor-then-value parameter order as the original TRL wrapper.
    parameters = list(bundle.policy.parameters()) + list(bundle.critic.parameters())
    optimizer = torch.optim.AdamW(parameters, **options["optimizer"])
    policy, critic, optimizer = accelerator.prepare(
        bundle.policy, bundle.critic, optimizer
    )
    bundle.policy, bundle.critic = policy, critic
    accelerator.register_for_checkpointing(_BundleOrphanCheckpoint(bundle))
    collector = SonicRollout(
        bundle, environment, schema, accelerator=accelerator, **options["rollout"]
    )
    trainer = SONICOnlinePPOTrainer(bundle, collector=collector, **options["trainer"])
    trainer.accelerator = accelerator
    loader = DataLoader(
        OnlineTrackingRollouts(collector, options["steps"]),
        batch_size=None,
        num_workers=0,
    )
    hooks = [
        LoggerHook(interval=50),
        CheckpointHook(interval=options["save_interval"], max_keep_ckpts=3),
    ]
    if options.get("resample_frequency", 0) > 0 and hasattr(
        environment, "resample_motion"
    ):
        hooks.insert(
            0,
            SonicMotionResampling(
                environment,
                options["resample_frequency"],
                options.get("skip_resample_frequency"),
            ),
        )
    runner = AccelerateRunner(
        bundle,
        trainer,
        loader,
        None,
        {"joint": optimizer},
        {},
        accelerator,
        hooks,
        [],
        [],
        dict(
            by_epoch=False,
            max_iters=int(max_iterations or options["iterations"]),
            resume_skip_batches=False,
        ),
        str(Path(work_dir).expanduser().resolve()),
        auto_resume=auto_resume,
    )
    trainer.set_optimizers(runner.optimizers, runner.lr_schedulers)
    accelerator.register_for_checkpointing(SonicTrainingState(trainer))
    runner._handle_load()
    if accelerator.is_main_process:
        path = Path(work_dir)
        path.mkdir(parents=True, exist_ok=True)
        (path / "training_options.json").write_text(
            json.dumps(options, indent=2) + "\n"
        )
    return runner, collector
