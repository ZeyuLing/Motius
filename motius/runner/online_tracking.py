"""Native on-policy tracking composition using the existing AccelerateRunner."""

from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from motius.models.beyondmimic import BeyondMimicPolicyBundle
from motius.trainers.beyondmimic import BeyondMimicTrainer
from motius.datasets.online_tracking import OnlineTrackingRollouts
from motius.simulators.rl.rollout import VectorPolicyRollout
from .accelerate_runner import AccelerateRunner, _BundleOrphanCheckpoint


def beyondmimic_training_options(variant='flat'):
    path = Path(__file__).resolve().parents[1] / 'configuration/beyondmimic/g1.json'
    options = json.loads(path.read_text())
    if variant == 'low_frequency':
        scale = options['low_frequency_scale']
        options['num_steps_per_env'] = round(options['num_steps_per_env'] * scale)
        for key in ['gamma', 'lam']:
            options['algorithm'][key] = options['algorithm'][key] ** (1 / scale)
    return options


class _PPOProgressState:
    def __init__(self, trainer):
        self.trainer = trainer

    def state_dict(self):
        return dict(learning_rate=self.trainer.learning_rate)

    def load_state_dict(self, state):
        self.trainer.learning_rate = float(state['learning_rate'])


def build_beyondmimic_runner(
    environment,
    work_dir,
    *,
    accelerator=None,
    variant='flat',
    max_iterations=None,
    checkpoint=None,
    auto_resume=False,
    options=None,
    random_episode_lengths=True,
):
    """Bind fresh vector rollouts to native neural roots, PPO and runner hooks.

    The dataloader is deliberately process-local: each distributed rank owns an
    environment shard. Accelerator prepares neural/optimizer state only.
    """
    from motius.hooks.checkpoint_hook import CheckpointHook
    from motius.hooks.logger_hook import LoggerHook

    options = options or beyondmimic_training_options(variant)
    accelerator = accelerator or Accelerator(mixed_precision='no')
    work_dir = Path(work_dir).expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    observations, extras = environment.get_observations()
    critic = extras.get('observations', {}).get('critic', observations)
    if checkpoint is not None:
        path = Path(checkpoint)
        if path.is_dir():
            bundle = BeyondMimicPolicyBundle.from_pretrained(
                path, device=accelerator.device
            )
        else:
            bundle = BeyondMimicPolicyBundle.from_original_checkpoint(
                path,
                activation=options['policy']['activation'],
                device=accelerator.device,
            )
        expected = bundle.network_options
        if (
            expected['num_actor_obs'] != observations.shape[-1]
            or expected['num_critic_obs'] != critic.shape[-1]
        ):
            raise ValueError(
                'Checkpoint observation dimensions do not match the simulator'
            )
    else:
        neural = {
            **options['policy'],
            **dict(
                num_actor_obs=observations.shape[-1],
                num_critic_obs=critic.shape[-1],
                num_actions=environment.num_actions,
            ),
        }
        bundle = BeyondMimicPolicyBundle(
            neural,
            empirical_normalization=options['empirical_normalization'],
            device=accelerator.device,
        )
    algorithm = {
        key: value
        for key, value in options['algorithm'].items()
        if key not in {'gamma', 'lam'}
    }
    trainer = BeyondMimicTrainer(bundle, **algorithm)
    optimizer = torch.optim.Adam(
        bundle.trainable_parameters(), lr=algorithm['learning_rate']
    )
    if checkpoint is not None and Path(checkpoint).is_file():
        from motius.utils.torch_checkpoint import load_torch_checkpoint

        source = load_torch_checkpoint(checkpoint, weights_only=False)
        if 'optimizer_state_dict' in source:
            optimizer.load_state_dict(source['optimizer_state_dict'])
            trainer.learning_rate = optimizer.param_groups[0]['lr']
    bundle.network, optimizer = accelerator.prepare(bundle.network, optimizer)
    accelerator.register_for_checkpointing(_BundleOrphanCheckpoint(bundle))
    accelerator.register_for_checkpointing(bundle.actor_normalizer)
    accelerator.register_for_checkpointing(bundle.critic_normalizer)
    accelerator.register_for_checkpointing(_PPOProgressState(trainer))

    # The collector initializes lazily after resume restores weights, statistics
    # and RNG. It keeps its normalized observation state across PPO updates.
    class Collector:
        value = None

        def collect(self, steps):
            if self.value is None:
                base = getattr(environment, 'environment', environment)
                if random_episode_lengths and hasattr(base, 'episode_length_buf'):
                    base.episode_length_buf = torch.randint_like(
                        base.episode_length_buf, high=int(base.max_episode_length)
                    )
                self.value = VectorPolicyRollout(
                    bundle,
                    environment,
                    gamma=options['algorithm']['gamma'],
                    lam=options['algorithm']['lam'],
                )
            return self.value.collect(steps)

    dataset = OnlineTrackingRollouts(Collector(), options['num_steps_per_env'])
    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )
    hooks = [
        LoggerHook(interval=50),
        CheckpointHook(interval=options['save_interval'], max_keep_ckpts=3),
    ]
    runner = AccelerateRunner(
        bundle,
        trainer,
        loader,
        None,
        dict(policy=optimizer),
        {},
        accelerator,
        hooks,
        [],
        [],
        dict(
            by_epoch=False,
            max_iters=(
                options['max_iterations']
                if max_iterations is None
                else int(max_iterations)
            ),
            resume_skip_batches=False,
        ),
        str(work_dir),
        auto_resume=auto_resume,
    )
    trainer.set_optimizers(runner.optimizers, runner.lr_schedulers)
    runner._handle_load()
    if accelerator.is_main_process:
        (work_dir / 'training_options.json').write_text(
            json.dumps(options, indent=2) + '\n'
        )
    return runner
