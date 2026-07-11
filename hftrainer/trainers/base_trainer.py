"""
BaseTrainer: abstract base class for all task trainers.

Trainers are responsible for:
  - Assembling the training/validation forward graph
  - Computing the loss
  - Returning structured output dicts

Trainers do NOT handle (by default):
  - Optimizer creation / step (done by AccelerateRunner)
  - Checkpoint saving/loading (done by CheckpointHook)
  - Distributed communication (done by Accelerator)

Multi-optimizer trainers (GAN, DMD distillation, Self-Forcing, etc.):
  Set ``trainer_controls_optimization = True`` as a class attribute.
  When this flag is True:
    - The runner injects optimizers/schedulers via set_optimizers()
    - The runner SKIPS backward/step/zero_grad in the training loop
    - The trainer must call self.accelerator.backward(loss), opt.step(),
      opt.zero_grad(), and sched.step() inside train_step()
    - train_step should return {'loss': None, ...} with per-phase losses
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import torch.nn as nn

from hftrainer.models.base_model_bundle import ModelBundle


class BaseTrainer(nn.Module, ABC):
    """
    Abstract base class for all trainers.

    Subclasses must implement:
      - train_step(batch) -> dict  with at least {'loss': Tensor}
      - val_step(batch) -> dict    with task-specific keys

    The trainer holds a reference to the ModelBundle and calls its
    atomic forward methods (encode_text, predict_noise, etc.).
    The accelerator instance is injected by AccelerateRunner after prepare().

    Multi-optimizer protocol:
      Set ``trainer_controls_optimization = True`` to take full control of
      backward/step/zero_grad within train_step(). The runner will inject
      optimizers via set_optimizers() and skip its own optimization logic.
    """

    # When True, the runner delegates optimization entirely to the trainer.
    # The trainer must call accelerator.backward(), opt.step(), opt.zero_grad(),
    # and sched.step() inside train_step().
    trainer_controls_optimization = False

    def __init__(self, bundle: ModelBundle, **kwargs):
        super().__init__()
        self.bundle = bundle
        self.accelerator = None  # injected by AccelerateRunner
        self.runner = None  # injected by AccelerateRunner
        self.optimizers: Dict[str, Any] = {}
        self.lr_schedulers: Dict[str, Any] = {}

    def set_optimizers(
        self,
        optimizers: Dict[str, Any],
        lr_schedulers: Optional[Dict[str, Any]] = None,
    ):
        """
        Inject optimizers and LR schedulers into the trainer.

        Called by AccelerateRunner when trainer_controls_optimization=True.
        After this call, use get_optimizer(name) and get_lr_scheduler(name)
        to access them in train_step().
        """
        self.optimizers = optimizers
        self.lr_schedulers = lr_schedulers or {}

    def get_optimizer(self, name: str):
        """
        Get a named optimizer.

        Args:
            name: optimizer name as defined in the config (e.g. 'generator',
                  'discriminator', 'student')

        Raises:
            KeyError: if the optimizer name is not found
        """
        if name not in self.optimizers:
            raise KeyError(
                f"Optimizer '{name}' not found. "
                f"Available: {list(self.optimizers.keys())}"
            )
        return self.optimizers[name]

    def get_lr_scheduler(self, name: str):
        """
        Get a named LR scheduler. Returns None if not found.

        Args:
            name: scheduler name (should match optimizer name)
        """
        return self.lr_schedulers.get(name)

    def get_global_step(self) -> int:
        """
        Return the number of completed training steps.

        The runner updates ``global_step`` after each successful train_step(),
        so inside ``train_step()`` this value represents the number of fully
        completed iterations before the current one starts.
        """
        if self.runner is None:
            return 0
        return int(getattr(self.runner, 'global_step', 0))

    def get_current_step(self) -> int:
        """
        Return the 1-based step index of the current iteration.

        Example:
          - Before the very first optimization step: current_step == 1
          - After 5000 completed iterations:        current_step == 5001
        """
        return self.get_global_step() + 1

    def get_discriminator_factor(
        self,
        base_weight: float = 1.0,
        start_step: int = 0,
        warmup_steps: int = 0,
        schedule: str = 'linear',
    ) -> float:
        """
        Return the effective discriminator/adversarial weight for this step.

        Args:
            base_weight: final weight after warmup.
            start_step: discriminator becomes active after this many completed
                training steps.
            warmup_steps: number of active steps used to ramp from 0 to
                ``base_weight``. ``0`` means no ramp-up.
            schedule: ``'linear'`` or ``'constant'``.
        """
        start_step = max(0, int(start_step))
        warmup_steps = max(0, int(warmup_steps))
        schedule = str(schedule).lower()

        steps_since_start = self.get_current_step() - start_step
        if steps_since_start <= 0:
            return 0.0

        if warmup_steps == 0 or schedule == 'constant':
            return float(base_weight)

        if schedule != 'linear':
            raise ValueError(
                f"Unsupported discriminator schedule '{schedule}'. "
                "Expected 'linear' or 'constant'."
            )

        ramp = min(1.0, steps_since_start / float(warmup_steps))
        return float(base_weight) * ramp

    def should_update_discriminator(
        self,
        start_step: int = 0,
        update_interval: int = 1,
    ) -> bool:
        """
        Return whether the discriminator optimizer should step this iteration.

        The update interval is counted relative to the first active
        discriminator step so that delayed starts do not shift the cadence.
        """
        update_interval = max(1, int(update_interval))
        steps_since_start = self.get_current_step() - max(0, int(start_step))
        if steps_since_start <= 0:
            return False
        return (steps_since_start - 1) % update_interval == 0

    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one training step.

        Args:
            batch: dict from DataLoader

        Returns:
            dict with at least {'loss': Tensor}. May include additional
            loss components for logging (e.g. 'loss_mse', 'loss_kl', ...).

            When trainer_controls_optimization=True, return {'loss': None}
            plus per-phase losses (e.g. 'loss_d', 'loss_g') for logging.
        """

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform one validation step.

        Args:
            batch: dict from DataLoader

        Returns:
            Task-specific dict. See method documentation for per-task key
            conventions.
            Default implementation raises NotImplementedError.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement val_step(). "
            "Override this method to enable validation."
        )

    def get_bundle(self) -> ModelBundle:
        """Return the ModelBundle held by this trainer."""
        return self.bundle

    @staticmethod
    def sum_train_losses(losses: Dict[str, Any]):
        """Sum differentiable training losses while leaving diagnostics out.

        Some loss modules return detached per-component scalars for logging
        next to the actual optimization terms.  Including those detached
        diagnostics in ``result['loss']`` inflates the reported loss scale
        without changing gradients, which makes training curves misleading.
        """
        train_losses = [
            v for v in losses.values()
            if getattr(v, 'requires_grad', False)
        ]
        if not train_losses:
            keys = ', '.join(losses.keys()) or '<empty>'
            raise ValueError(
                'No differentiable training losses found. '
                f'Available loss keys: {keys}'
            )

        total = train_losses[0]
        for value in train_losses[1:]:
            total = total + value
        return total

    def forward(self, *args, **kwargs):
        """Redirect forward() to train_step() for compatibility."""
        return self.train_step(*args, **kwargs)
