"""EMA (Exponential Moving Average) hook."""

import copy
from hftrainer.registry import HOOKS
from hftrainer.utils.logger import get_logger

logger = get_logger()


@HOOKS.register_module()
class EMAHook:
    """
    Maintains an EMA copy of the trainable modules.

    Note:
      - This hook only updates ``ema_bundle`` state.
      - The runner does not automatically swap EMA weights into validation or
        inference yet.
    """

    priority = 15

    def __init__(self, decay: float = 0.9999, update_interval: int = 1):
        self.decay = decay
        self.update_interval = update_interval
        self.runner = None
        self.ema_bundle = None

    def before_run(self):
        if self.runner is None:
            return
        bundle = self.runner.bundle
        self.ema_bundle = copy.deepcopy(bundle)
        self.ema_bundle.requires_grad_(False)
        self.ema_bundle.eval()
        logger.info("EMA hook initialized.")

    def after_train_iter(self, global_step: int, output: dict = None):
        if self.ema_bundle is None:
            return
        if (global_step + 1) % self.update_interval != 0:
            return

        bundle = self.runner.bundle
        for name in bundle._trainable_modules:
            ema_module = getattr(self.ema_bundle, name, None)
            src_module = getattr(bundle, name, None)
            if ema_module is None or src_module is None:
                continue
            for ema_p, src_p in zip(ema_module.parameters(), src_module.parameters()):
                ema_p.data.mul_(self.decay).add_(src_p.data, alpha=1.0 - self.decay)
