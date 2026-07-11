"""LR Scheduler hook (legacy, schedulers are handled by AccelerateRunner)."""

from hftrainer.registry import HOOKS


@HOOKS.register_module()
class LRSchedulerHook:
    """
    Placeholder hook for LR scheduler stepping.

    AccelerateRunner already steps schedulers directly after optimizer steps,
    so this hook is effectively a no-op and kept for config compatibility.
    """

    priority = 20

    def __init__(self, by_epoch: bool = False):
        self.by_epoch = by_epoch
        self.runner = None
