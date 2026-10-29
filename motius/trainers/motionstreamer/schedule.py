"""Original two-scheduler warmup/cosine transitions with complete native resume state."""

from torch.optim.lr_scheduler import (
    LRScheduler,
    LambdaLR,
    CosineAnnealingLR,
    MultiStepLR,
)


class MotionStreamerWarmupCosine(LRScheduler):
    def __init__(self, optimizer, warmup_iters, total_iters, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_iters = int(warmup_iters)
        self.total_iters = int(total_iters)
        self.min_lr = float(min_lr)
        self.iteration = 0
        self.warmup_scheduler = LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer, T_max=self.total_iters - self.warmup_iters, eta_min=self.min_lr
        )
        self.base_lrs = list(self.warmup_scheduler.base_lrs)
        self.last_epoch = -1
        self._step_count = 0
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def warmup_lambda(self, current_iter):
        if current_iter < self.warmup_iters:
            return float(current_iter) / float(max(1, self.warmup_iters))
        return 1.0

    def step(self, current_iter=None):
        current_iter = self.iteration if current_iter is None else int(current_iter)
        if current_iter < self.warmup_iters:
            self.warmup_scheduler.step()
        else:
            self.cosine_scheduler.step()
        self.iteration = current_iter + 1
        self.last_epoch = current_iter
        self._step_count += 1
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return dict(
            iteration=self.iteration,
            last_epoch=self.last_epoch,
            step_count=self._step_count,
            last_lr=self._last_lr,
            warmup=self.warmup_scheduler.state_dict(),
            cosine=self.cosine_scheduler.state_dict(),
            warmup_iters=self.warmup_iters,
            total_iters=self.total_iters,
            min_lr=self.min_lr,
        )

    def load_state_dict(self, state):
        self.iteration = state["iteration"]
        self.last_epoch = state["last_epoch"]
        self._step_count = state["step_count"]
        self._last_lr = state["last_lr"]
        self.warmup_iters = state["warmup_iters"]
        self.total_iters = state["total_iters"]
        self.min_lr = state["min_lr"]
        self.warmup_scheduler.load_state_dict(state["warmup"])
        self.cosine_scheduler.load_state_dict(state["cosine"])


class MotionStreamerTAESchedule(LRScheduler):
    def __init__(self, optimizer, warmup_iters=1000, milestones=(1900000,), gamma=0.05):
        self.optimizer = optimizer
        self.warmup_iters = int(warmup_iters)
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self.main_scheduler = MultiStepLR(
            optimizer, milestones=list(milestones), gamma=gamma
        )
        self.iteration = 0
        self.last_epoch = -1
        self._step_count = 0
        if self.warmup_iters > 1:
            for group, base in zip(optimizer.param_groups, self.base_lrs):
                group["lr"] = base * 2 / (self.warmup_iters + 1)
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def step(self):
        self.iteration += 1
        warmup_updates = max(0, self.warmup_iters - 1)
        if self.iteration < warmup_updates:
            for group, base in zip(self.optimizer.param_groups, self.base_lrs):
                group["lr"] = base * (self.iteration + 2) / (self.warmup_iters + 1)
        elif self.iteration > warmup_updates:
            self.main_scheduler.step()
        self.last_epoch = self.iteration - 1
        self._step_count += 1
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def state_dict(self):
        return dict(
            iteration=self.iteration,
            last_epoch=self.last_epoch,
            step_count=self._step_count,
            last_lr=self._last_lr,
            warmup_iters=self.warmup_iters,
            base_lrs=self.base_lrs,
            main=self.main_scheduler.state_dict(),
        )

    def load_state_dict(self, state):
        self.iteration = state["iteration"]
        self.last_epoch = state["last_epoch"]
        self._step_count = state["step_count"]
        self._last_lr = state["last_lr"]
        self.base_lrs = state["base_lrs"]
        self.warmup_iters = state["warmup_iters"]
        self.main_scheduler.load_state_dict(state["main"])
