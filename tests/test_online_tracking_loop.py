"""Iteration budgets must not consume unused on-policy simulator rollouts."""
from motius.runner.loops import IterBasedLoop


class Rollouts:
    def __init__(self):
        self.collected = 0

    def __iter__(self):
        while True:
            self.collected += 1
            yield self.collected


def test_iteration_budget_does_not_collect_an_unused_rollout():
    stream = Rollouts()
    assert list(IterBasedLoop(stream, 2).iter_batches()) == [(0, 1), (1, 2)]
    assert stream.collected == 2


def test_online_resume_only_collects_remaining_iterations():
    stream = Rollouts()
    assert list(IterBasedLoop(stream, 5, start_iter=3).iter_batches()) == [(3, 1), (4, 2)]
    assert stream.collected == 2


def test_completed_online_resume_does_not_start_the_simulator():
    stream = Rollouts()
    assert list(IterBasedLoop(stream, 5, start_iter=5).iter_batches()) == []
    assert stream.collected == 0
