"""Vector rollout processing shared by native policy trainers."""

from .advantages import generalized_advantage_estimation, rollout_minibatches

__all__ = ['generalized_advantage_estimation', 'rollout_minibatches']
