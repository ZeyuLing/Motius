"""Isaac Lab motion-tracking backend registration.

Isaac Lab must be launched before simulator objects are imported. Motius keeps
that lifecycle in the engine-specific runner and uses this registry to connect
the launched environment to the backend-neutral rollout loop.
"""

from __future__ import annotations

from typing import Any, Callable, Optional


IsaacLabEnvironmentFactory = Callable[..., Any]
_ENVIRONMENT_FACTORY: Optional[IsaacLabEnvironmentFactory] = None


def register_isaaclab_tracking_environment(
    factory: IsaacLabEnvironmentFactory,
) -> None:
    """Register the launched Isaac Lab environment factory for this process."""

    global _ENVIRONMENT_FACTORY
    if not callable(factory):
        raise TypeError("Isaac Lab environment factory must be callable.")
    _ENVIRONMENT_FACTORY = factory


def create_isaaclab_tracking_environment(**kwargs: Any) -> Any:
    """Create the registered Isaac Lab environment after AppLauncher startup."""

    if _ENVIRONMENT_FACTORY is None:
        raise RuntimeError(
            "Register an Isaac Lab environment after AppLauncher startup with "
            "`register_isaaclab_tracking_environment(factory)`. Direct "
            "in-process launch is intentionally disabled because "
            "AppLauncher must initialize Isaac Sim before Isaac Lab imports."
        )
    return _ENVIRONMENT_FACTORY(**kwargs)


__all__ = [
    "IsaacLabEnvironmentFactory",
    "create_isaaclab_tracking_environment",
    "register_isaaclab_tracking_environment",
]
