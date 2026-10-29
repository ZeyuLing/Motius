"""Restoring adaptive motion statistics preserves the source reset boundary."""

from types import SimpleNamespace
from motius.simulators.sonic.environment import SonicG1Environment


def test_restore_resets_only_when_the_motion_library_changes():
    for changed in (False, True):
        calls = []
        library = SimpleNamespace(
            use_adaptive_sampling=True,
            load_state_dict=lambda value: calls.append(("load", value)),
            load_motions_for_training=lambda **kw: changed,
        )
        env = object.__new__(SonicG1Environment)
        env._motion_lib = library
        env.num_envs = 4
        env.motion_command = SimpleNamespace(max_num_load_motions=8)
        env.reset_all = lambda: calls.append(("reset", None))
        env.load_env_state_dict({"motion_lib": {"weight": 1}})
        assert calls == [("load", {"weight": 1})] + (
            [("reset", None)] if changed else []
        )
