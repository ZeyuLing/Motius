from __future__ import annotations

import numpy as np
import pytest

from motius.simulators.g1 import G1_JOINT_NAMES
from motius.simulators.mujoco.g1_tracking import DEFAULT_DOF_POS
from motius.simulators.reference import TrackingReference, load_g1_reference


def test_opentrack_named_qpos_extends_and_reorders_g1_joints(tmp_path):
    source_names = ["root", G1_JOINT_NAMES[3], G1_JOINT_NAMES[0]]
    source_qpos = np.zeros((4, 9), dtype=np.float32)
    source_qpos[:, 2] = 0.8
    source_qpos[:, 3] = 1.0
    source_qpos[:, 7] = np.arange(4, dtype=np.float32)
    source_qpos[:, 8] = 10 + np.arange(4, dtype=np.float32)
    source = tmp_path / "named_g1.npz"
    np.savez(source, qpos=source_qpos, joint_names=source_names, frequency=40)

    reference = load_g1_reference(source, target_fps=40)

    assert reference.qpos.shape == (4, 36)
    np.testing.assert_array_equal(reference.qpos[:, 7 + 3], source_qpos[:, 7])
    np.testing.assert_array_equal(reference.qpos[:, 7], source_qpos[:, 8])
    untouched = [index for index in range(29) if index not in {0, 3}]
    assert np.all(reference.qpos[:, 7 + np.asarray(untouched)] == 0.0)
    assert reference.fps == 40


def test_tracking_reference_windows_cover_only_valid_remainder():
    qpos = np.zeros((2301, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    qvel = np.zeros((2301, 35), dtype=np.float32)
    reference = TrackingReference("motion", 50, qpos, qvel)

    windows = list(reference.iter_windows(1000, minimum_remainder_steps=250))

    assert [window.num_frames for window in windows] == [1001, 1001, 301]
    assert [window.name for window in windows] == [
        "motion__f000000_001000",
        "motion__f001000_002000",
        "motion__f002000_002300",
    ]


def test_mujoco_any2track_closed_loop_smoke():
    pytest.importorskip("mujoco")
    from motius.simulators.mujoco import MujocoG1TrackingEnvironment

    qpos = np.zeros((4, 36), dtype=np.float32)
    qpos[:, :3] = (0.0, 0.0, 0.793)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = DEFAULT_DOF_POS
    reference = TrackingReference(
        "stationary",
        50,
        qpos,
        np.zeros((4, 35), dtype=np.float32),
    )

    class Bundle:
        METHOD_NAME = "Any2Track"

    class Pipeline:
        bundle = Bundle()

        def infer_motion_tracking(self, observation):
            assert set(observation) == {
                "dif_joint_pos",
                "dif_joint_vel",
                "gvec_pelvis",
                "gyro_pelvis",
                "joint_pos",
                "joint_vel",
                "last_motor_targets",
                "ref_feet_height",
                "ref_root_height",
            }
            return {"continuous_actions": np.zeros((1, 29), dtype=np.float32)}

    with MujocoG1TrackingEnvironment(reference=reference) as environment:
        environment.reset()
        while not environment.done:
            environment.step(Pipeline(), Pipeline().infer_motion_tracking(**environment.policy_inputs(Pipeline())))
        result = environment.result()

    assert result.qpos.shape == (4, 36)
    assert result.metrics["completed_steps"] == 3
    assert result.termination_reason is None


def test_mujoco_humanoid_gpt_observation_adapter_smoke():
    pytest.importorskip("mujoco")
    from motius.models.humanoid_gpt import HumanoidGPTBundle
    from motius.simulators.mujoco import MujocoG1TrackingEnvironment

    qpos = np.zeros((3, 36), dtype=np.float32)
    qpos[:, :3] = (0.0, 0.0, 0.793)
    qpos[:, 3] = 1.0
    qpos[:, 7:] = DEFAULT_DOF_POS
    reference = TrackingReference(
        "stationary",
        50,
        qpos,
        np.zeros((3, 35), dtype=np.float32),
    )

    class Bundle:
        METHOD_NAME = "HumanoidGPT"

    class Pipeline:
        bundle = Bundle()

        def infer_motion_tracking(self, *, components):
            assert tuple(components) == tuple(
                name for name, _ in HumanoidGPTBundle.OBSERVATION_LAYOUT
            )
            for name, dimension in HumanoidGPTBundle.OBSERVATION_LAYOUT:
                assert components[name].shape == (1, dimension)
            return {
                "continuous_actions": np.zeros((1, 29), dtype=np.float32),
                "motor_targets": DEFAULT_DOF_POS[None],
            }

    pipeline = Pipeline()
    with MujocoG1TrackingEnvironment(reference=reference) as environment:
        environment.reset()
        while not environment.done:
            inputs = environment.policy_inputs(pipeline)
            environment.step(
                pipeline,
                pipeline.infer_motion_tracking(**inputs),
            )
        result = environment.result()

    assert result.qpos.shape == (3, 36)
    assert result.metrics["completed_steps"] == 2


def test_isaaclab_backend_requires_launcher_registration():
    from motius.simulators.isaaclab import create_isaaclab_tracking_environment

    with pytest.raises(RuntimeError, match="AppLauncher"):
        create_isaaclab_tracking_environment(reference=object())
