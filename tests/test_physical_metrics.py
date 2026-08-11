import numpy as np
import pytest

from motius.evaluation.metrics import (
    aggregate_physical_metrics,
    compute_physical_metrics,
    physical_metrics_from_motion,
    table_scaled_physical_metrics,
)


def _static_skeleton(frames: int = 8) -> np.ndarray:
    joints = np.zeros((frames, 22, 3), dtype=np.float32)
    joints[:, :, 1] = np.linspace(0.0, 1.0, 22, dtype=np.float32)
    joints[:, 10:12, 1] = 0.0
    return joints


def test_static_motion_has_zero_joint_physics_error():
    metrics = compute_physical_metrics(_static_skeleton())

    assert metrics == {
        "Jitter": 0.0,
        "Dynamic": 0.0,
        "Penet": 0.0,
        "Float": 0.0,
        "Slide": 0.0,
    }


def test_rigid_translation_reports_motion_and_contact_slide():
    joints = _static_skeleton()
    joints[:, :, 0] += np.arange(len(joints), dtype=np.float32)[:, None] * 0.01

    metrics = physical_metrics_from_motion(joints.reshape(len(joints), 66), "joints66")

    assert metrics["Jitter"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["Dynamic"] == pytest.approx(0.01, rel=1e-5)
    assert metrics["Slide"] == pytest.approx(0.01, rel=1e-5)
    assert metrics["Penet"] == 0.0


def test_aggregate_and_table_scaling():
    metrics = {
        "Jitter": 0.002,
        "Dynamic": 0.003,
        "Penet": 0.004,
        "Float": 0.05,
        "Slide": 0.006,
    }

    aggregate = aggregate_physical_metrics([metrics, metrics, None])
    scaled = table_scaled_physical_metrics(aggregate)

    assert aggregate["n"] == 2
    assert scaled == {
        "Slide": 6.0,
        "Float": 5.0,
        "Jitter": 2.0,
        "Dynamic": 3.0,
        "Penet": 4.0,
    }


def test_validation_and_motion135_requires_offsets():
    with pytest.raises(ValueError, match="At least four frames"):
        compute_physical_metrics(np.zeros((3, 22, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="bone_offsets"):
        physical_metrics_from_motion(np.zeros((4, 135), dtype=np.float32), "motion135")
