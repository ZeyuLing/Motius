"""Rigid-pair invariants, runnable without model checkpoints or GPU packages."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

# This module has a NumPy-only contract; exercise it independently of the
# framework's optional training registry dependencies.
spec = importlib.util.spec_from_file_location(
    'interhuman_geometry', Path(__file__).parents[1] / 'motius/motion/representation/interhuman262.py'
)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)


@pytest.mark.parametrize('coordinates', ['y_up', 'interhuman_raw'])
@pytest.mark.parametrize('angle', [0., .7, np.pi])
def test_pair_preserves_all_contacts_and_associated_mesh(coordinates, angle):
    rng = np.random.default_rng(2026)
    joints = rng.normal(size=(8, 2, 22, 3)).astype(np.float32)
    joints[:, 0, 1] = [-.2, 1., 0.]
    joints[:, 0, 2] = [.2, 1., 0.]
    joints[:, 1, :, 1] += 3.  # Raised actor must not get its own ground plane.
    joints[:, 1, 20] = joints[:, 0, 21]  # Exact wrist contact.
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    joints = joints @ rotation.T + [2., .4, -3.]
    if coordinates == 'interhuman_raw':
        joints = joints @ module._RAW_TO_Y_UP
    local_rot = np.tile([1., 0., 0., 1., 0., 0.], (8, 2, 21, 1))
    original = joints.copy()
    encoded, transform = module.joints_pair_to_interhuman262(
        joints, local_rot, source_coordinates=coordinates, return_transform=True,
    )
    decoded = module.interhuman262_to_joints(encoded)
    before = np.linalg.norm(joints[:, 0, :, None] - joints[:, 1, None, :], axis=-1)
    after = np.linalg.norm(decoded[:, 0, :, None] - decoded[:, 1, None, :], axis=-1)
    np.testing.assert_allclose(after, before[:-1], atol=2e-6)
    np.testing.assert_allclose(decoded[:, 0, 21], decoded[:, 1, 20], atol=1e-6)
    np.testing.assert_allclose(transform.apply(joints)[:-1], decoded, atol=1e-6)
    np.testing.assert_array_equal(joints, original)
    np.testing.assert_allclose(transform.rotation @ transform.rotation.T, np.eye(3), atol=1e-6)
    np.testing.assert_allclose(module.interhuman262_to_joint_velocities(encoded)[:-1], np.diff(decoded, axis=0), atol=1e-6)


def test_pair_rejects_invalid_input():
    with pytest.raises(ValueError):
        module.joints_pair_to_interhuman262(np.zeros((8, 22, 3)), np.zeros((8, 21, 6)))
