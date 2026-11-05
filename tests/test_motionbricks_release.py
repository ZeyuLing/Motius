"""Native MotionBricks architecture, input contracts and coordinate conventions."""

from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from motius.motion.conditioning.motionbricks_release import (
    validate_release_inputs,
    release_input_shapes,
)
from motius.pipelines.motionbricks import MotionBricksReleasePipeline
from motius.models.motionbricks.release_g1 import G1PoseNetwork, G1RootNetwork
from motius.models.motionbricks.release_soma import SOMAPoseNetwork, SOMARootNetwork
from test_motionbricks import options


@pytest.mark.parametrize(
    "pose,root", [(G1PoseNetwork, G1RootNetwork), (SOMAPoseNetwork, SOMARootNetwork)]
)
def test_neural_roots_do_not_own_motion_representation(pose, root):
    cfg = options()
    for module in (pose(cfg["pose"]), root(cfg["root"])):
        assert not hasattr(module, "motion_rep")
        assert not hasattr(module, "local_motion_rep")
        assert not hasattr(module, "global_motion_rep")
    root_module = root(cfg["root"])
    assert hasattr(root_module, "encode_conditions")
    assert hasattr(root_module, "decode_motion")


@pytest.mark.parametrize("model", ["g1", "soma"])
@pytest.mark.parametrize(
    "failure", ["mask", "tokens", "batch", "nan", "shape", "missing"]
)
def test_reject_invalid_inputs(model, failure):
    values = {
        k: np.ones((2, *shape)) for k, shape in release_input_shapes(model).items()
    }
    key = "context_qpos" if model == "g1" else "local_poses"
    if failure == "mask":
        values["has_local_poses"][0, 0] = 0.5
    elif failure == "tokens":
        values["allowed_pred_num_tokens"][0] = 0
    elif failure == "batch":
        values[key] = values[key][:1]
    elif failure == "nan":
        values[key][0, 0, 0] = np.nan
    elif failure == "shape":
        values[key] = values[key][:, :3]
    else:
        values.pop(key)
    with pytest.raises(ValueError):
        validate_release_inputs(values, model, "cpu")


def test_g1_zero_quaternion_rejected():
    values = {
        k: np.ones((2, *shape)) for k, shape in release_input_shapes("g1").items()
    }
    values["context_qpos"][0, 0, 3:7] = 0
    with pytest.raises(ValueError, match="zero root"):
        validate_release_inputs(values, "g1", "cpu")


def test_g1_quaternion_boundary_and_length_contract():
    pipe = object.__new__(MotionBricksReleasePipeline)
    pipe.bundle = SimpleNamespace(model="g1")

    def raw(inputs):
        assert inputs["context_qpos"][0, 0, 3:7].tolist() == [
            0.2,
            0.3,
            0.4,
            0.1,
        ] or np.allclose(inputs["context_qpos"][0, 0, 3:7], [0.2, 0.3, 0.4, 0.1])
        return {
            "pred_qpos": torch.as_tensor(
                np.repeat(inputs["context_qpos"][:, :1], 64, axis=1)
            ),
            "pred_num_tokens": torch.tensor([6, 16]),
            "pred_foot_contacts": torch.ones(2, 64, 4),
        }

    pipe.infer_raw = raw
    q = np.zeros((2, 4, 36), dtype=np.float32)
    q[..., 3:7] = [0.1, 0.2, 0.3, 0.4]
    before = q.copy()
    result = pipe.infer_g1_inbetweening(q, q)
    np.testing.assert_array_equal(before, q)
    np.testing.assert_array_equal(result["qpos"][0][0], q[0, 0])
    assert result["lengths"].tolist() == [18, 58]
    assert result["representation"] == "g1_qpos" and not result["endpoint_corrected"]


def test_no_onnx_or_upstream_runtime_dependency():
    root = Path(__file__).resolve().parents[1] / "motius"
    for folder in ["models/motionbricks", "pipelines/motionbricks"]:
        for file in (root / folder).glob("release*.py"):
            text = file.read_text()
            assert "import onnxruntime" not in text
            assert "from motionbricks." not in text
            assert "from motionbricks_onnx" not in text
