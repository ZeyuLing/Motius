from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from motius.models.any2track import Any2TrackBundle
from motius.models.beyondmimic import BeyondMimicBundle
from motius.models.humanoid_gpt import HumanoidGPTBundle
from motius.models.protomotions import ProtoMotionsBundle
from motius.models.sonic import SONICBundle
from motius.pipelines.auto import Pipeline


class FakeSession:
    def __init__(self, outputs, metadata=None, inputs=None):
        self.outputs = outputs
        self.inputs = dict(inputs or {})
        self.metadata = dict(metadata or {})
        self.last_feeds = None

    def run(self, feeds, output_names=None):
        self.last_feeds = feeds
        if output_names is None:
            return dict(self.outputs)
        return {name: self.outputs[name] for name in output_names}


def _touch_files(root: Path, files: dict[str, str]) -> dict[str, Path]:
    paths = {}
    for role, relative in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
        paths[role] = path
    return paths


def test_sonic_preserves_official_encoder_decoder_layout(tmp_path):
    files = _touch_files(tmp_path, dict(SONICBundle.DEFAULT_FILES))
    bundle = SONICBundle(file_paths=files, load_model=False)
    encoder = FakeSession(
        {"encoded_tokens": np.full((1, 64), 3.0, dtype=np.float32)}
    )
    decoder = FakeSession({"action": np.ones((1, 29), dtype=np.float32)})
    bundle._sessions = {"encoder": encoder, "decoder": decoder}

    result = bundle.forward(
        np.zeros((1, 1762), dtype=np.float32),
        np.full((1, 930), 5.0, dtype=np.float32),
    )

    decoder_input = decoder.last_feeds["obs_dict"]
    np.testing.assert_array_equal(decoder_input[:, :64], 3.0)
    np.testing.assert_array_equal(decoder_input[:, 64:], 5.0)
    assert result["action"].shape == (1, 29)


def test_sonic_rejects_batching_not_supported_by_official_graph(tmp_path):
    class Spec:
        def __init__(self, shape):
            self.shape = shape

    files = _touch_files(tmp_path, dict(SONICBundle.DEFAULT_FILES))
    bundle = SONICBundle(file_paths=files, load_model=False)
    bundle._sessions = {
        "encoder": FakeSession(
            {"encoded_tokens": Spec([1, 64])},
            inputs={"obs_dict": Spec([1, 1762])},
        ),
        "decoder": FakeSession(
            {"action": Spec([1, 29])},
            inputs={"obs_dict": Spec([1, 994])},
        ),
    }

    try:
        bundle.encode(np.zeros((2, 1762), dtype=np.float32))
    except ValueError as error:
        assert "shape [1, 1762]" in str(error)
    else:
        raise AssertionError("SONIC must reject a batch size unsupported by its graph")


def test_sonic_reads_dimensions_from_exported_onnx_contract(tmp_path):
    class Spec:
        def __init__(self, shape):
            self.shape = shape

    files = _touch_files(tmp_path, dict(SONICBundle.DEFAULT_FILES))
    bundle = SONICBundle(file_paths=files, load_model=False)
    encoder = FakeSession(
        {"encoded_tokens": Spec([1, 64])},
        inputs={"obs_dict": Spec([1, 1751])},
    )
    decoder = FakeSession(
        {"action": Spec([1, 29])},
        inputs={"obs_dict": Spec([1, 994])},
    )
    bundle._sessions = {"encoder": encoder, "decoder": decoder}

    contract = bundle.artifact_contract()

    assert contract["encoder_input"]["shape"] == [1, 1751]
    assert contract["decoder_state_dimension"] == 930


def test_protomotions_forwards_named_inputs_without_reordering(tmp_path):
    files = _touch_files(tmp_path, dict(ProtoMotionsBundle.DEFAULT_FILES))
    bundle = ProtoMotionsBundle(file_paths=files, load_model=False)
    outputs = {
        "actions": np.zeros((1, 29), dtype=np.float32),
        "joint_pos_targets": np.ones((1, 29), dtype=np.float32),
        "stiffness_targets": np.ones((1, 29), dtype=np.float32) * 2,
        "damping_targets": np.ones((1, 29), dtype=np.float32) * 3,
    }
    session = FakeSession(outputs)
    bundle._sessions = {"policy": session}
    feeds = {
        name: np.asarray([index], dtype=np.float32)
        for index, name in enumerate(bundle.INPUT_NAMES)
    }

    assert bundle.forward(feeds) is not outputs
    assert list(session.last_feeds) == list(feeds)
    for name in feeds:
        assert session.last_feeds[name] is feeds[name]


def test_protomotions_reads_the_exported_observation_contract(tmp_path):
    files = _touch_files(tmp_path, dict(ProtoMotionsBundle.DEFAULT_FILES))
    files["deployment_config"].write_text(
        yaml.safe_dump(
            {
                "dt": 0.02,
                "policy_inputs": [
                    {
                        "name": "current_rigid_body_pos",
                        "key": "current.rigid_body_pos",
                        "shape": [1, 33, 3],
                    },
                    {
                        "name": "historical_actions",
                        "key": "historical.actions",
                        "shape": [1, 1, 29],
                    },
                ],
                "policy_outputs": [{"name": "actions"}],
                "_runtime": {
                    "onnx_in_names": [
                        "current_rigid_body_pos",
                        "historical_actions",
                    ],
                    "onnx_out_names": ["actions"],
                },
                "motion": {"future_step_indices": [1]},
            }
        )
    )
    bundle = ProtoMotionsBundle(file_paths=files, load_model=False)

    contract = bundle.artifact_contract()

    assert contract["inputs"] == [
        "current_rigid_body_pos",
        "historical_actions",
    ]
    assert contract["future_steps"] == [1]
    assert contract["future_seconds"] == [0.02]


def _any2track_training_config() -> dict:
    return {
        "env_config": {
            "ctrl_dt": 0.02,
            "obs_keys": [
                "dif_joint_pos",
                "dif_joint_vel",
                "gvec_pelvis",
                "gyro_pelvis",
                "joint_pos",
                "joint_vel",
                "last_motor_targets",
                "ref_feet_height",
                "ref_root_height",
            ],
        },
        "policy_config": {"policy_args": {"obs_dim": 156, "act_dim": 29}},
    }


def test_any2track_assembles_official_156d_observation(tmp_path):
    files = _touch_files(tmp_path, dict(Any2TrackBundle.DEFAULT_FILES))
    files["training_config"].write_text(json.dumps(_any2track_training_config()))
    bundle = Any2TrackBundle(file_paths=files, load_model=False)
    dims = [29, 29, 3, 3, 29, 29, 29, 4, 1]
    components = {
        key: np.full((2, dim), index, dtype=np.float32)
        for index, (key, dim) in enumerate(zip(bundle.observation_keys, dims))
    }
    session = FakeSession(
        {"continuous_actions": np.zeros((2, 29), dtype=np.float32)}
    )
    bundle._sessions = {"policy": session}

    output = bundle.forward(components)

    assert session.last_feeds["obs"].shape == (2, 156)
    np.testing.assert_array_equal(session.last_feeds["obs"][:, :29], 0.0)
    np.testing.assert_array_equal(session.last_feeds["obs"][:, 29:58], 1.0)
    assert output["continuous_actions"].shape == (2, 29)


def test_any2track_rejects_a_mismatched_training_contract(tmp_path):
    files = _touch_files(tmp_path, dict(Any2TrackBundle.DEFAULT_FILES))
    config = _any2track_training_config()
    config["policy_config"]["policy_args"]["obs_dim"] = 155
    files["training_config"].write_text(json.dumps(config))

    try:
        Any2TrackBundle(file_paths=files, load_model=False)
    except ValueError as error:
        assert "obs_dim=156" in str(error)
    else:
        raise AssertionError("Any2Track must reject a mismatched training config")


def test_beyondmimic_keeps_time_step_and_embedded_reference_outputs(tmp_path):
    files = _touch_files(tmp_path, dict(BeyondMimicBundle.DEFAULT_FILES))
    bundle = BeyondMimicBundle(file_paths=files, load_model=False)
    outputs = {
        "actions": np.zeros((1, 29), dtype=np.float32),
        "joint_pos": np.zeros((1, 29), dtype=np.float32),
        "joint_vel": np.zeros((1, 29), dtype=np.float32),
        "body_pos_w": np.zeros((1, 13, 3), dtype=np.float32),
        "body_quat_w": np.zeros((1, 13, 4), dtype=np.float32),
        "body_lin_vel_w": np.zeros((1, 13, 3), dtype=np.float32),
        "body_ang_vel_w": np.zeros((1, 13, 3), dtype=np.float32),
    }
    session = FakeSession(outputs, metadata={"joint_names": "a,b"})
    bundle._sessions = {"policy": session}

    result = bundle.forward(np.zeros(154, dtype=np.float32), time_step=7)

    np.testing.assert_array_equal(
        session.last_feeds["time_step"], np.asarray([[7]], dtype=np.float32)
    )
    assert set(outputs) <= set(result)
    assert bundle.policy_metadata == {"joint_names": "a,b"}


def test_humanoid_gpt_preserves_official_observation_and_action_contract(tmp_path):
    files = _touch_files(tmp_path, dict(HumanoidGPTBundle.DEFAULT_FILES))
    bundle = HumanoidGPTBundle(file_paths=files, load_model=False)
    action = np.ones((2, 29), dtype=np.float32)
    session = FakeSession(
        {
            "continuous_actions": action,
            "std_param": np.zeros(29, dtype=np.float32),
        }
    )
    bundle._sessions = {"policy": session}
    components = {
        name: np.full((2, dimension), index, dtype=np.float32)
        for index, (name, dimension) in enumerate(bundle.OBSERVATION_LAYOUT)
    }

    result = bundle.forward(components=components)

    assert session.last_feeds["obs"].shape == (2, 136)
    np.testing.assert_array_equal(session.last_feeds["obs"][:, :3], 0.0)
    np.testing.assert_array_equal(session.last_feeds["obs"][:, 3:6], 1.0)
    np.testing.assert_allclose(
        result["motor_targets"],
        bundle.decode_motor_targets(action),
    )


def test_humanoid_gpt_artifact_copies_complete_robot_assets(tmp_path):
    files = _touch_files(tmp_path / "source", dict(HumanoidGPTBundle.DEFAULT_FILES))
    mesh = files["scene"].parent / "assets" / "pelvis.STL"
    mesh.parent.mkdir(parents=True, exist_ok=True)
    mesh.write_bytes(b"mesh")
    source = HumanoidGPTBundle(file_paths=files, load_model=False)
    artifact = tmp_path / "artifact"

    source.save_pretrained(artifact)
    pipe = Pipeline.from_pretrained(
        artifact,
        bundle_kwargs={"load_model": False},
    )

    assert pipe.__class__.__name__ == "HumanoidGPTPipeline"
    assert (artifact / "unitree_g1_5010/assets/pelvis.STL").is_file()
    model_index = json.loads((artifact / "model_index.json").read_text())
    assert "unitree_g1_5010/assets/pelvis.STL" in model_index["required_files"]


def test_motius_artifact_loads_through_pipeline_facade(tmp_path):
    files = _touch_files(tmp_path / "source", dict(Any2TrackBundle.DEFAULT_FILES))
    files["training_config"].write_text(json.dumps(_any2track_training_config()))
    source = Any2TrackBundle(file_paths=files, load_model=False)
    artifact = tmp_path / "artifact"
    source.save_pretrained(artifact)

    pipe = Pipeline.from_pretrained(
        artifact,
        bundle_kwargs={"load_model": False},
    )

    assert pipe.__class__.__name__ == "Any2TrackPipeline"
    assert pipe.bundle.observation_keys[0] == "dif_joint_pos"
    model_index = json.loads((artifact / "model_index.json").read_text())
    assert model_index["tasks"] == ["motion_tracking"]
