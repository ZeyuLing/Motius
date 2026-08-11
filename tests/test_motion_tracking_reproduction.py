from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from tools.verify_motion_tracking_reproduction import (
    TRAINER_STAGES,
    _identity_quaternion,
    verify_inventory,
    verify_replay,
    verify_rollouts,
    verify_trainer_evidence,
    verify_viewer,
)


def test_identity_quaternion_uses_the_declared_xyzw_convention():
    value = np.zeros((2, 4), dtype=np.float32)

    _identity_quaternion(value)

    np.testing.assert_array_equal(
        value,
        np.asarray([[0.0, 0.0, 0.0, 1.0]] * 2, dtype=np.float32),
    )


def test_inventory_rejects_modified_artifact(tmp_path: Path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"official")
    inventory = {
        "schema_version": 1,
        "files": {
            "model.onnx": {
                "bytes": len(b"official"),
                "sha256": hashlib.sha256(b"official").hexdigest(),
            }
        },
    }
    (tmp_path / "artifact_inventory.json").write_text(json.dumps(inventory))

    assert verify_inventory(tmp_path)["status"] == "pass"
    model.write_bytes(b"modified")
    assert verify_inventory(tmp_path)["status"] == "fail"


def _write_rollout(
    path: Path,
    case_id: str,
    *,
    action_offset: float = 0.0,
    metric_scale: float = 1.0,
) -> None:
    qpos = np.zeros((3, 36), dtype=np.float32)
    qpos[:, 3] = 1.0
    actions = np.zeros((2, 29), dtype=np.float32)
    actions[0, 0] = action_offset
    np.savez_compressed(
        path,
        qpos=qpos,
        reference_qpos=qpos,
        actions=actions,
        method=np.asarray("Any2Track"),
        backend=np.asarray("mujoco"),
        protocol_id=np.asarray("test"),
        metrics_json=np.asarray(
            json.dumps(
                {
                    "joint_position_mae_rad": 0.1 * metric_scale,
                    "success_rate": 1.0,
                    "survival_rate": 1.0,
                }
            )
        ),
    )


def test_rollout_and_viewer_must_use_the_same_cases(tmp_path: Path):
    rollout_root = tmp_path / "any2track"
    (rollout_root / "rollouts").mkdir(parents=True)
    artifact = rollout_root / "rollouts/walk__f000000_000002.npz"
    _write_rollout(artifact, "walk")
    result = {
        "method": "Any2Track",
        "protocol_id": "test",
        "cases": [
            {
                "id": "walk__f000000_000002",
                "artifact": str(artifact),
            }
        ],
    }
    result_path = rollout_root / "results.json"
    result_path.write_text(json.dumps(result))
    report = verify_rollouts("any2track", result_path)
    assert report["status"] == "pass"

    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "columns": [{"key": "any2track"}],
                "cases": [
                    {
                        "case_id": "walk",
                        "assets": {"any2track": {"path": "walk"}},
                    }
                ],
            }
        )
    )
    assert verify_viewer(manifest_path, {"any2track": report})["status"] == "pass"

    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][0]["case_id"] = "run"
    manifest_path.write_text(json.dumps(manifest))
    assert verify_viewer(manifest_path, {"any2track": report})["status"] == "fail"


def test_rollout_method_names_allow_display_and_registry_spelling(tmp_path: Path):
    rollout_root = tmp_path / "humanoid_gpt"
    (rollout_root / "rollouts").mkdir(parents=True)
    artifact = rollout_root / "rollouts/walk.npz"
    _write_rollout(artifact, "walk")
    result_path = rollout_root / "results.json"
    result_path.write_text(
        json.dumps(
            {
                "method": "HumanoidGPT",
                "protocol_id": "test",
                "cases": [{"id": "walk", "artifact": str(artifact)}],
            }
        )
    )

    assert verify_rollouts("humanoid_gpt", result_path)["status"] == "pass"


def test_replay_allows_only_numerical_closed_loop_drift(tmp_path: Path):
    baseline = tmp_path / "baseline.npz"
    replay = tmp_path / "replay.npz"
    _write_rollout(baseline, "walk")
    _write_rollout(replay, "walk", action_offset=5e-8, metric_scale=1.04)

    report = verify_replay("any2track", replay, baseline)
    assert report["status"] == "pass"
    assert report["first_bitwise_action_difference"]["step"] == 0

    _write_rollout(replay, "walk", action_offset=2e-6, metric_scale=1.04)
    assert verify_replay("any2track", replay, baseline)["status"] == "fail"

    _write_rollout(replay, "walk", action_offset=5e-8, metric_scale=1.06)
    assert verify_replay("any2track", replay, baseline)["status"] == "fail"


def test_trainer_evidence_requires_the_complete_train_export_chain(tmp_path: Path):
    stages = {name: {"status": "pass"} for name in TRAINER_STAGES}
    stages["parameter_update"].update(
        {"before_sha256": "before", "after_sha256": "after"}
    )
    path = tmp_path / "trainer.json"
    path.write_text(json.dumps({"method": "sonic", "stages": stages}))

    assert verify_trainer_evidence("sonic", path)["status"] == "pass"
    del stages["resume"]
    path.write_text(json.dumps({"method": "sonic", "stages": stages}))
    assert verify_trainer_evidence("sonic", path)["status"] == "fail"
