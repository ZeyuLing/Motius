from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


evaluator = _load_module(
    "gentrack_unified_evaluator",
    "tools/gentrack/evaluate.py",
)
materializer = _load_module(
    "materialize_canonical_tracker_cases",
    "tools/gentrack/materialize_cases.py",
)


def _write_case(
    path: Path,
    execution: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    unexpected_fall: bool | None = None,
    with_root_quaternions: bool = True,
    root_quaternions: tuple[np.ndarray, np.ndarray] | None = None,
    qpos: tuple[np.ndarray, np.ndarray] | None = None,
    completion: float | None = None,
    terminated: bool | None = None,
    source_path: str | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    if reference is None:
        reference = np.zeros_like(execution)
    payload = {
        "reference_joints": reference,
        "execution_joints": execution,
        "joint_names": np.asarray(evaluator.CANONICAL_G1_BODY_NAMES),
        "fps": np.float32(30.0),
        "case_id": np.asarray(path.stem),
    }
    if unexpected_fall is not None:
        payload["unexpected_fall"] = np.asarray(unexpected_fall)
    if with_root_quaternions:
        quaternions = np.zeros((*execution.shape[:2], 4), dtype=np.float32)
        quaternions[..., 0] = 1.0
        payload["reference_body_quat"] = quaternions
        payload["execution_body_quat"] = quaternions.copy()
        payload["body_names"] = np.asarray(evaluator.CANONICAL_G1_BODY_NAMES)
    if root_quaternions is not None:
        reference_quat = np.zeros((*execution.shape[:2], 4), dtype=np.float32)
        execution_quat = np.zeros((*execution.shape[:2], 4), dtype=np.float32)
        reference_quat[..., 0] = 1.0
        execution_quat[..., 0] = 1.0
        reference_quat[:, 0] = root_quaternions[0]
        execution_quat[:, 0] = root_quaternions[1]
        payload["reference_body_quat"] = reference_quat
        payload["execution_body_quat"] = execution_quat
        payload["body_names"] = np.asarray(evaluator.CANONICAL_G1_BODY_NAMES)
    if qpos is not None:
        payload["reference_qpos"] = qpos[0]
        payload["execution_qpos"] = qpos[1]
    if completion is not None:
        payload["completion"] = np.asarray(completion, dtype=np.float32)
    if terminated is not None:
        payload["terminated"] = np.asarray(terminated)
    if source_path is not None:
        payload["source_path"] = np.asarray(source_path)
    if metadata is not None:
        payload["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    np.savez_compressed(path, **payload)


def test_sonic_paper_success_is_the_default_tracker_protocol(tmp_path: Path) -> None:
    execution = np.zeros((4, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    path = tmp_path / "success.npz"
    _write_case(path, execution)

    row = evaluator._evaluate_case(path).row

    assert row["success_unified"] is True
    assert row["success_protocol"] == "sonic_paper_mujoco_fall_only_30fps_full_export"
    assert row["failure_reasons"] == []
    assert row["unexpected_fall"] is False


def test_native_termination_is_diagnostic_for_unified_tracker_sr(tmp_path: Path) -> None:
    execution = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    path = tmp_path / "terminated_prefix.npz"
    _write_case(path, execution, completion=0.10, terminated=True)

    row = evaluator._evaluate_case(path).row

    assert row["success_unified"] is True
    assert row["completion"] == 1.0
    assert row["completion_source"] == "trajectory_length"
    assert row["native_completion"] == np.float32(0.10)
    assert row["native_terminated"] is True
    assert row["native_rollout_status_used"] is False


def test_native_termination_requires_explicit_opt_in(tmp_path: Path) -> None:
    execution = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    path = tmp_path / "native_termination.npz"
    _write_case(path, execution, completion=0.10, terminated=True)

    row = evaluator._evaluate_case(path, use_native_rollout_status=True).row

    assert row["success_unified"] is False
    assert row["completion"] == np.float32(0.10)
    assert row["completion_source"] == "native_simulator_progress"
    assert row["terminated"] is True
    assert row["failure_reasons"] == ["native_rollout_failure"]


def test_canonical_eval_uses_external_termination_for_padded_rollout(tmp_path: Path) -> None:
    body_count = len(evaluator.CANONICAL_G1_BODY_NAMES)
    body_pos = np.zeros((30, body_count, 3), dtype=np.float32)
    body_quat = np.zeros((30, body_count, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    reference_dir = tmp_path / "reference"
    execution_dir = tmp_path / "execution"
    reference_dir.mkdir()
    execution_dir.mkdir()
    for directory in (reference_dir, execution_dir):
        np.savez_compressed(
            directory / "case_a.npz",
            body_pos=body_pos,
            body_quat=body_quat,
            body_names=np.asarray(evaluator.CANONICAL_G1_BODY_NAMES),
        )

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(["case_a"]))
    rollout_summary = tmp_path / "rollout_summary.json"
    rollout_summary.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "case_id": "case_a",
                        "progress": 0.25,
                        "terminated": True,
                    }
                ]
            }
        )
    )

    summary = evaluator._evaluate_canonical_dirs(
        reference_dir,
        execution_dir,
        manifest,
        tmp_path / "out",
        method="sonic",
        split="test",
        fps=30.0,
        workers=1,
        rollout_summary_path=rollout_summary,
        use_native_rollout_status=True,
    )
    row = json.loads((tmp_path / "out" / "case_metrics.jsonl").read_text())

    assert summary["final_table_eligible"] is True
    assert summary["success_rate_unified"] == 0.0
    assert row["completion"] == 0.25
    assert row["completion_source"] == "native_simulator_progress"
    assert row["terminated"] is True
    assert row["failure_reasons"] == ["native_rollout_failure"]


def test_canonical_eval_accepts_canonical_amass_source_and_rich_manifest(
    tmp_path: Path,
) -> None:
    body_count = len(evaluator.CANONICAL_G1_BODY_NAMES)
    body_pos = np.zeros((4, body_count, 3), dtype=np.float32)
    body_quat = np.zeros((4, body_count, 4), dtype=np.float32)
    body_quat[..., 0] = 1.0
    reference_dir = tmp_path / "reference"
    execution_dir = tmp_path / "execution"
    reference_dir.mkdir()
    execution_dir.mkdir()
    for directory in (reference_dir, execution_dir):
        np.savez_compressed(
            directory / "case_a.npz",
            body_pos=body_pos,
            body_quat=body_quat,
            body_names=np.asarray(evaluator.CANONICAL_G1_BODY_NAMES),
            source=np.asarray("data/AMASS_GMR_for_G1/g1/CMU/case_a.npz"),
        )
    manifest = (
        tmp_path
        / "outputs/evaluation/gentrack/table2_tracker/unified_protocol_v1"
        / "inputs/amass_test_fixed600/manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({"method": "test", "rows": [{"case_id": "case_a"}]}))

    summary = evaluator._evaluate_canonical_dirs(
        reference_dir,
        execution_dir,
        manifest,
        tmp_path / "out",
        method="test",
        split="amass_test_g1",
        fps=30.0,
        workers=1,
    )
    row = json.loads((tmp_path / "out" / "case_metrics.jsonl").read_text())

    assert summary["final_table_eligible"] is True
    assert summary["num_cases"] == 1
    assert row["source_path"] == "data/AMASS_GMR_for_G1/g1/CMU/case_a.npz"
    assert (
        json.loads(
            (tmp_path / "out" / "reference_input_manifest.json").read_text()
        )["rows"][0]["source_path"]
        == row["source_path"]
    )


def test_any2track_mean_link_error_remains_an_explicit_diagnostic_protocol(tmp_path: Path) -> None:
    execution = np.zeros((4, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    execution[:, 1:, 0] = 0.25
    path = tmp_path / "link_error.npz"
    _write_case(path, execution)

    row = evaluator._evaluate_case(path, success_protocol=evaluator.SUCCESS_PROTOCOL_ANY2TRACK).row

    assert row["success_unified"] is False
    assert row["failure_reasons"] == ["mean_link_position_error"]


def test_any2track_mean_root_height_error_remains_an_explicit_diagnostic_protocol(tmp_path: Path) -> None:
    execution = np.zeros((4, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    execution[:, :, 2] = 0.21
    path = tmp_path / "root_height_error.npz"
    _write_case(path, execution)

    row = evaluator._evaluate_case(path, success_protocol=evaluator.SUCCESS_PROTOCOL_ANY2TRACK).row

    assert row["success_unified"] is False
    assert row["failure_reasons"] == ["mean_root_height_error"]


def test_sonic_paper_protocol_uses_max_reference_relative_root_height_error(tmp_path: Path) -> None:
    reference = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    reference[..., 2] = 0.90
    execution = reference.copy()
    execution[10, 0, 2] -= 0.26
    path = tmp_path / "root_height_termination.npz"
    _write_case(path, execution, reference=reference)

    row = evaluator._evaluate_case(path).row

    assert row["success_unified"] is False
    assert row["failure_reasons"] == ["root_height_termination"]
    assert np.isclose(row["max_root_height_err_m"], 0.26)


def test_sonic_paper_protocol_does_not_use_end_effector_height(tmp_path: Path) -> None:
    reference = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    execution = reference.copy()
    wrist = list(evaluator.CANONICAL_G1_BODY_NAMES).index("left_wrist_yaw_link")
    execution[10, wrist, 2] += 0.26
    path = tmp_path / "end_effector_height_termination.npz"
    _write_case(path, execution, reference=reference)

    row = evaluator._evaluate_case(path).row

    assert row["success_unified"] is True
    assert row["failure_reasons"] == []
    assert row["failure_reasons_sonic_release_eval"] == [
        "end_effector_height_termination"
    ]
    assert np.isclose(row["max_end_effector_height_err_m"], 0.26)


def test_sonic_release_eval_protocol_uses_reference_relative_end_effector_height(
    tmp_path: Path,
) -> None:
    reference = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    execution = reference.copy()
    wrist = list(evaluator.CANONICAL_G1_BODY_NAMES).index("left_wrist_yaw_link")
    execution[10, wrist, 2] += 0.26
    path = tmp_path / "release_end_effector_height_termination.npz"
    _write_case(path, execution, reference=reference)

    row = evaluator._evaluate_case(
        path,
        success_protocol=evaluator.SUCCESS_PROTOCOL_SONIC_RELEASE_EVAL,
    ).row

    assert row["success_unified"] is False
    assert row["failure_reasons"] == ["end_effector_height_termination"]


def test_sonic_release_eval_protocol_uses_pelvis_orientation_error(tmp_path: Path) -> None:
    body_count = len(evaluator.CANONICAL_G1_BODY_NAMES)
    reference = np.zeros((30, body_count, 3), dtype=np.float32)
    execution = reference.copy()
    ref_quat = np.zeros((30, 4), dtype=np.float32)
    ref_quat[:, 0] = 1.0
    exe_quat = ref_quat.copy()
    angle = 1.01
    exe_quat[10, 0] = np.cos(angle / 2.0)
    exe_quat[10, 1] = np.sin(angle / 2.0)
    path = tmp_path / "root_orientation_termination.npz"
    _write_case(path, execution, reference=reference, root_quaternions=(ref_quat, exe_quat))

    row = evaluator._evaluate_case(
        path,
        success_protocol=evaluator.SUCCESS_PROTOCOL_SONIC_RELEASE_EVAL,
    ).row

    assert row["success_unified"] is False
    assert row["failure_reasons"] == ["root_orientation_termination"]
    assert np.isclose(row["max_root_orientation_err_rad"], angle, atol=1e-5)


def test_sonic_paper_protocol_accepts_position_only_export(tmp_path: Path) -> None:
    execution = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    path = tmp_path / "missing_orientation.npz"
    _write_case(path, execution, with_root_quaternions=False)

    row = evaluator._evaluate_case(path).row

    assert row["success_unified"] is True
    assert row["max_root_orientation_err_rad"] is None


def test_sonic_release_eval_protocol_refuses_position_only_approximation(tmp_path: Path) -> None:
    execution = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    path = tmp_path / "missing_release_orientation.npz"
    _write_case(path, execution, with_root_quaternions=False)

    with np.testing.assert_raises_regex(ValueError, "requires reference/execution pelvis quaternions"):
        evaluator._evaluate_case(
            path,
            success_protocol=evaluator.SUCCESS_PROTOCOL_SONIC_RELEASE_EVAL,
        )


def test_sonic_paper_protocol_tolerates_one_resampling_endpoint_frame(tmp_path: Path) -> None:
    body_count = len(evaluator.CANONICAL_G1_BODY_NAMES)
    reference = np.zeros((30, body_count, 3), dtype=np.float32)
    execution = np.zeros((29, body_count, 3), dtype=np.float32)
    path = tmp_path / "one_endpoint_short.npz"
    _write_case(path, execution, reference=reference)

    row = evaluator._evaluate_case(path).row

    assert row["completion"] == 29 / 30
    assert row["full_export_with_endpoint_tolerance"] is True
    assert row["success_unified"] is True


def test_sonic_paper_protocol_rejects_more_than_one_missing_export_frame(tmp_path: Path) -> None:
    body_count = len(evaluator.CANONICAL_G1_BODY_NAMES)
    reference = np.zeros((30, body_count, 3), dtype=np.float32)
    execution = np.zeros((28, body_count, 3), dtype=np.float32)
    path = tmp_path / "two_frames_short.npz"
    _write_case(path, execution, reference=reference)

    row = evaluator._evaluate_case(path).row

    assert row["full_export_with_endpoint_tolerance"] is False
    assert row["success_unified"] is False
    assert row["failure_reasons"] == ["incomplete_export"]


def test_missing_fall_export_does_not_block_main_table(tmp_path: Path) -> None:
    cases_dir = tmp_path / "cases"
    cases_dir.mkdir()
    execution = np.zeros((4, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    _write_case(cases_dir / "missing_fall.npz", execution)

    summary = evaluator._evaluate_dir(cases_dir, tmp_path / "out", allow_joint_crop=False)

    assert summary["final_table_eligible"] is True
    assert summary["unexpected_fall_rate"] == 0.0
    metric_row = json.loads((tmp_path / "out" / "case_metrics.jsonl").read_text())
    for role in ("reference", "execution"):
        manifest_path = tmp_path / "out" / summary[f"{role}_manifest"]
        assert manifest_path.is_file()
        assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == summary[
            f"{role}_manifest_sha256"
        ]
        manifest = json.loads(manifest_path.read_text())
        assert manifest["schema"] == evaluator.INPUT_MANIFEST_SCHEMA
        assert manifest["role"] == role
        assert manifest["rows"][0]["sha256"] == metric_row[f"{role}_input_sha256"]


def test_paper_provenance_rejects_legacy_retarged_amass_source(tmp_path: Path) -> None:
    execution = np.zeros((4, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    path = tmp_path / "legacy_source.npz"
    _write_case(
        path,
        execution,
        metadata={
            "reference_metadata": {
                "source": "data/AMASS_Retarged_for_G1/ACCAD/example.npz"
            }
        },
    )

    with np.testing.assert_raises_regex(ValueError, "forbidden legacy AMASS source"):
        evaluator._evaluate_case(path)


def test_paper_provenance_rejects_legacy_gentrack_july9_manifest_path(
    tmp_path: Path,
) -> None:
    manifest = (
        tmp_path
        / "outputs/evaluation/physflow/gentrack_aaai2027/table_tracker/unified_protocol_v1"
        / "inputs/amass_test_g1/manifest.json"
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text("[]\n")

    with np.testing.assert_raises_regex(ValueError, "legacy GenTrack protocol"):
        evaluator._evaluate_canonical_dirs(
            tmp_path / "reference",
            tmp_path / "execution",
            manifest,
            tmp_path / "out",
            method="test",
            split="amass_test_g1",
            fps=30.0,
            workers=1,
        )


def test_low_floor_reference_is_not_an_unexpected_fall(tmp_path: Path) -> None:
    reference = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    reference[..., 2] = 0.20
    execution = reference.copy()
    path = tmp_path / "low_floor.npz"
    _write_case(
        path,
        execution,
        reference=reference,
        with_root_quaternions=True,
    )

    row = evaluator._evaluate_case(path).row

    assert row["unexpected_fall"] is False
    assert row["unexpected_fall_source"] == "reference_conditioned_root_drop_or_tilt"


def test_persistent_reference_relative_collapse_is_an_unexpected_fall(tmp_path: Path) -> None:
    reference = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    reference[..., 2] = 0.90
    execution = reference.copy()
    execution[8:, :, 2] -= 0.40
    path = tmp_path / "collapse.npz"
    _write_case(
        path,
        execution,
        reference=reference,
        with_root_quaternions=True,
    )

    row = evaluator._evaluate_case(path).row

    assert row["unexpected_fall"] is True
    assert row["unexpected_fall_longest_run_frames"] == 22
    assert row["unexpected_fall_persistence_frames"] == 6


def test_transient_reference_relative_dip_is_not_an_unexpected_fall(tmp_path: Path) -> None:
    reference = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    reference[..., 2] = 0.90
    execution = reference.copy()
    execution[8:11, :, 2] -= 0.40
    path = tmp_path / "transient.npz"
    _write_case(
        path,
        execution,
        reference=reference,
        with_root_quaternions=True,
    )

    row = evaluator._evaluate_case(path).row

    assert row["unexpected_fall"] is False
    assert row["unexpected_fall_longest_run_frames"] == 3


def test_persistent_yaw_error_is_not_an_unexpected_fall(tmp_path: Path) -> None:
    reference = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    reference[..., 2] = 0.90
    execution = reference.copy()
    reference_root = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (30, 1))
    execution_root = np.tile(
        np.array([np.cos(np.pi / 4.0), 0.0, 0.0, np.sin(np.pi / 4.0)], dtype=np.float32),
        (30, 1),
    )
    path = tmp_path / "yaw_only.npz"
    _write_case(
        path,
        execution,
        reference=reference,
        root_quaternions=(reference_root, execution_root),
    )

    row = evaluator._evaluate_case(path).row

    assert row["unexpected_fall"] is False
    assert row["unexpected_fall_longest_run_frames"] == 0


def test_persistent_pelvis_tilt_is_an_unexpected_fall(tmp_path: Path) -> None:
    reference = np.zeros((30, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    reference[..., 2] = 0.90
    execution = reference.copy()
    reference_root = np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (30, 1))
    execution_root = np.tile(
        np.array([np.cos(0.60), np.sin(0.60), 0.0, 0.0], dtype=np.float32),
        (30, 1),
    )
    path = tmp_path / "tilted.npz"
    _write_case(
        path,
        execution,
        reference=reference,
        root_quaternions=(reference_root, execution_root),
    )

    row = evaluator._evaluate_case(path).row

    assert row["unexpected_fall"] is True
    assert row["unexpected_fall_longest_run_frames"] == 30


def test_rlpf_low_level_errors_use_wrapped_dofs_and_fixed_14_bodies(tmp_path: Path) -> None:
    reference = np.zeros((10, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    execution = reference.copy()
    execution[:, evaluator.CANONICAL_G1_BODY_NAMES.index("left_hip_pitch_link"), 0] = 1.0
    execution[:, evaluator.CANONICAL_G1_BODY_NAMES.index("left_knee_link"), 0] = 0.14

    reference_qpos = np.zeros((10, 36), dtype=np.float32)
    execution_qpos = reference_qpos.copy()
    reference_qpos[:, 3] = 1.0
    execution_qpos[:, 3] = 1.0
    execution_qpos[:, 7] = 0.29
    path = tmp_path / "rlpf_metrics.npz"
    _write_case(
        path,
        execution,
        reference=reference,
        qpos=(reference_qpos, execution_qpos),
    )

    row = evaluator._evaluate_case(path).row

    assert np.isclose(row["e_joint_rad"], 0.01, atol=1e-6)
    assert np.isclose(row["e_key_m"], 0.01, atol=1e-6)


def test_paper_mpjpe_uses_sonic_root_relative_14_body_set(tmp_path: Path) -> None:
    reference = np.zeros((4, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    pelvis_translation = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 0.3], [2.0, 0.5, -0.1], [-1.0, 1.0, 0.2]],
        dtype=np.float32,
    )
    execution = reference + pelvis_translation[:, None, :]
    # This canonical body is intentionally outside SONIC's 14-body metric set.
    execution[:, evaluator.CANONICAL_G1_BODY_NAMES.index("left_hip_pitch_link"), 0] += 1.0
    # One 14-body link contributes 0.14 / 14 = 0.01 m to MPJPE-L.
    execution[:, evaluator.CANONICAL_G1_BODY_NAMES.index("left_knee_link"), 0] += 0.14
    path = tmp_path / "sonic_14_body_mpjpe.npz"
    _write_case(path, execution, reference=reference)

    row = evaluator._evaluate_case(path).row

    assert np.isclose(row["mpjpe_mm"], 10.0, atol=1e-5)
    assert row["er_mpjpe_mm"] > row["mpjpe_mm"]
    assert row["mpjpe_body_count"] == 14
    assert row["mpjpe_body_names"] == list(evaluator.SONIC_KEY_BODY_NAMES)
    assert row["mpjpe_protocol"] == evaluator.ROW_MPJPE_PROTOCOL


def test_paper_eg_uses_same_14_bodies_start_xy_only_and_keeps_z(tmp_path: Path) -> None:
    reference = np.zeros((4, len(evaluator.CANONICAL_G1_BODY_NAMES), 3), dtype=np.float32)
    execution = reference.copy()
    execution[..., 0] += 1.5
    execution[..., 1] -= 0.5
    execution[..., 2] += 0.20
    # A non-SONIC body must not affect E_g.
    execution[:, evaluator.CANONICAL_G1_BODY_NAMES.index("left_hip_pitch_link"), 2] += 1.0
    path = tmp_path / "sonic_14_body_eg.npz"
    _write_case(path, execution, reference=reference)

    row = evaluator._evaluate_case(path).row

    assert np.isclose(row["eg_mpjpe_mm"], 200.0, atol=1e-4)
    assert np.isclose(row["e_key_m"], 0.20, atol=1e-6)
    assert np.isclose(row["mpjpe_mm"], 0.0, atol=1e-5)
    assert row["eg_protocol"] == evaluator.ROW_EG_PROTOCOL
    assert row["eg_body_count"] == 14


def test_success_only_summary_is_weighted_by_valid_frames() -> None:
    rows = [
        {
            "success_unified": True,
            "num_eval_frames": 2,
            "mpjpe_mm": 10.0,
            "mpjve_mps": 1.0,
            "root_vel_err_mps": 2.0,
            "root_traj_err_m": 0.1,
            "eg_mpjpe_mm": 10.0,
            "er_mpjpe_mm": 20.0,
            "evel_mps": 1.0,
            "eacc_mps2": 2.0,
            "failure_reasons": [],
        },
        {
            "success_unified": True,
            "num_eval_frames": 4,
            "mpjpe_mm": 40.0,
            "mpjve_mps": 4.0,
            "root_vel_err_mps": 8.0,
            "root_traj_err_m": 0.4,
            "eg_mpjpe_mm": 40.0,
            "er_mpjpe_mm": 50.0,
            "evel_mps": 4.0,
            "eacc_mps2": 8.0,
            "failure_reasons": [],
        },
    ]

    summary = evaluator._summary_from_rows(rows, table_eligible=True, warnings=[])

    assert np.isclose(summary["mpjpe_mm_success_only"], 30.0)
    assert np.isclose(summary["mpjve_mps_success_only"], 3.25)
    assert np.isclose(summary["root_vel_err_mps_success_only"], 6.5)
    assert np.isclose(summary["eg_mpjpe_mm_success_only"], 30.0)
    assert np.isclose(summary["er_mpjpe_mm_success_only"], 40.0)
    assert np.isclose(summary["evel_mps_success_only"], 3.25)
    assert np.isclose(summary["eacc_mps2_success_only"], 8.0)
    assert summary["success_only_is_diagnostic"] is True
    assert summary["mpjpe_body_count"] == 14


def test_main_table_errors_use_all_trajectory_frame_micro_average() -> None:
    rows = [
        {
            "success_unified": True,
            "num_eval_frames": 2,
            "mpjpe_mm": 10.0,
            "mpjve_mps": 1.0,
            "root_vel_err_mps": 2.0,
            "root_traj_err_m": 0.1,
            "eg_mpjpe_mm": 10.0,
            "er_mpjpe_mm": 20.0,
            "evel_mps": 1.0,
            "eacc_mps2": 2.0,
            "failure_reasons": [],
        },
        {
            "success_unified": False,
            "num_eval_frames": 4,
            "mpjpe_mm": 40.0,
            "mpjve_mps": 4.0,
            "root_vel_err_mps": 8.0,
            "root_traj_err_m": 0.4,
            "eg_mpjpe_mm": 40.0,
            "er_mpjpe_mm": 50.0,
            "evel_mps": 4.0,
            "eacc_mps2": 8.0,
            "failure_reasons": ["root_height_termination"],
        },
    ]

    summary = evaluator._summary_from_rows(rows, table_eligible=True, warnings=[])

    assert np.isclose(summary["mpjpe_mm"], 30.0)
    assert np.isclose(summary["mpjve_mps"], 3.25)
    assert np.isclose(summary["root_vel_err_mps"], 6.5)
    assert np.isclose(summary["root_traj_err_m"], 0.3)
    assert np.isclose(summary["eg_mpjpe_mm"], 30.0)
    assert np.isclose(summary["er_mpjpe_mm"], 40.0)
    assert np.isclose(summary["evel_mps"], 3.25)
    assert np.isclose(summary["eacc_mps2"], 8.0)
    assert np.isclose(summary["mpjpe_mm_success_only"], 10.0)
    assert np.isclose(summary["eg_mpjpe_mm_success_only"], 10.0)
    assert summary["eacc_mps2_success_only"] is None
    assert summary["paper_metric_source"] == "all_cases"
    assert summary["paper_continuous_aggregation"] == evaluator.PAPER_CONTINUOUS_AGGREGATION
    assert (
        summary["mpjpe_protocol"]
        == evaluator.PAPER_MPJPE_PROTOCOL
    )


def test_materializer_uses_fixed_30_link_body_set() -> None:
    canonical = list(materializer.CANONICAL_G1_BODY_NAMES)
    ref_names = ["head", *canonical, "left_rubber_hand", "right_rubber_hand"]
    exe_names = list(reversed(ref_names))

    def payload(names: list[str]) -> dict[str, np.ndarray]:
        positions = np.zeros((2, len(names), 3), dtype=np.float32)
        for index in range(len(names)):
            positions[:, index, 0] = index
        quaternions = np.zeros((2, len(names), 4), dtype=np.float32)
        return {
            "body_names": np.asarray(names),
            "body_pos": positions,
            "body_quat": quaternions,
        }

    ref_pos, exe_pos, _, _, names, report = materializer._align_body_arrays(
        payload(ref_names),
        payload(exe_names),
    )

    assert names == canonical
    assert ref_pos.shape[1] == exe_pos.shape[1] == 30
    assert report["canonical_body_set"] is True
    assert set(report["dropped_reference_bodies"]) == {
        "head",
        "left_rubber_hand",
        "right_rubber_hand",
    }
