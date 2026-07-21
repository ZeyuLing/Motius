from __future__ import annotations

import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "materialize_motion272_joints.py"
SPEC = importlib.util.spec_from_file_location("materialize_motion272_joints", TOOL_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

HML_TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "materialize_hml263_smpl_joints.py"
)
HML_SPEC = importlib.util.spec_from_file_location(
    "materialize_hml263_smpl_joints", HML_TOOL_PATH
)
assert HML_SPEC and HML_SPEC.loader
HML_MODULE = importlib.util.module_from_spec(HML_SPEC)
HML_SPEC.loader.exec_module(HML_MODULE)


def _good_mesh_integrity() -> dict[str, float]:
    return {
        "rotation_jump_deg_p99": 20.0,
        "rotation_jump_deg_max": 35.0,
        "mesh_edge_ratio_p01": 0.7,
        "mesh_edge_ratio_p99": 1.3,
        "mesh_edge_ratio_max": 4.0,
        "mesh_sample_count": 7.0,
    }


def test_hml263_ids_loader_defers_lengths_and_reads_only_its_shard(
    tmp_path: Path, monkeypatch
) -> None:
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("a\nb\nc\nd\n")
    loaded = []

    def fail_if_loaded(*args, **kwargs):
        loaded.append((args, kwargs))
        raise AssertionError("The ids loader must not open motion arrays")

    monkeypatch.setattr(HML_MODULE.np, "load", fail_if_loaded)
    args = Namespace(
        manifest=None,
        ids_file=ids_file,
        shard_index=1,
        num_shards=2,
    )

    _protocol, cases = HML_MODULE._load_cases(args, tmp_path)

    assert loaded == []
    assert [case["total_frames"] for case in cases] == [0, None, 0, None]


def test_load_motion272_supports_npy_and_npz(tmp_path: Path) -> None:
    motion = np.zeros((7, 272), dtype=np.float32)
    np.save(tmp_path / "case.npy", motion)
    np.savez_compressed(tmp_path / "case.npz", motion_272=motion)

    assert MODULE._load_motion272(tmp_path / "case.npy", "motion_272").shape == (7, 272)
    assert MODULE._load_motion272(tmp_path / "case.npz", "motion_272").shape == (7, 272)


def test_load_motion272_rejects_wrong_shape(tmp_path: Path) -> None:
    np.save(tmp_path / "bad.npy", np.zeros((7, 263), dtype=np.float32))
    with pytest.raises(ValueError, match=r"Expected \[T, 272\]"):
        MODULE._load_motion272(tmp_path / "bad.npy", "motion_272")


def test_hml263_materializer_preserves_positions_via_smpl_fit(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "case.npy"
    np.save(source, np.zeros((5, 263), dtype=np.float32))
    captured = {}

    def fake_retarget(features, **kwargs):
        captured.update(kwargs)
        frames = int(kwargs["target_len"])
        joints = np.zeros((frames, 22, 3), dtype=np.float32)
        joints[:, 1, 0] = 0.2
        joints[:, 2, 0] = -0.2
        joints[:, 16, 0] = 0.35
        joints[:, 17, 0] = -0.35
        joints[:, [7, 8, 10, 11], 1] = 0.0
        return {
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "body_pose": np.zeros((frames, 63), dtype=np.float32),
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "motion_135": np.zeros((frames, 135), dtype=np.float32),
            "fitted_joints": joints,
            "fit_mpjpe_mm": np.full(frames, 12.0, dtype=np.float32),
            "rotation_init": np.asarray("hml263_end_effectors"),
            "mesh_integrity": _good_mesh_integrity(),
        }

    monkeypatch.setattr(HML_MODULE, "retarget_hml263_clip", fake_retarget)
    stats = HML_MODULE.materialize_case(
        source,
        tmp_path / "smpl" / "case.npz",
        tmp_path / "joints66" / "case.npy",
        smpl_rest=object(),
        expected_frames=7,
        device="cpu",
        source_fps=20.0,
        target_fps=30.0,
        refine_iters=80,
        refine_lr=0.02,
        rotation_init="hml263_end_effectors",
    )

    assert captured["rotation_init"] == "hml263_end_effectors"
    assert stats["fit_mpjpe_mm_mean"] == 12.0
    assert np.load(tmp_path / "joints66" / "case.npy").shape == (7, 66)
    with np.load(tmp_path / "smpl" / "case.npz", allow_pickle=False) as smpl:
        assert smpl["rotation_init"].item() == "hml263_end_effectors"


def test_hml263_materializer_derives_target_length_from_source(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "case.npy"
    np.save(source, np.zeros((8, 263), dtype=np.float32))

    def fake_retarget(_features, **kwargs):
        frames = int(kwargs["target_len"])
        return {
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "body_pose": np.zeros((frames, 63), dtype=np.float32),
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "motion_135": np.zeros((frames, 135), dtype=np.float32),
            "fitted_joints": np.zeros((frames, 22, 3), dtype=np.float32),
            "fit_mpjpe_mm": np.full(frames, 10.0, dtype=np.float32),
            "rotation_init": np.asarray("position_ik"),
            "mesh_integrity": _good_mesh_integrity(),
        }

    monkeypatch.setattr(HML_MODULE, "retarget_hml263_clip", fake_retarget)
    HML_MODULE.materialize_case(
        source,
        tmp_path / "smpl" / "case.npz",
        tmp_path / "joints66" / "case.npy",
        smpl_rest=object(),
        expected_frames=None,
        device="cpu",
        source_fps=20.0,
        target_fps=30.0,
        refine_iters=80,
        refine_lr=0.02,
        rotation_init="position_ik",
    )

    assert np.load(tmp_path / "joints66" / "case.npy").shape == (12, 66)


def test_hml263_materialization_validation_rejects_truncated_output(
    tmp_path: Path,
) -> None:
    frames = 7
    smpl_path = tmp_path / "case.npz"
    joints_path = tmp_path / "case.npy"
    np.savez_compressed(
        smpl_path,
        motion_135=np.zeros((frames, 135), dtype=np.float32),
        global_orient=np.zeros((frames, 3), dtype=np.float32),
        body_pose=np.zeros((frames, 63), dtype=np.float32),
        transl=np.zeros((frames, 3), dtype=np.float32),
        fit_mpjpe_mm=np.zeros(frames, dtype=np.float32),
        rotation_jump_deg_p99=np.asarray(0.0, dtype=np.float32),
        mesh_edge_ratio_p99=np.asarray(1.0, dtype=np.float32),
    )
    np.save(joints_path, np.zeros((frames, 66), dtype=np.float32))

    assert HML_MODULE._valid_materialization(smpl_path, joints_path)
    joints_path.write_bytes(b"\x93NUMPY")
    assert not HML_MODULE._valid_materialization(smpl_path, joints_path)


def test_hml263_materializer_rejects_bad_smpl_fit(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "bad.npy"
    np.save(source, np.zeros((5, 263), dtype=np.float32))

    def fake_retarget(_features, **kwargs):
        frames = int(kwargs["target_len"])
        return {
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "body_pose": np.zeros((frames, 63), dtype=np.float32),
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "fitted_joints": np.zeros((frames, 22, 3), dtype=np.float32),
            "fit_mpjpe_mm": np.full(frames, 75.0, dtype=np.float32),
            "rotation_init": np.asarray("hml263_end_effectors"),
        }

    monkeypatch.setattr(HML_MODULE, "retarget_hml263_clip", fake_retarget)
    with pytest.raises(RuntimeError, match="exceeds the 50.00 mm quality gate"):
        HML_MODULE.materialize_case(
            source,
            tmp_path / "smpl" / "bad.npz",
            tmp_path / "joints66" / "bad.npy",
            smpl_rest=object(),
            expected_frames=7,
            device="cpu",
            source_fps=20.0,
            target_fps=30.0,
            refine_iters=80,
            refine_lr=0.02,
            rotation_init="hml263_end_effectors",
        )


def test_hml263_materializer_falls_back_when_stable_twist_deforms_mesh(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "case.npy"
    np.save(source, np.zeros((5, 263), dtype=np.float32))
    calls = []

    def fake_retarget(_features, **kwargs):
        stable = bool(kwargs["temporal_twist_stabilization"])
        calls.append(stable)
        frames = int(kwargs["target_len"])
        integrity = _good_mesh_integrity()
        integrity["mesh_edge_ratio_p99"] = 2.3 if stable else 1.35
        return {
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "body_pose": np.zeros((frames, 63), dtype=np.float32),
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "motion_135": np.zeros((frames, 135), dtype=np.float32),
            "fitted_joints": np.zeros((frames, 22, 3), dtype=np.float32),
            "fit_mpjpe_mm": np.full(frames, 25.0, dtype=np.float32),
            "rotation_init": np.asarray("position_ik"),
            "temporal_twist_stabilization": np.asarray(stable),
            "mesh_integrity": integrity,
        }

    monkeypatch.setattr(HML_MODULE, "retarget_hml263_clip", fake_retarget)
    destination = tmp_path / "smpl" / "case.npz"
    stats = HML_MODULE.materialize_case(
        source,
        destination,
        tmp_path / "joints" / "case.npy",
        smpl_rest=object(),
        expected_frames=7,
        device="cpu",
        source_fps=20.0,
        target_fps=30.0,
        refine_iters=0,
        refine_lr=0.02,
        rotation_init="position_ik",
    )

    assert calls == [True, False]
    assert stats["mesh_edge_ratio_p99"] == 1.35
    assert stats["temporal_twist_stabilization"] == 0.0
    with np.load(destination, allow_pickle=False) as payload:
        assert not bool(payload["temporal_twist_stabilization"].item())
