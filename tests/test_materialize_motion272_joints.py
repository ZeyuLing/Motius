from __future__ import annotations

import importlib.util
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
            "fitted_joints": joints,
            "fit_mpjpe_mm": np.full(frames, 12.0, dtype=np.float32),
            "rotation_init": np.asarray("hml263_end_effectors"),
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
