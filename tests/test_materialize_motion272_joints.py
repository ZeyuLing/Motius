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
