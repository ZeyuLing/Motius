from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


TOOL_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "materialize_tm2d_aistpp_smpl.py"
)
SPEC = importlib.util.spec_from_file_location("materialize_tm2d_aistpp_smpl", TOOL_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_tm2d_aistpp_materializer_disables_position_only_refinement(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "case.npz"
    np.savez_compressed(
        source,
        joints=np.zeros((12, 24, 3), dtype=np.float32),
        fps=np.asarray(60.0, dtype=np.float32),
    )
    captured = {}

    def fake_retarget(joints, **kwargs):
        captured["joints"] = joints
        captured.update(kwargs)
        frames = 6
        return {
            "motion_135": np.zeros((frames, 135), dtype=np.float32),
            "global_orient": np.zeros((frames, 3), dtype=np.float32),
            "body_pose": np.zeros((frames, 63), dtype=np.float32),
            "transl": np.zeros((frames, 3), dtype=np.float32),
            "target_joints": np.zeros((frames, 22, 3), dtype=np.float32),
            "fitted_joints": np.zeros((frames, 22, 3), dtype=np.float32),
            "fit_mpjpe_mm": np.full(frames, 20.0, dtype=np.float32),
            "rotation_init": np.asarray("position_ik"),
            "mesh_integrity": {
                "rotation_jump_deg_p99": 25.0,
                "rotation_jump_deg_max": 40.0,
                "mesh_edge_ratio_p01": 0.7,
                "mesh_edge_ratio_p99": 1.3,
                "mesh_edge_ratio_max": 4.0,
                "mesh_sample_count": 6.0,
            },
        }

    monkeypatch.setattr(MODULE, "retarget_hml263_clip", fake_retarget)
    destination = tmp_path / "smpl" / "case.npz"
    stats = MODULE.materialize_case(
        source,
        destination,
        smpl_rest=object(),
        device="cpu",
        target_fps=30.0,
    )

    assert captured["joints"].shape == (12, 22, 3)
    assert captured["source_fps"] == 60.0
    assert captured["target_fps"] == 30.0
    assert captured["refine_iters"] == 0
    assert captured["compute_mesh_metrics"] is True
    assert stats["mesh_edge_ratio_p99"] == 1.3
    with np.load(destination, allow_pickle=False) as payload:
        assert payload["motion_135"].shape == (6, 135)
        assert payload["rotation_jump_deg_p99"].item() == 25.0
