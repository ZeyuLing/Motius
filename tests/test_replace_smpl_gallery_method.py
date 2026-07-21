from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "replace_smpl_gallery_method.py"
SPEC = importlib.util.spec_from_file_location("replace_smpl_gallery_method", TOOL_PATH)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _motion(frames: int) -> np.ndarray:
    value = np.zeros((frames, 135), dtype=np.float32)
    value[:, 3:] = np.tile(np.eye(3, dtype=np.float32)[:, :2].reshape(6), 22)
    return value


def test_replace_method_preserves_grouping_and_rewrites_offsets(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    for case_id, error in (("a", 20.0), ("b", 25.0)):
        np.savez_compressed(
            source / f"{case_id}.npz",
            motion_135=_motion(6),
            fit_mpjpe_mm=np.full(6, error, dtype=np.float32),
            rotation_jump_deg_p99=np.asarray(10.0, dtype=np.float32),
            mesh_edge_ratio_p99=np.asarray(1.3, dtype=np.float32),
        )
    manifest = {
        "motion_methods": [{"key": "tm2d", "label": "TM2D"}],
        "cases": [
            {
                "case_id": case_id,
                "motions": {
                    "tm2d": {
                        "asset": "assets/tm2d_000.smpl",
                        "display_frames": 6,
                        "frames": 3,
                        "stride": 2,
                        "fps": 30.0,
                    }
                },
            }
            for case_id in ("a", "b")
        ],
    }
    output = tmp_path / "gallery"
    summary = MODULE.replace_method_assets(
        manifest, method_key="tm2d", motion_dir=source, output_dir=output
    )

    assert summary["assets"] == 1
    first, second = manifest["cases"]
    assert first["motions"]["tm2d"]["translation_offset"] == 0
    assert second["motions"]["tm2d"]["translation_offset"] > 0
    assert first["motions"]["tm2d"]["fit_mpjpe_mm_mean"] == 20.0
    assert first["motions"]["tm2d"]["rotation_jump_deg_p99"] == 10.0
    assert (output / "assets" / "tm2d_000.smpl").is_file()
    assert (output / "manifest.json").is_file()
