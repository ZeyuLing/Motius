import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_babel_sequential_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_babel_protocol", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sample(babel_id="7", frames=90):
    poses = np.zeros((frames, 52, 3), dtype=np.float32)
    trans = np.zeros((frames, 3), dtype=np.float32)
    trans[:, 0] = np.linspace(0, 1, frames)
    trans[:, 2] = 1.0
    joints = np.zeros((frames, 73, 3), dtype=np.float32)
    offsets = np.arange(73, dtype=np.float32)[:, None] * np.array([[0.002, 0.001, 0.004]])
    joints[:] = trans[:, None] + offsets[None]
    return {
        "babel_id": babel_id,
        "fps": 30,
        "poses": poses.reshape(frames, -1),
        "trans": trans,
        "joint_positions": joints,
    }


def test_reference_segments_follow_official_duration_filter():
    annotation = {
        "frame_ann": {
            "labels": [
                {"proc_label": "walk", "start_t": 0.0, "end_t": 1.0},
                {"proc_label": "transition", "start_t": 1.0, "end_t": 2.0},
                {"proc_label": "too short", "start_t": 2.0, "end_t": 2.5},
            ]
        }
    }
    assert list(MODULE.extract_reference_segments(_sample(), annotation)) == [
        ("walk", 0, 30)
    ]


def test_protocol_contains_compositions_and_independent_pools(tmp_path):
    sample = _sample()
    annotations = {
        "7": {
            "frame_ann": {
                "labels": [
                    {"proc_label": "walk", "start_t": 0.0, "end_t": 1.0},
                    {"proc_label": "turn", "start_t": 1.0, "end_t": 2.5},
                ]
            }
        }
    }
    compositions = [
        {
            "id": "000",
            "scenario": "in-distribution",
            "text": ["walk", "turn"],
            "lengths": [30, 45],
        }
    ]
    manifest = MODULE.build_protocol(
        [sample], annotations, compositions, tmp_path, offset_samples=1
    )
    assert manifest["protocol"] == "babel-flowmdm-val-joints66-v2"
    assert manifest["cases"][0]["total_frames"] == 75
    assert len(manifest["reference_segments"]) == 2
    assert len(manifest["reference_transitions"]) == 3
    for key in ("reference_segment_pool", "reference_transition_pool"):
        with np.load(tmp_path / manifest[key]) as pool:
            assert pool["motions"].ndim == 3 and pool["motions"].shape[2] == 66
            assert np.all(pool["lengths"] >= 30)
