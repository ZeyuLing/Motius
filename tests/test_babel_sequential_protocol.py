import importlib.util
import json
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "tools" / "build_babel_sequential_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_babel_protocol", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _record():
    return {
        "id": "val_7",
        "babel_id": "7",
        "split": "val",
        "fps": 30.0,
        "duration_sec_npz": 4.0,
        "amass_path": "amass/example.npz",
        "segments": [
            {"caption": "walk", "raw_label": "walking", "start_t": 0.0, "end_t": 0.6},
            {"caption": "transition", "start_t": 0.6, "end_t": 1.0, "is_transition": True},
            {"caption": "turn", "raw_label": "turning", "start_t": 1.0, "end_t": 2.5},
            {"caption": "transition", "start_t": 2.5, "end_t": 3.0, "is_transition": True},
            {"caption": "stand", "raw_label": "standing", "start_t": 3.0, "end_t": 4.0},
        ],
    }


def _motion272(frames=120):
    motion = np.zeros((frames, 272), dtype=np.float32)
    motion[:, 2:8] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    motion[:, 140:272] = np.tile(
        np.array([1, 0, 0, 0, 1, 0], dtype=np.float32), 22
    )
    time = np.arange(frames, dtype=np.float32)[:, None, None]
    joint = np.arange(22, dtype=np.float32)[None, :, None]
    xyz = np.arange(3, dtype=np.float32)[None, None, :]
    motion[:, 8:74] = (time * 0.001 + joint * 0.01 + xyz * 0.1).reshape(frames, 66)
    return motion


def _smpl22_offsets():
    offsets = np.zeros((22, 3), dtype=np.float32)
    offsets[0, 1] = -0.2
    offsets[1:, 1] = -0.1
    return offsets


def _babel_annotations():
    return {
        "7": {
            "seq_ann": {"labels": [{"act_cat": ["walk", "turn", "stand"]}]},
            "frame_ann": {
                "labels": [
                    {
                        "raw_label": "walking",
                        "proc_label": "walk",
                        "act_cat": ["walk"],
                    },
                    {
                        "raw_label": "transition",
                        "proc_label": "transition",
                        "act_cat": ["transition"],
                    },
                    {
                        "raw_label": "turning",
                        "proc_label": "turn",
                        "act_cat": ["turn"],
                    },
                    {
                        "raw_label": "standing",
                        "proc_label": "stand",
                        "act_cat": ["stand"],
                    },
                ]
            },
        }
    }


def test_transition_midpoints_and_short_action_merge():
    episode, _ = MODULE.build_official_episode(_record())
    assert [item["end"] for item in episode["segments"]] == [24, 82, 120]
    merged, stats = MODULE.merge_short_segments(episode, min_frames=30)
    assert [(item["start"], item["end"]) for item in merged["segments"]] == [
        (0, 82),
        (82, 120),
    ]
    assert stats["merged_groups"] == 1
    assert stats["remaining_short_segments"] == 0


def test_protocol_uses_llm_rewrites_and_episode_references(tmp_path):
    episode, _ = MODULE.build_official_episode(_record())
    episode, _ = MODULE.merge_short_segments(episode, min_frames=30)
    cache = {
        MODULE._rewrite_key(episode["segments"][0], "labels"): "A person walks, then turns.",
        MODULE._rewrite_key(episode["segments"][1], "labels"): "A person stands.",
    }
    motion_dir = tmp_path / "ms272"
    motion_dir.mkdir()
    np.savez_compressed(motion_dir / "val_7.npz", motion_272=_motion272())
    output = tmp_path / "protocol"
    manifest = MODULE.build_protocol(
        [_record()],
        motion272_dir=motion_dir,
        smpl22_offsets=_smpl22_offsets(),
        rewrite_cache=cache,
        babel_annotations=_babel_annotations(),
        output_root=output,
    )
    assert manifest["protocol"] == MODULE.PROTOCOL
    assert manifest["counts"] == {
        "episodes": 1,
        "original_action_segments": 3,
        "captioned_segments": 2,
        "transition_boundaries": 1,
        "merged_groups": 1,
        "rewrite_hits": 2,
        "rewrite_misses": 0,
        "action_groups": 2,
    }
    case = manifest["cases"][0]
    assert [item["caption"] for item in case["segments"]] == [
        "A person walks, then turns.",
        "A person stands.",
    ]
    assert all(item["action_group_id"].startswith("babel-act-cat-v1:") for item in case["segments"])
    reference = np.load(output / case["reference_path"])
    assert reference.shape == (120, 66)
    assert np.isfinite(reference).all()
    assert manifest["smpl22_offsets"] == "smpl22_offsets_y.npy"


def test_protocol_accepts_preclipped_ms272(tmp_path):
    record = _record()
    for segment in record["segments"]:
        segment["start_t"] += 1.0
        segment["end_t"] += 1.0
    record["duration_sec_npz"] = 5.0
    episode, _ = MODULE.build_official_episode(record)
    episode, _ = MODULE.merge_short_segments(episode, min_frames=30)
    cache = {
        MODULE._rewrite_key(item, "labels"): f"A person performs action {index}."
        for index, item in enumerate(episode["segments"])
    }
    motion_dir = tmp_path / "ms272"
    motion_dir.mkdir()
    np.savez_compressed(motion_dir / "val_7.npz", motion_272=_motion272(120))
    manifest = MODULE.build_protocol(
        [record],
        motion272_dir=motion_dir,
        smpl22_offsets=_smpl22_offsets(),
        rewrite_cache=cache,
        babel_annotations=_babel_annotations(),
        output_root=tmp_path / "protocol",
    )
    reference = np.load(
        tmp_path / "protocol" / manifest["cases"][0]["reference_path"]
    )
    assert reference.shape == (120, 66)


def test_public_babel_viewer_has_colored_captioned_subclips():
    root = Path(__file__).resolve().parents[1] / "assets/evaluation/babel_sequential_demo"
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["protocol"].endswith("actiongroups-v3")
    assert len(manifest["episodes"]) == 3
    for episode in manifest["episodes"]:
        assert len(episode["segments"]) >= 5
        assert len({segment["color"] for segment in episode["segments"]}) == len(
            episode["segments"]
        )
        for key in ("gt_file", "prediction_file"):
            path = root / episode[key]
            assert path.stat().st_size == episode["frames"] * 22 * 3 * 4
    viewer = (root / "index.html").read_text()
    assert "three@0.128.0" in viewer
    assert "Captioned subclips" in viewer
    assert "addTrajectories" in viewer
