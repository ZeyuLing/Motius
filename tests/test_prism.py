"""Contract tests for the native PRISM integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from motius.models.prism import PRISMBundle, PRISMMotionProcessor


def _write_stats(path: Path) -> Path:
    stats = {
        "transl": {"mean": [0.1, 0.2, 0.3], "std": [1.0, 2.0, 3.0]},
        "transl_vel": {"mean": [0.0] * 3, "std": [0.5] * 3},
        "global_orient": {
            "rotation_6d": {"mean": [0.0] * 6, "std": [1.0] * 6}
        },
        "body_pose": {
            "rotation_6d": {"mean": [0.0] * 126, "std": [1.0] * 126}
        },
    }
    path.write_text(json.dumps(stats))
    return path


def test_prism_bundle_aliases_without_loading_weights():
    one = PRISMBundle.from_pretrained("1.0", load_model=False)
    kt = PRISMBundle.from_pretrained("kt", load_model=False)
    assert one.checkpoint_path == "1.0"
    assert kt.checkpoint_path == "kt"
    assert one.CHECKPOINTS["1.0"].endswith("motius-prism-1.0-humanml3d")
    assert kt.CHECKPOINTS["kt"].endswith("motius-prism-kt-humanml3d")
    assert set(kt.SUPPORTED_TASKS) == {"T2M", "TP2M", "Sequential Generation"}


def test_prism_motion138_stats_roundtrip(tmp_path: Path):
    processor = PRISMMotionProcessor(_write_stats(tmp_path / "stats.json"))
    assert processor.mean.shape == (138,)
    assert processor.std.shape == (138,)

    motion = torch.randn(2, 17, 138)
    restored = processor.denormalize(processor.normalize(motion))
    torch.testing.assert_close(restored, motion, atol=1e-6, rtol=1e-6)


def test_prism_abs_rel_translation_decode(tmp_path: Path):
    processor = PRISMMotionProcessor(_write_stats(tmp_path / "stats.json"))
    encoded = torch.zeros(1, 4, 6)
    encoded[0, :, :3] = torch.tensor(
        [[1.0, 2.0, 3.0], [9.0, 4.0, 9.0], [9.0, 6.0, 9.0], [9.0, 8.0, 9.0]]
    )
    encoded[0, 1:, 3:] = torch.tensor(
        [[0.5, 7.0, 1.0], [0.5, 7.0, 1.0], [0.5, 7.0, 1.0]]
    )
    decoded = processor.inv_convert_transl(encoded, "xz_rollout_y_absolute")
    expected = torch.tensor(
        [[[1.0, 2.0, 3.0], [1.5, 4.0, 4.0], [2.0, 6.0, 5.0], [2.5, 8.0, 6.0]]]
    )
    torch.testing.assert_close(decoded, expected)


def test_prism_smpl_output_is_body22_at_30fps():
    frames = 5
    smpl = PRISMMotionProcessor.transl_pose_to_smplx_dict(
        np.zeros((frames, 3), dtype=np.float32),
        np.zeros((frames, 66), dtype=np.float32),
    )
    assert smpl["poses"].shape == (frames, 165)
    assert smpl["body_pose"].shape == (frames, 63)
    assert smpl["mocap_framerate"] == 30.0
