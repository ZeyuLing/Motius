"""Condition tensors survive the file-to-Trainer boundary without frame shifts."""

import json
import numpy as np
import pytest
import torch
from motius.datasets.prepared_motion import PreparedMotionDataset


def test_prepared_condmdi_observations_keep_frame_feature_alignment(tmp_path):
    motion = np.arange(4 * 263, dtype=np.float32).reshape(4, 263)
    observed = np.zeros_like(motion, dtype=bool)
    observed[1, 7] = True
    np.savez(
        tmp_path / "motion.npz",
        motion=motion,
        length=np.array(3),
        obs_mask=observed,
        obs_x0=motion,
    )
    manifest = tmp_path / "records.json"
    manifest.write_text(json.dumps([dict(path="motion.npz", text="walk")]))
    result = PreparedMotionDataset(manifest, "condmdi")[0]
    assert result["obs_mask"].shape == (263, 1, 4)
    assert result["obs_mask"][7, 0, 1] and result["obs_mask"].sum() == 1
    torch.testing.assert_close(result["obs_x0"][:, 0].T, result["motion"])
    assert result["motion_lens"] == 3 and result["text"] == "walk"
    observed[3, 7] = True
    np.savez(
        tmp_path / "motion.npz", motion=motion, length=np.array(3), obs_mask=observed
    )
    with pytest.raises(ValueError, match="Observed padding"):
        PreparedMotionDataset(manifest, "condmdi")[0]


def test_prepared_omnicontrol_preserves_normalized_hints(tmp_path):
    hint = np.arange(4 * 66, dtype=np.float32).reshape(4, 66) / 100
    np.savez(
        tmp_path / "motion.npz",
        motion=np.zeros((4, 263), np.float32),
        length=np.array(4),
        hint=hint,
    )
    manifest = tmp_path / "records.jsonl"
    manifest.write_text(json.dumps(dict(path="motion.npz", text="walk")) + "\n")
    result = PreparedMotionDataset(manifest, "omnicontrol")[0]
    np.testing.assert_array_equal(result["hint"].numpy(), hint)
