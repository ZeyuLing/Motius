"""Tests for the Motius OmniControl control contract."""

import numpy as np
import pytest
import torch

pytest.importorskip("clip")

from motius.pipelines.omnicontrol import OmniControlPipeline
from tools.eval_omnicontrol_temporal_humanml3d import _caption, _keyframe_indices


class _HintBundle(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("raw_mean", torch.zeros(66))
        self.register_buffer("raw_std", torch.ones(66))

    @property
    def device(self):
        return self.raw_mean.device


def test_temporal_first_frame_controls_all_joint_axes():
    pipeline = OmniControlPipeline(_HintBundle())
    motion = np.zeros((40, 263), dtype=np.float32)

    hint, axis_mask, frame_mask = pipeline._build_hints(
        [motion], [40], 40, "first_frame", None, "xyz", None, 0.2, 0.1
    )

    assert hint.shape == (1, 40, 66)
    assert axis_mask[0, 0].all()
    assert not axis_mask[0, 1:].any()
    assert frame_mask[0].tolist() == [True] + [False] * 39


def test_sparse_root_xz_keeps_axes_independent():
    pipeline = OmniControlPipeline(_HintBundle())
    motion = np.zeros((40, 263), dtype=np.float32)

    _, axis_mask, _ = pipeline._build_hints(
        [motion], [40], 40, "keyframes", [0], "xz", [[3, 17]], 0.2, 0.1
    )

    assert axis_mask[0, 3, 0].tolist() == [True, False, True]
    assert axis_mask[0, 17, 0].tolist() == [True, False, True]
    assert not axis_mask[0, :, 1:].any()


def test_adaptive_keyframes_map_by_fraction_to_model_length():
    entry = {"T": 300, "keyframe_indices": [0, 150, 299]}

    assert _keyframe_indices(entry, 196) == [0, 98, 195]


def test_temporal_runner_reads_protocol_caption_records():
    assert _caption({"caption": "a person walks"}) == "a person walks"
    assert _caption({"caption_en": "a person turns"}) == "a person turns"
