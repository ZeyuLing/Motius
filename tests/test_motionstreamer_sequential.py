import numpy as np
import pytest
import torch

from motius.pipelines.motionstreamer.pipeline import (
    MotionStreamerPipeline,
    _resample_frames,
)


class _FakeAR:
    def __init__(self):
        self.calls = []

    def sample_for_eval_CFG(self, captions, *, length, **kwargs):
        self.calls.append((captions[0], length, 0))
        return torch.zeros((1, length // 4, 16), dtype=torch.float32)

    def sample_for_eval_CFG_babel_inference_new_demo(
        self, *, B_text, A_motion, length, **kwargs
    ):
        take = length // 4 - len(A_motion)
        self.calls.append((B_text, length, take))
        return None, torch.ones((1, take, 16), dtype=torch.float32)


class _FakeTAE:
    @staticmethod
    def forward_decoder(latents):
        frames = latents.repeat_interleave(4, dim=1)
        return torch.nn.functional.pad(frames, (0, 272 - frames.shape[-1]))


class _FakeBundle:
    def __init__(self):
        self.ar = _FakeAR()
        self.tae = _FakeTAE()
        self.text_model = object()
        self.guidance_param = 4.0
        self.device = torch.device("cpu")

    def eval(self):
        return self

    @staticmethod
    def denormalize(value):
        return value


def test_exact_length_resampling_preserves_endpoints_and_shape():
    motion = np.arange(8 * 3, dtype=np.float32).reshape(8, 3)
    result = _resample_frames(motion, 5)
    assert result.shape == (5, 3)
    np.testing.assert_allclose(result[0], motion[0])
    np.testing.assert_allclose(result[-1], motion[-1])


@pytest.mark.parametrize("target", [0, -1])
def test_exact_length_resampling_rejects_invalid_targets(target):
    with pytest.raises(ValueError, match="positive"):
        _resample_frames(np.ones((4, 272), dtype=np.float32), target)


def test_exact_sequential_generation_supports_long_first_segment():
    bundle = _FakeBundle()
    pipeline = MotionStreamerPipeline(bundle)
    result = pipeline.infer_sequential_t2m(
        [["a long first action", "then stop"]],
        [[313, 5]],
        exact_lengths=True,
        seed=7,
    )[0]

    assert result.shape == (318, 272)
    assert bundle.ar.calls == [
        ("a long first action", 308, 0),
        ("a long first action", 72, 2),
        ("then stop", 72, 2),
    ]
