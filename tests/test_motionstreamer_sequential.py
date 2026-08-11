import numpy as np
import pytest
import torch

from motius.models.motionstreamer.network.llama_model import LLaMAHF, LLaMAHFConfig
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


class _FakeTextEncoder:
    @staticmethod
    def encode(_text):
        return np.ones(768, dtype=np.float32)


class _FakeTransformer:
    @staticmethod
    def cond_embed(value):
        return value

    @staticmethod
    def wte(value):
        return torch.nn.functional.pad(value, (0, 768 - value.shape[-1]))


class _FakeDiffusionLoss:
    @staticmethod
    def sample(conditions, **_kwargs):
        return torch.ones((len(conditions), 16), dtype=torch.float32)


class _FakeLlama:
    def __init__(self):
        self.transformer = _FakeTransformer()
        self.diff_loss = _FakeDiffusionLoss()

    @staticmethod
    def forward_babel_eval(value, return_attention=False):
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


def test_t5_continuation_does_not_require_clip(monkeypatch):
    original_import = __import__

    def guarded_import(name, *args, **kwargs):
        if name == "clip":
            raise AssertionError("T5 continuation must not import CLIP")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", guarded_import)
    _, latents = LLaMAHF.sample_for_eval_CFG_babel_inference_new_demo(
        _FakeLlama(),
        B_text="then stop",
        A_motion=torch.zeros((1, 16), dtype=torch.float32),
        length=8,
        clip_model=_FakeTextEncoder(),
        device=torch.device("cpu"),
        tokenizer="t5-xxl",
    )
    assert latents.shape == (1, 1, 16)


@pytest.mark.parametrize("use_out_proj", [True, False])
def test_llama_initializes_babel_projection_mode(use_out_proj):
    config = LLaMAHFConfig(
        block_size=8,
        n_layer=1,
        n_head=1,
        n_embd=16,
        T5_xxl_dim=16,
    )
    model = LLaMAHF(
        config,
        num_diffusion_head_layers=1,
        input_token_dim=4,
        device=torch.device("cpu"),
        width=16,
        use_out_proj=use_out_proj,
    )
    assert model.use_out_proj is use_out_proj
