"""Strict parity gates for canonical infer_{task} pipeline entrypoints."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from motius.pipelines.ardy import ARDYPipeline
from motius.pipelines.hymotion_t2m import HyMotionT2MPipeline
from motius.pipelines.kimodo import KIMODOPipeline


def _load_audit_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "tools"
        / "audit_pipeline_task_apis.py"
    )
    spec = importlib.util.spec_from_file_location("pipeline_task_api_audit", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_all_registered_task_routes_are_parity_verified():
    report = _load_audit_module().audit_task_apis()

    assert report["summary"] == {
        "artifacts": 48,
        "artifact_task_bindings": 82,
        "source_pipeline_classes": 47,
        "pipeline_classes": 43,
        "task_methods": 74,
        "native": 26,
        "identity": 43,
        "verified_adapters": 5,
        "class_exclusions": 4,
        "failed": 0,
    }
    assert report["coverage_errors"] == []
    assert {
        entry["class_name"]: entry["reason"]
        for entry in report["class_exclusions"]
    } == {
        "BasePipeline": "abstract pipeline base class",
        "MotionCLIPPipeline": "evaluator/retrieval API, not a method Pipeline",
        "Pipeline": "automatic artifact loader that returns a method Pipeline",
        "PrismARPipeline": "internal Diffusers backend used by PRISMPipeline",
    }


def test_ardy_sequential_adapter_matches_manual_stream_steps_bit_exact():
    pipeline = ARDYPipeline.__new__(ARDYPipeline)
    pipeline.bundle = SimpleNamespace(
        fps=30.0,
        model=SimpleNamespace(skeleton=SimpleNamespace(name="G1")),
    )

    def stream_step(caption, state=None, **kwargs):
        state = 0 if state is None else int(state)
        value = np.asarray(
            [state, len(caption), kwargs["seed"]],
            dtype=np.int64,
        )
        return {"motion": value}, state + 1

    pipeline.stream_step = stream_step
    prompts = ["walk", "turn", "sit"]
    kwargs = {"seed": 7}

    expected_segments = []
    expected_state = None
    for prompt in prompts:
        output, expected_state = stream_step(prompt, expected_state, **kwargs)
        expected_segments.append(output)

    actual = pipeline.infer_sequential_text_to_motion(prompts, **kwargs)

    assert actual["state"] == expected_state
    assert actual["fps"] == 30.0
    assert actual["representation"] == "ardy_g1_414"
    assert len(actual["segments"]) == len(expected_segments)
    for result, expected in zip(actual["segments"], expected_segments):
        assert result.keys() == expected.keys()
        assert np.array_equal(result["motion"], expected["motion"])


def test_hymotion_text_adapter_forwards_the_exact_legacy_batch():
    result = {"latent": np.arange(6, dtype=np.float32).reshape(2, 3)}

    class ProbePipeline(HyMotionT2MPipeline):
        def __call__(self, batch):
            self.received_batch = batch
            return result

    pipeline = ProbePipeline.__new__(ProbePipeline)
    actual = pipeline.infer_text_to_motion(
        ["walk", "turn"],
        [120, 180],
        data_src=["HYMotion", "MotionHub"],
    )

    assert actual is result
    assert pipeline.received_batch == {
        "caption": ["walk", "turn"],
        "num_frames": [120, 180],
        "data_src": ["HYMotion", "MotionHub"],
    }


def test_kimodo_scalar_text_adapter_matches_legacy_call_bit_exact():
    sentinel = {"motion": np.arange(12, dtype=np.float32).reshape(2, 6)}
    calls = []

    class Bundle:
        def generate(self, *args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

    pipeline = KIMODOPipeline(Bundle())
    legacy = pipeline.text_to_motion("walk forward", 120, seed=11)
    legacy_call = calls.pop()
    canonical = pipeline.infer_text_to_motion("walk forward", 120, seed=11)
    canonical_call = calls.pop()

    assert canonical is legacy
    assert canonical_call == legacy_call


def test_kimodo_batch_text_adapter_forwards_without_data_transforms():
    sentinel = {"motion": np.arange(8, dtype=np.float32).reshape(2, 4)}
    calls = []

    class Bundle:
        def generate(self, *args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

    pipeline = KIMODOPipeline(Bundle())
    actual = pipeline.infer_text_to_motion(
        ["walk", "turn"],
        [120, 180],
        seed=13,
    )

    assert actual is sentinel
    assert calls == [
        (
            (["walk", "turn"], [120, 180]),
            {"constraints": None, "multi_prompt": False, "seed": 13},
        )
    ]


def test_kimodo_sequential_adapter_matches_multi_prompt_bit_exact():
    sentinel = {"motion": np.arange(10, dtype=np.float32).reshape(2, 5)}
    calls = []

    class Bundle:
        def generate(self, *args, **kwargs):
            calls.append((args, kwargs))
            return sentinel

    pipeline = KIMODOPipeline(Bundle())
    legacy = pipeline.multi_prompt(["walk", "sit"], [90, 120], seed=17)
    legacy_call = calls.pop()
    canonical = pipeline.infer_sequential_text_to_motion(
        ["walk", "sit"],
        [90, 120],
        seed=17,
    )
    canonical_call = calls.pop()

    assert canonical is legacy
    assert canonical_call == legacy_call


def test_kimodo_temporal_adapter_accepts_native_reference(monkeypatch):
    from motius.pipelines.kimodo import kimodo_pipeline

    reference = {
        "local_rot_mats": np.zeros((12, 77, 3, 3), dtype=np.float32),
        "root_positions": np.zeros((12, 3), dtype=np.float32),
    }
    constraint = object()
    calls = []

    class Bundle:
        model = object()

        def generate(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"posed_joints": np.zeros((12, 77, 3), dtype=np.float32)}

    monkeypatch.setattr(
        kimodo_pipeline,
        "_make_native_prefix_constraint",
        lambda model, native_motion, cond_frames: (
            constraint,
            len(native_motion["root_positions"]),
        ),
    )
    pipeline = KIMODOPipeline(Bundle())
    output = pipeline.infer_temporal_motion_completion(
        ["walk and turn"],
        [reference],
        condition_frames=4,
        seed=5,
    )

    assert output[0]["posed_joints"].shape == (12, 77, 3)
    assert calls == [
        (
            ("walk and turn", 12),
            {"constraints": [constraint], "seed": 5},
        )
    ]


def test_kimodo_constraints_default_to_model_device():
    pipeline = KIMODOPipeline(
        SimpleNamespace(model=SimpleNamespace(device="cuda:3"))
    )

    assert pipeline._constraint_device() == "cuda:3"
    assert pipeline._constraint_device("cpu") == "cpu"


def test_kimodo_multi_prompt_defaults_to_one_sample(monkeypatch):
    import torch
    from motius.models.kimodo.network.model.kimodo_model import Kimodo

    class StopGeneration(Exception):
        pass

    class MotionRep:
        def create_conditions_from_constraints_batched(
            self,
            constraint_lst,
            lengths,
            **kwargs,
        ):
            del constraint_lst, kwargs
            shape = (len(lengths), int(lengths.max()), 1)
            return torch.zeros(shape), torch.zeros(shape, dtype=torch.bool)

        @staticmethod
        def normalize(value):
            return value

    model = object.__new__(Kimodo)
    torch.nn.Module.__init__(model)
    model.device = torch.device("cpu")
    model.motion_rep = MotionRep()
    captured = {}

    def stop_after_heading(*args, first_heading_angle, **kwargs):
        del args, kwargs
        captured["heading"] = first_heading_angle
        raise StopGeneration

    monkeypatch.setattr(model, "_generate", stop_after_heading)
    with pytest.raises(StopGeneration):
        model._multiprompt(
            ["walk forward"],
            [12],
            num_denoising_steps=2,
            constraint_lst=[],
            num_samples=None,
        )

    assert torch.equal(captured["heading"], torch.zeros(1))


def test_kimodo_moves_plain_text_encoder_to_runtime_device(monkeypatch):
    import torch
    from motius.models.kimodo.network.model import kimodo_model

    class Denoiser(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.motion_rep = SimpleNamespace(
                skeleton=object(),
                fps=30,
            )

    class TextEncoder:
        def __init__(self):
            self.devices = []

        def to(self, device):
            self.devices.append(torch.device(device))
            return self

    monkeypatch.setattr(
        kimodo_model,
        "ClassifierFreeGuidedModel",
        lambda denoiser, cfg_type: denoiser,
    )
    text_encoder = TextEncoder()
    kimodo_model.Kimodo(
        Denoiser(),
        text_encoder,
        num_base_steps=10,
        device="cpu",
    )

    assert text_encoder.devices == [torch.device("cpu")]
