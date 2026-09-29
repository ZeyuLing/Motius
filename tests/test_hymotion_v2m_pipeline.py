import pytest
import torch

from motius.utils.monocular_parity import MonocularParityTrace
from motius.models.hymotion_v2m.bundle import HyMotionV2MBundle
from motius.pipelines.hymotion_v2m import HyMotionV2MPipeline


class _BodyModel:
    def __call__(self, inputs):
        frames = int(inputs["trans"].shape[0])
        return {
            "keypoints3d": torch.zeros(
                frames,
                22,
                3,
                device=inputs["trans"].device,
            )
        }


class _FixedCanvasModel(torch.nn.Module):
    def __init__(self, train_frames: int = 360):
        super().__init__()
        self.train_frames = train_frames
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.received_length = None
        self.body_model = _BodyModel()

    def generate(self, *, feature, length, **kwargs):
        self.received_length = length
        batch = len(kwargs["seeds"])
        device = feature["feature"].device
        return {
            "rot6d": torch.zeros(batch, self.train_frames, 22, 6, device=device),
            "trans": torch.zeros(batch, self.train_frames, 3, device=device),
            "shapes": torch.zeros(batch, 1, 16, device=device),
            "global_orient": torch.zeros(
                batch, self.train_frames, 6, device=device
            ),
            "local_transl_vel": torch.zeros(
                batch, self.train_frames, 3, device=device
            ),
            "end_effector_vel": torch.zeros(
                batch, self.train_frames, 4, 3, device=device
            ),
        }


def _fake_bundle(train_frames: int = 360) -> HyMotionV2MBundle:
    bundle = HyMotionV2MBundle.__new__(HyMotionV2MBundle)
    torch.nn.Module.__init__(bundle)
    bundle.model = _FixedCanvasModel(train_frames)
    return bundle


def test_bundle_uses_full_canvas_then_trims_ragged_window():
    bundle = _fake_bundle()
    feature = {
        "feature": torch.zeros(1, 360, 1024),
        "camera_R": torch.zeros(1, 360, 9),
        "camera_T": torch.zeros(1, 360, 3),
    }

    output = bundle.generate_from_feature(feature, seeds=[7], length=238)

    assert bundle.model.received_length == 360
    assert output["rot6d"].shape == (1, 238, 22, 6)
    assert output["trans"].shape == (1, 238, 3)
    assert output["shapes"].shape == (1, 1, 16)


def test_bundle_rejects_unpadded_feature_window():
    bundle = _fake_bundle()
    feature = {
        "feature": torch.zeros(1, 238, 1024),
        "camera_R": torch.zeros(1, 238, 9),
        "camera_T": torch.zeros(1, 238, 3),
    }

    with pytest.raises(ValueError, match="temporal length 360"):
        bundle.generate_from_feature(feature, seeds=[0], length=238)


def test_stitched_output_is_trimmed_to_original_video_length():
    first = {
        "rot6d": torch.zeros(1, 360, 22, 6),
        "trans": torch.zeros(1, 360, 3),
    }
    second = {
        "rot6d": torch.ones(1, 360, 22, 6),
        "trans": torch.ones(1, 360, 3),
    }
    third = {
        "rot6d": torch.full((1, 10, 22, 6), 2.0),
        "trans": torch.full((1, 10, 3), 2.0),
    }

    stitched = HyMotionV2MPipeline._concat_segment_outputs(
        [first, second, third],
        hop=330,
    )
    trimmed = HyMotionV2MPipeline._trim_temporal_outputs(stitched, 670)

    assert trimmed["rot6d"].shape[1] == 670
    assert trimmed["trans"].shape[1] == 670


def test_nonfinite_window_is_rejected_before_stitching():
    output = {"rot6d": torch.zeros(1, 20, 22, 6)}
    output["rot6d"][0, 3, 4, 2] = torch.nan

    with pytest.raises(RuntimeError, match="window_start=330"):
        HyMotionV2MPipeline._require_finite_window(
            output,
            start=330,
            valid_length=20,
        )


def test_feature_inference_captures_every_numerical_stage():
    pipeline = HyMotionV2MPipeline(_fake_bundle())
    trace = MonocularParityTrace("motius")

    output = pipeline.infer_from_feature(
        torch.zeros(5, 1024),
        floor_method="lowest",
        parity_trace=trace,
    )

    assert output["keypoints3d"].shape == (1, 5, 22, 3)
    assert trace.stage_names == (
        "04_model_conditioning",
        "05_window_input_0000",
        "06_window_output_0000",
        "07_stitched_motion",
        "08_smpl_fk",
        "09_ground_alignment",
        "10_native_output",
    )
