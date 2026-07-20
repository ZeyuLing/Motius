"""Contract tests for the native PRISM integration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from motius.models.prism import PRISMBundle, PRISMMotionProcessor
from motius.pipelines.prism.backend import PrismARPipeline
from motius.pipelines.prism.pipeline import PRISMPipeline
from tools.generate_babel_sequential_prism import parse_args


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


def test_prism_fixed_canvas_never_expands_360_for_carried_prefix():
    calls = []

    class Progress:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def update(self):
            pass

    class Backend:
        vae_scale_factor_temporal = 4

        def progress_bar(self, total):
            assert total == 2
            return Progress()

        def generate_single_segment(
            self, *, first_frame_motion, num_frames, valid_num_frames, **_kwargs
        ):
            calls.append(
                (
                    int(num_frames),
                    int(valid_num_frames),
                    0 if first_frame_motion is None else first_frame_motion.shape[1],
                )
            )
            # A nominal 360-frame PRISM canvas decodes to 357 frames.
            return torch.zeros(1, 357, 23, 6)

        def extract_last_frame_motion(self, motion, num_frames):
            return motion[:, -min(num_frames, motion.shape[1]) :]

        def post_process_motion(self, motion, **_kwargs):
            return {"transl": np.zeros((motion.shape[1], 3), dtype=np.float32)}

    result = PrismARPipeline.__call__(
        Backend(),
        prompts=["walk", "turn"],
        num_frames_per_segment=[360, 360],
        generation_num_frames_per_segment=[360, 360],
        valid_num_frames_per_segment=[360, 360],
        preserve_segment_lengths=True,
        fixed_generation_canvas=True,
        align_generation_frames=False,
        use_blend=False,
        return_motion_vec=True,
    )

    assert [num_frames for num_frames, _, _ in calls] == [360, 360, 360, 360]
    assert calls == [(360, 357, 0), (360, 8, 5), (360, 357, 5), (360, 13, 5)]
    assert result["motion_vec"].shape[1] == 720
    assert result["smplx_dict"]["_prism_generation_num_frames"].tolist() == [360, 360]
    assert result["smplx_dict"]["_prism_generation_chunk_num_frames"].tolist() == [
        360,
        360,
        360,
        360,
    ]


def test_prism_public_sequential_api_enables_fixed_canvas_for_long_segments(monkeypatch):
    captured = {}

    def fake_generate(self, prompts, segment_frames, **kwargs):
        captured.update(kwargs)
        return prompts, segment_frames

    monkeypatch.setattr(PRISMPipeline, "generate", fake_generate)
    pipeline = object.__new__(PRISMPipeline)
    result = pipeline.sequential_generation(["dance", "turn"], [1657, 120])

    assert result == (["dance", "turn"], [1657, 120])
    assert captured["generation_num_frames_per_segment"] == [360, 360]
    assert captured["valid_num_frames_per_segment"] == [1657, 120]
    assert captured["fixed_generation_canvas"] is True
    assert captured["allow_segment_padding"] is False
    assert captured["ar_condition_frames"] == 9


def test_prism_public_default_guidance_is_conservative(monkeypatch):
    captured = {}

    class Backend:
        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {
                "smplx_dict": {
                    "global_orient": np.zeros((1, 3), dtype=np.float32),
                    "body_pose": np.zeros((1, 63), dtype=np.float32),
                    "transl": np.zeros((1, 3), dtype=np.float32),
                },
                "motion_vec": torch.zeros(1, 1, 23, 6),
            }

    class Bundle:
        variant = "kt"
        processor = None

        def load_model(self):
            return Backend()

    pipeline = object.__new__(PRISMPipeline)
    pipeline.bundle = Bundle()
    monkeypatch.setattr(pipeline, "_format_output", lambda result: result)
    pipeline.generate("walk", num_frames=81)

    assert captured["guidance_scale"] == 1.5


def test_prism_babel_runner_default_guidance_is_conservative(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "generate_babel_sequential_prism.py",
            "--manifest",
            "manifest.json",
            "--output-dir",
            "outputs",
        ],
    )
    args = parse_args()
    assert args.guidance_scale == 1.5
    assert args.ar_condition_frames == 9
