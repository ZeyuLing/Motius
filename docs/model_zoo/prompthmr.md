# PromptHMR-Video

PromptHMR is a promptable SMPL-X human mesh recovery model. The video release
combines spatial prompts with tracking and world-motion estimation.

- Paper: <https://arxiv.org/abs/2504.06397>
- Official source: <https://github.com/yufu-wang/PromptHMR>
- Pinned source revision: `3b566b7dbb28ce506c7ea972c18693f4c705ce8c`
- Default video checkpoint: BEDLAM1+BEDLAM2 `phmr_b1b2.ckpt`
- Checkpoint SHA-256:
  `2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5`
- License: upstream non-commercial research terms

**Tasks:** Monocular Motion Capture

## Motius integration

`PromptHMRBundle` verifies the official checkout and local checkpoints.
`PromptHMRPipeline` runs `scripts/demo_video.py` in an isolated environment and
converts the exact `results.pkl` schema to `MonocularCaptureResult`.

```bash
PROMPTHMR_ACCEPT_LICENSE=1 \
DOWNLOAD_VIDEO_CHECKPOINT=true \
bash tools/setup_prompthmr_env.sh
```

The image model, SMPL-X files, and third-party detection, tracking, depth, and
SLAM weights remain user supplied under their respective licenses.

```python
from motius.models.prompthmr import PromptHMRBundle
from motius.pipelines.prompthmr import PromptHMRPipeline

bundle = PromptHMRBundle(
    upstream_dir="outputs/tmp/prompthmr/upstream",
    video_checkpoint="bedlam1+2",
    python_command=("conda", "run", "-n", "phmr_pt2.4", "python"),
)
pipeline = PromptHMRPipeline(bundle)
result = pipeline(
    "input.mp4",
    original_fps=30.0,
)
```

Camera-space fields map from `smplx_cam`. World fields are populated only when
the official result contains `smplx_world` or `camera_world`; Motius never
promotes camera coordinates to world coordinates. Official result files do not
store joints or vertices. Formal evaluation uses the separately licensed replay
step, which hashes the user model and writes a pickle-free Motius artifact:

```bash
python tools/materialize_prompthmr_smplx.py \
  --official-results outputs/tmp/prompthmr/upstream/results/clip/results.pkl \
  --output outputs/evaluation/monocular_capture/prompthmr/clip.motius.npz \
  --smplx-model /private/models/SMPLX_NEUTRAL.npz \
  --model-version 1.1 \
  --video-checkpoint-sha256 2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5 \
  --original-fps 30
```

Only valid frames are replayed. Camera/world geometry remains explicitly marked
as `licensed_smplx_replay`, and missing world parameters are never synthesized.

## Evaluation Results

Official full-frame, multi-person inference completed all 24 3DPW Test videos
and matched all 37 official target tracks. Licensed neutral SMPL-X replay
materialized camera joints without redistributing the body model. The official
tracker retained 35,130 of 35,515 valid target frames (98.92% frame coverage);
missing frames remain masked and are never interpolated for metrics.

| Protocol | Coverage | MPJPE | PA-MPJPE | Acceleration |
| --- | ---: | ---: | ---: | ---: |
| 3DPW Test camera, official full-frame multi-track | 98.92% | 64.50 mm | 46.17 mm | 4.801 m/s² |

- Full metric artifact:
  `outputs/evaluation/monocular_capture/3dpw_test/prompthmr_official_multitrack_v1/metrics.json`
- Full prediction root:
  `outputs/evaluation/monocular_capture/3dpw_test/prompthmr_official_multitrack_v1/predictions/`
- Native SMPL-X Mesh artifact:
  `outputs/visualization/monocular_capture/3dpw_test/prompthmr/downtown_arguing_00_mesh.motius.npz`
  (2 tracks, 30 frames, 10,475 vertices per frame)
- Video-head SHA-256:
  `2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5`
- Image-model SHA-256:
  `a3ef04ef8a12c3068682b03c62c95f8959cd8554424e105c63bf97f6c8e97e99`
- Runtime revision: `3b566b7dbb28ce506c7ea972c18693f4c705ce8c`

PromptHMR processes a resized video internally. Association maps its saved
tracking boxes back to the original 3DPW pixel space using the recorded camera
principal points before computing IoU. This result is not ranked against
per-target GT-crop runs because its input is the official full-frame
multi-person protocol.

The optional MCS/GLB exporter requires a separately licensed slim SMPL-X asset;
its absence does not invalidate the authoritative `results.pkl`, which is
serialized before visualization export.
