<h1 align="center">PromptHMR-Video Model Card</h1>

<p align="center">
  <strong>Promptable SMPL-X recovery with tracking and world-motion estimation.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2504.06397">Paper</a> ·
  <a href="https://github.com/yufu-wang/PromptHMR">Original GitHub</a> ·
  <a href="https://github.com/yufu-wang/PromptHMR#installation">Official Weights</a>
</p>

PromptHMR is a promptable SMPL-X human mesh recovery model. Motius provides a
pinned official-runtime adapter, output conversion, licensed SMPL-X geometry
replay, and evaluation. It is not a redistributable Motius package.
Official source: [yufu-wang/PromptHMR](https://github.com/yufu-wang/PromptHMR),
pinned to `3b566b7dbb28ce506c7ea972c18693f4c705ce8c`. The default BEDLAM1+BEDLAM2
checkpoint has SHA-256
`2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5`.

| Task status | Restricted upstream runtime |
| --- | --- |

<!-- MOTIUS_MODEL_CARD_NAV:START -->
<p align="center">
  <a href="#visual-results">Visual Results</a> ·
  <a href="#model-overview">Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#evaluation-results">Evaluation</a> ·
  <a href="#motion-representation">Motion Representation</a>
</p>
<!-- MOTIUS_MODEL_CARD_NAV:END -->

## Visual Results

<!-- MOTIUS_TASK_DEMOS:START -->

### Task Demos

| Task | Input / condition | Rendered output | More |
| --- | --- | --- | --- |
| No registered public task | See the capability boundary below | No redistributable repository preview | — |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


No repository-hosted Motius preview is published because the upstream license
does not permit redistributing the runtime or checkpoint. The measured 3DPW
result below was produced from a user-accepted official installation and
licensed SMPL-X replay; it is not presented as a downloadable demo artifact.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Not registered | No canonical task API | See the capability boundary below |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | No single fixed motion clock; the image model is trained frame-wise |
| Public preview | Input-video clock unless `output_fps` is requested |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


`PromptHMRBundle` verifies the official checkout and local checkpoints.
`PromptHMRPipeline` runs `scripts/demo_video.py` in an isolated environment and
converts the exact `results.pkl` schema to `MonocularCaptureResult`.

## Quick Start

The upstream license prohibits reproducing, modifying, or making the software
available to a third party without prior written permission. Motius therefore
does not vendor PromptHMR, redistribute a derivative runtime, or publish a
complete Hugging Face artifact. Inference requires the user's own accepted,
pinned official checkout. `Pipeline.from_pretrained` is intentionally
unavailable for this restricted adapter.

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
result = pipeline.infer_monocular_motion_capture(
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
  --smplx-model checkpoints/body_models/smplx/SMPLX_NEUTRAL.npz \
  --model-version 1.1 \
  --video-checkpoint-sha256 2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5 \
  --original-fps 30
```

Only valid frames are replayed. Camera/world geometry remains explicitly marked
as `licensed_smplx_replay`, and missing world parameters are never synthesized.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| No registered public task | Not applicable | See the capability boundary |

<!-- MOTIUS_CANONICAL_METRICS:END -->


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

## Motion Representation

PromptHMR natively emits per-frame SMPL-X body, hand, face, shape, and
translation parameters in camera space. World-space fields are retained only
when the official runtime supplies `smplx_world` or `camera_world`. Licensed
SMPL-X replay materializes joints and 10,475-vertex meshes without embedding the
body model in the result artifact; valid-frame masks remain part of the public
output contract.

## Citation and License

PromptHMR is described in the
[CVPR 2025 paper](https://arxiv.org/abs/2504.06397) and released through the
[official repository](https://github.com/yufu-wang/PromptHMR). Its code,
checkpoints, and derivative runtimes are restricted to the uses allowed by the
upstream license. Motius does not vendor or redistribute them. SMPL-X and the
detector, tracker, depth, and SLAM dependencies retain separate terms; see
[`motius/models/prompthmr/ATTRIBUTIONS.md`](../../motius/models/prompthmr/ATTRIBUTIONS.md)
before setup or publication.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
