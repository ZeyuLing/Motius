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
native image and temporal neural bundles, a native video Pipeline, temporal
Trainer, licensed SMPL-X geometry replay, and evaluation for local research use.
Official source: [yufu-wang/PromptHMR](https://github.com/yufu-wang/PromptHMR),
pinned to `3b566b7dbb28ce506c7ea972c18693f4c705ce8c`. The default BEDLAM1+BEDLAM2
checkpoint has SHA-256
`2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5`.

| Task status | Native research implementation; validation scope below |
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
| Monocular capture | RGB video | Local research output | — |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


The historical 3DPW result below used the official installation and licensed
SMPL-X replay. It has not yet been recomputed with the native implementation.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Monocular capture | `infer_monocular_motion_capture` | Source comparison scope below |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | No single fixed motion clock; the image model is trained frame-wise |
| Public preview | Input-video clock unless `output_fps` is requested |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


`PromptHMRBundle` owns six neural roots: image encoder, camera encoder, text
encoder, prompt encoder, SMPL decoder and temporal network. Detection, mask
tracking, camera calibration, DROID and Metric3D use separate neural bundles.
Video processing, prompt construction, metric SLAM, body geometry, floor fitting
and post-optimization belong to `PromptHMRPipeline` and motion utilities.
Inference runs in process without an original checkout or original demo command.
`PromptHMRVideoTrainer` implements the released temporal objective through the
same temporal network. Image training code/dataset was not published upstream.

## Quick Start

Install `motius[prompthmr]` in the validated PyTorch 2.6 / CUDA 12.6
environment. The original compiled SAM 2, Detectron2, DROID-Calib, lietorch and
PyG torch-scatter wheels must match that stack. FFmpeg/ffprobe handle video I/O.
Only the needed native components are imported at inference; the original
application repository is not a runtime dependency.

Supply a local artifact directory with `checkpoint.ckpt`, image `config.yaml`,
`video/phmr_b1b2.ckpt`, `video/prhmr_release_002.yaml`, the original frontend
weights under `data/pretrain/`, and the public SMPL-X mapping files under
`data/body_models/`. The body models are provided separately by the user.
A native `save_pretrained` export stores current neural parameters and the
required pipeline assets, without copying source repositories or licensed
SMPL/SMPL-X model files.

```python
from motius.pipelines.prompthmr import PromptHMRPipeline

pipeline = PromptHMRPipeline.from_pretrained(
    "checkpoints/prompthmr",
    body_models_root="checkpoints/body_models",
)
result = pipeline.infer_monocular_motion_capture(
    "input.mp4", output_dir="outputs/prompthmr", static_camera=False,
    deterministic=True, seed=1234,
)
pipeline.save_pretrained("outputs/prompthmr_native")
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
| Monocular capture | Historical official 3DPW run below | Native full-dataset rerun pending |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Native validation uses the independent pinned original checkout and identical
checkpoints. Prepared image/body/temporal outputs (96), frontend outputs (21),
DROID neural outputs (20), Metric3D recurrent outputs (58), full masked SLAM
trajectory/BA outputs, static-video results (181), moving-video results (179),
and temporal training losses/all active gradients (520) compare bitwise equal.
The moving-camera comparison enables the same Torch deterministic algorithms
and cuDNN settings in both workers; unconstrained CUDA SLAM aggregation varies
between runs. Use `deterministic=True` and the same seed for a reproducible run.
These are one-video/prepared-batch certificates, not new full-dataset metrics.
Current-weight export restores all 986 neural tensors exactly; cold Auto Pipeline
video inference and actual two-process temporal DDP also passed. Reports are in
[the validation directory](../architecture/validation/).

The following historical benchmark used the original application:

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
upstream license. Local derivatives preserve those restrictions. SMPL-X and the
detector, tracker, depth, and SLAM dependencies retain separate terms; see
[`motius/licenses/prompthmr/ATTRIBUTIONS.md`](../../motius/licenses/prompthmr/ATTRIBUTIONS.md)
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
