<h1 align="center">GVHMR Model Card</h1>

<p align="center">
  <strong>Camera-relative and world-grounded SMPL motion from monocular RGB video.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2409.06662">Paper</a> ·
  <a href="https://github.com/zju3dv/GVHMR">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-GVHMR">Motius Checkpoint</a>
</p>

GVHMR recovers camera-relative and world-grounded human motion from monocular
RGB video through gravity-view coordinates. The Official source is zju3dv/GVHMR
at revision `6ec3ca39336c50492c0fae65fba2fb831fc7d866`; the released checkpoint
SHA-256 is
`4fae7da2de388d5da3514cb27a2d003f364dacb280e9cf88972b710e589c6b91`.

**Tasks:** Monocular Motion Capture

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
| Monocular Motion Capture | Monocular RGB video | <video src="https://github.com/user-attachments/assets/a2547fb6-f4a0-4b27-a758-f274d96d60ab" controls></video> | [MP4](https://github.com/user-attachments/assets/a2547fb6-f4a0-4b27-a758-f274d96d60ab) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


All previews below were produced through the public Motius pipeline at
812×720 and 30 FPS. The source is the public tennis example shipped by GVHMR;
each column is a different temporal clip.

| Clip 01 | Clip 02 | Clip 03 |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/a2547fb6-f4a0-4b27-a758-f274d96d60ab" controls></video> | <video src="https://github.com/user-attachments/assets/63e4ccdf-73ee-4f54-af99-0facacb1e3d8" controls></video> | <video src="https://github.com/user-attachments/assets/7534f4e5-9325-4fc2-88b0-303c74413181" controls></video> |

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Monocular Motion Capture | `infer_monocular_motion_capture` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/monocular-motion-capture-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 30 fps temporal clips |
| Public preview | 30 fps native model clock |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Task | Public API | Input | Output |
| --- | --- | --- | --- |
| Monocular Motion Capture | `infer_monocular_motion_capture` | RGB video | `MonocularCaptureResult` |

## Quick Start

Create the pinned CUDA 12.1 environment. The setup installs Motius itself and
does not clone another source repository:

```bash
bash tools/setup_gvhmr_env.sh outputs/envs/gvhmr
```

Download licensed SMPL and SMPL-X files as described in
[`checkpoints/body_models/README.md`](../../checkpoints/body_models/README.md).
The Hugging Face artifact already contains GVHMR, HMR2, ViTPose, and YOLO
checkpoints with their original bytes and filenames.

```python
from pathlib import Path

from motius.motion.representation.monocular_capture import (
    save_monocular_capture_result,
)
from motius import Pipeline

pipeline = Pipeline.from_pretrained(
    "ZeyuLing/Motius-GVHMR",
    bundle_kwargs={
        "python_executable": "outputs/envs/gvhmr/bin/python",
        "body_models_root": "checkpoints/body_models",
    },
)
result = pipeline.infer_monocular_motion_capture(
    "input.mp4",
    output_root="outputs/gvhmr/run_001",
    materialize_geometry=True,
)
save_monocular_capture_result(
    result,
    Path("outputs/gvhmr/run_001/result.npz"),
)
```

Pass a dense `(frames, 4)` `bbox_xyxy` array to evaluate a specific person.
Motius writes the official `bbx.pt` cache and applies GVHMR's native 192:256,
1.2x crop conversion. Set `render=True` to produce the in-camera and
world-grounded SMPL videos.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Monocular Motion Capture | [Published results](../leaderboards/hf_space_monocular_capture/monocular_capture_results.json) | 3DPW camera/world-space capture metrics |

### Canonical Monocular-Capture Snapshot

| Protocol | Coverage (%) | PA-MPJPE (mm) | MPJPE (mm) | PVE (mm) | Accel (m/s²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3dpw_test_camera_v1 | 100.0000 | 47.6594 | 62.6466 | — | 5.4721 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


### 3DPW Test

Protocol: `3dpw_test_camera_v1`, one inference item per official person track.
The target-crop run covers all 24 test sequences and all 37 person tracks.
Missing 2D annotations are interpolated only to form dense crops and remain
excluded from metrics.

| Protocol | Coverage | MPJPE ↓ | PA-MPJPE ↓ | Acceleration ↓ |
| --- | ---: | ---: | ---: | ---: |
| Official per-target crop | 100.00% | 62.65 mm | 47.66 mm | 5.47 m/s² |

The detector-driven single-track diagnostic covers 65.35% of valid
person-frames and obtains 86.85 mm MPJPE, 59.35 mm PA-MPJPE, and 7.19 m/s²
acceleration error. It is diagnostic only and is not used for complete-split
ranking.

### Exact Parity

The migration gate uses the same video, official checkpoint bytes, body
models, and Python environment for the pinned official runner and Motius:

| Boundary | Compared fields | Strict result |
| --- | ---: | ---: |
| Tracking and crop conversion | 2 | exact |
| Camera intrinsics and angular velocity | 2 | exact |
| ViTPose keypoints and HMR2 image features | 2 | exact |
| Complete GVHMR model input | 6 | exact |
| Network and decoded SMPL outputs | 27 | exact |
| **Total** | **39** | **`rtol=0`, `atol=0`** |

The strict comparison replays the official persisted preprocessing tensors to
isolate source migration from CUDA kernel nondeterminism. In two independent
end-to-end runs, tracking, HMR2 features, and SimpleVO remain bitwise exact;
ViTPose differs by at most `6.1035e-5`, and the largest final SMPL parameter
difference is `1.0586e-4`. These values describe repeatability of the same
official CUDA implementation, not a Motius algorithm change.

Reproduce the strict check with:

```bash
python tools/trace_gvhmr_run.py \
  --run-dir outputs/gvhmr/official/clip \
  --output outputs/gvhmr/official_trace.npz \
  --name gvhmr-official
python tools/verify_monocular_pipeline_parity.py \
  --reference outputs/gvhmr/official_trace.npz \
  --candidate outputs/gvhmr/motius_trace.npz
```

## Motion Representation

GVHMR natively predicts per-frame SMPL-X body parameters in camera and
gravity-aligned world coordinates. Motius preserves the native camera/world
parameters, then applies GVHMR's fixed sparse SMPL-X-to-SMPL mapping to expose
SMPL-24 joints and 6,890-vertex SMPL meshes. The output clock is 30 FPS, which
matches the official demo protocol.

The official result does not expose a complete camera-to-world transform.
Motius therefore leaves `camera_to_world` unset instead of constructing an
unverifiable transform.

## Citation and License

The vendored GVHMR inference source and checkpoints retain the upstream
educational, research, and non-profit terms. SMPL and SMPL-X body models are
not redistributed and must be supplied by the user. See
[`ATTRIBUTIONS.md`](../../motius/models/gvhmr/ATTRIBUTIONS.md) and the packaged
`GVHMR_LICENSE` before use.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
