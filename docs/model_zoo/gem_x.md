---
library_name: motius
pipeline_tag: video-to-video
license: other
tags:
  - human-motion
  - monocular-motion-capture
  - soma-x
---

<h1 align="center">GEM-X Model Card</h1>

<p align="center">
  <strong>Whole-body SOMA-X motion recovery from monocular RGB video.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.01425">Paper</a> ·
  <a href="https://github.com/NVlabs/GEM-X">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-GEM-X">Motius Checkpoint</a>
</p>

GEM-X is NVIDIA's monocular whole-body motion estimator built around the
SOMA-X parametric body model. The Official source is NVlabs/GEM-X at revision
`32992550dba114c62243fb55e361311972dce8f9`; SOMA-X is pinned to
`e0f8ff0ecfa3edbbb6058b1e0f08822ee2f84ee5`, and the released checkpoint
SHA-256 is
`4c1f85ca8c1e11e6588aead49fbc024bf660708def670043e0b537c101ee298e`.

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
| Monocular Motion Capture | Monocular RGB video | <video src="https://github.com/user-attachments/assets/5759b5ca-d4fd-45de-acfd-ca3b58109dd1" controls></video> | [MP4](https://github.com/user-attachments/assets/5759b5ca-d4fd-45de-acfd-ca3b58109dd1) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


This 768px, 30 FPS preview renders the world-space SOMA-X mesh returned by the
public Motius pipeline.

<video src="https://github.com/user-attachments/assets/5759b5ca-d4fd-45de-acfd-ca3b58109dd1" controls></video>

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

### Checkpoint

The Motius Hugging Face artifact is complete: GEM-X, SAM-3D-Body, DINOv3,
ViTPose, YOLOX, MHR, SOMA-X identity/corrective assets, and normalization
statistics are stored under their expected paths with SHA-256 provenance.
`Pipeline.from_pretrained` therefore needs no source checkout or second model
download.

## Quick Start

Create an isolated environment without cloning GEM-X:

```bash
python3.10 -m venv outputs/envs/gem-x
outputs/envs/gem-x/bin/pip install -e ".[gem-x]"
```

Run the standard task API:

```python
from pathlib import Path

from motius.motion.representation.monocular_capture import (
    save_monocular_capture_result,
)
from motius import Pipeline

pipeline = Pipeline.from_pretrained(
    "ZeyuLing/Motius-GEM-X",
    bundle_kwargs={
        "python_executable": "outputs/envs/gem-x/bin/python",
    },
)
result = pipeline.infer_monocular_motion_capture(
    "input.mp4",
    output_root="outputs/gem_x/run_001",
    materialize_geometry=True,
    render=True,
)
save_monocular_capture_result(
    result,
    Path("outputs/gem_x/run_001/result.npz"),
)
```

`render=True` requests the upstream keypoint, in-camera, and world previews.
Leave it off for evaluation and batch inference.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Monocular Motion Capture | [Published results](../leaderboards/hf_space_monocular_capture/monocular_capture_results.json) | 3DPW camera/world-space capture metrics |

### Canonical Monocular-Capture Snapshot

| Protocol | Coverage (%) | PA-MPJPE (mm) | MPJPE (mm) | PVE (mm) | Accel (m/s²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3dpw_test_camera_v1 | 100.0000 | 53.1974 | 84.3806 | — | 5.6159 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: `3dpw_test_camera_v1`, all 24 test videos and 37 official person
tracks, evaluated on `common_hmr15_named_v1`.

| Coverage | MPJPE ↓ | PA-MPJPE ↓ | Acceleration ↓ |
| ---: | ---: | ---: | ---: |
| 100.00% | 84.38 mm | 53.20 mm | 5.616 m/s² |

The official demo emits an identity camera trajectory when no external visual
odometry is supplied. These are camera-space metrics; world-space ranking is
unavailable for that run.

### Stage Parity

The strict gate compares all persisted boundaries. Deterministic fields are
bitwise exact. The official contact IK/SOMA CUDA postprocess is non-deterministic
across independent processes, so only its 15 explicitly named descendants use
a hard `3e-6` absolute-error ceiling. No global tolerance is applied.

| Boundary | Fields | Requirement | Result |
| --- | ---: | --- | --- |
| Tracking | 2 | exact | pass |
| Keypoints and camera | 2 | exact | pass |
| Visual features | 2 | exact | pass |
| Complete model input | 18 | exact | pass |
| Raw network output and deterministic decoded fields | 27 | exact | pass |
| Contact-postprocessed model fields | 7 | `atol ≤ 3e-6` | max `6.71e-7` |
| SOMA geometry | 4 | `atol ≤ 3e-6` | max `2.38e-6` |
| Public result | 9 | exact except 4 descendants | max `9.54e-7` |
| **Total** | **71** | **field-scoped policy** | **pass** |

```bash
python tools/verify_monocular_pipeline_parity.py \
  --profile gem-x \
  --reference outputs/parity/gem_x/reference_trace.npz \
  --candidate outputs/parity/gem_x/motius_trace.npz
```

## Motion Representation

GEM-X natively predicts SOMA-77:

- 77-joint axis-angle pose;
- 45 identity coefficients;
- 69 global/body-part scale parameters;
- camera and world root translations;
- 77 named joints and 4,505-vertex low-LOD SOMA meshes.

Motius keeps SOMA-X native. It does not manufacture SMPL vertices from a
different topology. Cross-model 3DPW evaluation uses the audited
`common_hmr15_named_v1` joint subset; PVE is reported as unavailable.

## Citation and License

GEM-X source is Apache-2.0 and the public weights use the NVIDIA Open Model
License. SAM-3D-Body, DINOv3, SOMA-X, MHR, and their assets retain their own
terms. See the
[GEM-X attributions](https://github.com/ZeyuLing/Motius/blob/main/motius/models/gem_x/ATTRIBUTIONS.md)
and the packaged third-party notices before use.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
