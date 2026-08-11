---
library_name: motius
pipeline_tag: video-to-video
license: other
tags:
  - human-motion
  - monocular-motion-capture
  - smpl
---

<h1 align="center">GEM-SMPL Model Card</h1>

<p align="center">
  <strong>World-grounded SMPL motion recovery from monocular RGB video.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2505.01425">Paper</a> ·
  <a href="https://github.com/NVlabs/GENMO">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-GEM-SMPL">Motius Checkpoint</a>
</p>

GEM-SMPL is the SMPL video-motion-estimation release of **GEM: A Generalist
Model for Human Motion**, originally released as GENMO. The Official source is
NVlabs/GENMO at revision
`16bebf402d8893184249ee206d957b8248cd8310`; the released checkpoint SHA-256 is
`1d15cbe2864d6de61a75e83fdbfe83bec3c7b183eee3d3dcdbd9107e4456454a`.

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
| Monocular Motion Capture | Monocular RGB video | <video src="https://github.com/user-attachments/assets/6895768a-a58a-48ef-805c-8bdcebe523cc" controls></video> | [MP4](https://github.com/user-attachments/assets/6895768a-a58a-48ef-805c-8bdcebe523cc) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


This 768px, 30 FPS preview renders the world-space SMPL vertices returned by
the public Motius pipeline.

<video src="https://github.com/user-attachments/assets/6895768a-a58a-48ef-805c-8bdcebe523cc" controls></video>

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

The Motius Hugging Face artifact contains the exact official GEM-SMPL,
HMR2, ViTPose, and YOLO checkpoint bytes and a manifest with every SHA-256.
The runtime source is shipped inside the `motius` wheel at its pinned revision;
inference never imports another repository checkout.

SMPL and SMPL-X files are license-gated and are not redistributed. Download
them into `checkpoints/body_models/` using
[`checkpoints/body_models/README.md`](https://github.com/ZeyuLing/Motius/blob/main/checkpoints/body_models/README.md).

## Quick Start

Create an isolated environment without cloning GENMO:

```bash
python3.10 -m venv outputs/envs/gem-smpl
outputs/envs/gem-smpl/bin/pip install -e ".[gem-smpl]"
```

Run the standard task API:

```python
from pathlib import Path

from motius.motion.representation.monocular_capture import (
    save_monocular_capture_result,
)
from motius import Pipeline

pipeline = Pipeline.from_pretrained(
    "ZeyuLing/Motius-GEM-SMPL",
    bundle_kwargs={
        "python_executable": "outputs/envs/gem-smpl/bin/python",
        "body_models_root": "checkpoints/body_models",
    },
)
result = pipeline.infer_monocular_motion_capture(
    "input.mp4",
    output_root="outputs/gem_smpl/run_001",
    materialize_geometry=True,
    render=True,
)
save_monocular_capture_result(
    result,
    Path("outputs/gem_smpl/run_001/result.npz"),
)
```

`render=True` requests the upstream in-camera/world previews. It can be left
off for evaluation and batch inference.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Monocular Motion Capture | [Published results](../leaderboards/hf_space_monocular_capture/monocular_capture_results.json) | 3DPW camera/world-space capture metrics |

### Canonical Monocular-Capture Snapshot

| Protocol | Coverage (%) | PA-MPJPE (mm) | MPJPE (mm) | PVE (mm) | Accel (m/s²) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3dpw_test_camera_v1 | 100.0000 | 46.4470 | 64.4566 | — | 5.7129 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: `3dpw_test_camera_v1`, one inference item per official 3DPW person
track using the released target-crop protocol.

| Coverage | MPJPE ↓ | PA-MPJPE ↓ | Acceleration ↓ |
| ---: | ---: | ---: | ---: |
| 100.00% | 64.46 mm | 46.45 mm | 5.713 m/s² |

### Stage Parity

The migration gate replays the same video, checkpoint bytes, licensed body
models, precomputed visual tensors, and random seed through the pinned official
source and the Motius package.

| Boundary | Fields | Requirement | Result |
| --- | ---: | --- | --- |
| Tracking | 1 | exact | pass |
| Keypoints | 1 | exact | pass |
| Visual features | 2 | exact | pass |
| Complete model input | 15 | exact | pass |
| Network and decoded output | 9 | exact | pass |
| SMPL geometry | 4 | exact | pass |
| Public result | 10 | exact | pass |
| **Total** | **42** | **`rtol=0`, `atol=0`** | **pass** |

```bash
python tools/verify_monocular_pipeline_parity.py \
  --reference outputs/parity/gem_smpl/reference_trace.npz \
  --candidate outputs/parity/gem_smpl/motius_trace.npz
```

## Motion Representation

GEM-SMPL predicts camera-space and gravity-aligned global body parameters.
Motius preserves the native 21-joint body pose, root orientation, translation,
and ten shape coefficients. Geometry materialization exposes:

- SMPL-24 named joints;
- 6,890-vertex SMPL meshes;
- camera and world root trajectories;
- per-frame camera intrinsics;
- the source video clock without temporal resampling.

The internal SMPL-X body layer is converted with the same fixed sparse
SMPL-X-to-SMPL map used by the pinned implementation.

## Citation and License

The vendored source retains the NVIDIA OneWay Noncommercial license. Public
weights retain the NVIDIA Open Model License. SMPL and SMPL-X have separate
terms. See the
[GEM-SMPL attributions](https://github.com/ZeyuLing/Motius/blob/main/motius/models/gem_smpl/ATTRIBUTIONS.md)
and the packaged license files before use.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
