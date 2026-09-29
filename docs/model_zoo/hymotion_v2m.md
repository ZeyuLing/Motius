<h1 align="center">HYMotion-V2M Model Card</h1>

<p align="center">
  <strong>Global SMPL-H motion recovery from tracked monocular RGB video.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.23464">Paper</a> ·
  <a href="https://github.com/Tencent-Hunyuan/HY-Motion-1.0">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-HYMotion-V2M">Motius Checkpoint</a>
</p>

HYMotion-V2M combines YOLOX/ByteTrack and SAM-3D-Body image features with a
camera-conditioned flow-matching model. The Official source is
Tencent-Hunyuan/HY-Motion-1.0. Motius packages runtime revision
`motius_hymotion_v2m_release_v1`; the source checkpoint SHA-256 is
`ed301af6b6fc6bc22dc69d8c7f48c1d6b2fff31d48a1717c8f5bc5ea92bc71df`.

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
| Monocular Motion Capture | Monocular RGB video | <video src="https://github.com/user-attachments/assets/49feb171-d3d9-45b1-8ada-7746680e6a28" controls></video> | [MP4](https://github.com/user-attachments/assets/49feb171-d3d9-45b1-8ada-7746680e6a28) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


The inline preview replays the native 52-joint SMPL-H output stored by the
exact five-frame end-to-end parity trace. It avoids adding a body-model
materialization step to this checkpoint parity diagnostic. HYMotion-V2M
remains **unranked** because the short trace is not a leaderboard-eligible
monocular-capture sequence.

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
| Training motion | 30 fps video and motion clips |
| Public preview | 30 fps native model clock |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Task | Public API | Input | Output |
| --- | --- | --- | --- |
| Monocular Motion Capture | `infer_monocular_motion_capture` | RGB video | `MonocularCaptureResult` |

HYMotion-V2M is not listed on the public monocular-motion leaderboard yet. Its
current static-camera diagnostic is useful for integration validation but is
not comparable with methods that recover the camera trajectory.

## Quick Start

Install the V2M dependencies and provide a licensed neutral SMPL-H body model:

```bash
python -m pip install -e ".[video-to-motion]"
```

```python
from pathlib import Path

from motius.motion.representation.monocular_capture import (
    save_monocular_capture_result,
)
from motius import Pipeline

pipeline = Pipeline.from_pretrained(
    "ZeyuLing/Motius-HYMotion-V2M",
    bundle_kwargs={
        "body_model_path": (
            "checkpoints/body_models/smplh/neutral/model.npz"
        ),
        "device": "cuda",
    },
)
result = pipeline.infer_monocular_motion_capture(
    "input.mp4",
    work_dir="outputs/hymotion_v2m/run_001",
)
save_monocular_capture_result(
    result,
    Path("outputs/hymotion_v2m/run_001/result.npz"),
)
```

The Hugging Face artifact contains the generator, normalization statistics,
SAM-3D-Body/MHR assets, and YOLOX weights. No external source checkout is used.
The licensed SMPL-H body model remains user supplied.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Monocular Motion Capture | [Published results](../leaderboards/hf_space_monocular_capture/monocular_capture_results.json) | 3DPW camera/world-space capture metrics |

<!-- MOTIUS_CANONICAL_METRICS:END -->


The target-conditioned diagnostic covers all 24 3DPW Test videos and all 37
official person tracks. Each track uses its dense official 2D crop; outputs
retain the exact source duration.

| Protocol | Coverage | MPJPE ↓ | PA-MPJPE ↓ | Acceleration ↓ |
| --- | ---: | ---: | ---: | ---: |
| 3DPW Test camera, per-target crop | 100.00% | 270.65 mm | 139.25 mm | 6.118 m/s² |

The camera-space metrics above are not ranking eligible because this run uses
an identity camera trajectory when no video camera estimator is available.

### Exact Parity

The release gate records every named stage as a pickle-free NPZ trace and
requires exact element-wise equality (`rtol=0`, `atol=0`):

| Comparison | Shape or fields | Maximum absolute error |
| --- | ---: | ---: |
| YOLOX raw detector logits | `(1, 8400, 85)` | `0` |
| SAM-3D-Body image token | `(3072,)` | `0` |
| Generation, stitching, FK, and grounding | 34 fields | `0` |

The full public path records tracking, camera conditioning, visual features,
model conditioning, every inference window, stitched motion, SMPL-H forward
kinematics, ground alignment, native output, and `MonocularCaptureResult`.

## Motion Representation

The native prediction contains SMPL-H rotations for 52 joints, global
translation, ten shape coefficients, and 30 FPS timing. Motius retains this
native output and also exposes the shared body-22 joints/mesh contract. It does
not relabel SMPL-H as SMPL, and it reports camera/world availability explicitly.

## Citation and License

HY-Motion, SAM-3D-Body, DINOv3, YOLOX, and SMPL-H retain their respective
licenses. The checkpoint artifact includes the applicable license and
attribution files. Review them before use or redistribution.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
