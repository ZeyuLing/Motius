<h1 align="center">FlowHMR Model Card</h1>

<p align="center"><strong>491-D SMPL-H motion capture from monocular RGB video, with native flow-matching training.</strong></p>

<p align="center">
  <a href="https://flowhmr.github.io/">Project / Paper updates</a> ·
  <a href="https://github.com/flowhmr/flowhmr">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-FlowHMR-Base">Base Checkpoint</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-FlowHMR-Latest">Latest Checkpoint</a>
</p>

Official source: [FlowHMR](https://github.com/flowhmr/flowhmr), revision
`f12e6a2d46a63d771a66dbb6b5598a1be65b9eea`. Only the formal 0.46B,
491-D release is supported. The retired HYMotionV2M model is not included.

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
| Monocular Motion Capture | Monocular RGB video | <video src="https://github.com/user-attachments/assets/791ec4c4-638d-478c-b79d-7ddb09d3582b" controls></video> | [MP4](https://github.com/user-attachments/assets/791ec4c4-638d-478c-b79d-7ddb09d3582b) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Base | Latest |
| --- | --- |
| <video src="https://github.com/user-attachments/assets/791ec4c4-638d-478c-b79d-7ddb09d3582b" controls></video> | <video src="https://github.com/user-attachments/assets/ca588fd7-52c8-49a8-af11-e0def808a715" controls></video> |

Real-video previews are generated with the public Motius pipeline, seed 0,
20 Euler steps, CFG 1 and the official first-frame floor alignment.

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
| Training motion | 30 fps, 360-frame windows |
| Public preview | 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Variant | Official checkpoint SHA256 | Use |
| --- | --- | --- |
| Base | `77b903ba545669bafd6f7709c0f1748f5fa030a4f25f67392c8c082a21f5e672` | Base flow-matching training and inference |
| Latest | `b7eb06b119feaf5ad7287bafdb2fff38d42dfc22ea9d256f8cdd2ada26dfc53a` | Inference after upstream PHC+ GRPO post-training |

Both artifacts contain inference safetensors, configuration, motion statistics,
YOLOX, VGGT-Omega, SAM-3D-Body and MHR frontend weights, source provenance and
component licenses. Licensed SMPL-H and the SMPL joint regressor are supplied
separately. Network code ships inside Motius; no external checkout is needed.
This release implements base training; PHC+ GRPO training is outside its scope.

## Quick Start

```bash
pip install -e '.[flowhmr,video-to-motion]'
```

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-FlowHMR-Latest",  # or ZeyuLing/Motius-FlowHMR-Base
    bundle_kwargs=dict(
        body_model_path="outputs/checkpoints/flowhmr/body_models/smplh/neutral/model.npz",
        j_regressor_path="outputs/checkpoints/flowhmr/body_models/smpl_neutral_J_regressor.pt",
        device="cuda",
    ),
)
result = pipe.infer_monocular_motion_capture(
    "input.mp4", work_dir="outputs/inference/flowhmr", seeds=[0],
)
```

Provide the official AMASS neutral SMPL-H model (52 joints, 16 shape parameters).
`infer_from_feature(tokens, camera_RT)` accepts precomputed `(T,3072)` tokens
and OpenCV world-to-camera transforms at 30 fps. `infer_v2m` transcodes video,
tracks a person with YOLOX, predicts VGGT cameras and extracts SAM features.
`postprocess=True` enables upstream contact/IK cleanup for one seed.
The CLI is `tools/infer_flowhmr.py`; local artifact export is
`tools/export_flowhmr_hf.py`. No assets are downloaded implicitly during training.

### Training


**Data.** The official training corpus is not public. Use your own official-format
WebDataset shards: each sample groups `motion.npz`, `camera.npz`, `bbox.npz`,
`feature.pt` and metadata as specified in the
[upstream data guide](https://github.com/flowhmr/flowhmr/blob/main/docs/resources.md).
The adapter uses upstream camera/world conversion, SMPL-H FK, 491-D encoding,
360-frame crop/padding, and 10% feature dropout. It keeps all hand joints.
`FlowHMRDataset` also accepts the upstream `roots` list for directory samples.
Replacement data constitute a new training run, not an exact dataset reproduction.

**Precision.** FP32 (`mixed_precision='no'`), matching the official base launch.

**Objective.** Official x1 flow matching with logit-normal time sampling
(mean −0.8, std 0.8), SmoothL1 losses for root XZ velocity, root Y position,
local joint positions, rotations, shape and contact, plus FK rotation and
consistency losses. Weights are 10/10/10/10/1/4/5/5. Loss masks exclude padding.
The native runner manages AdamW, gradient clipping (10), cosine LR
1e-4 → 1e-5, and checkpoint state. The public recipe uses 250,000 iterations,
equivalent to the official 25 epochs of 10,000 iterations. Eight processes give
the official global batch size 64; another process count changes the recipe.

```bash
export MOTIUS_SMPLH_MODEL=outputs/checkpoints/flowhmr/body_models/smplh/neutral/model.npz
export MOTIUS_J_REGRESSOR=outputs/checkpoints/flowhmr/body_models/smpl_neutral_J_regressor.pt
export MOTIUS_FLOWHMR_SHARDS='data/flowhmr/train-{00000..00025}.tar'
accelerate launch --num_processes 8 tools/train.py configs/flowhmr/train_flowhmr.py
# Resume model, optimizer, scheduler and runner state:
accelerate launch --num_processes 8 tools/train.py configs/flowhmr/train_flowhmr.py --auto-resume --load-scope full
```

**Checkpoints.** Saved every 10,000 steps, five retained, plus final state under
`outputs/training/flowhmr`. `MOTIUS_PRETRAINED_WEIGHTS` optionally initializes from
an official `.ckpt`; it is not a full-state resume. WebDataset is resampled and
worker/rank partitioned; resume restores training state, not the exact stream
cursor. `bundle.save_pretrained(directory)` exports safetensors, config,
statistics, provenance and licenses for `Pipeline.from_pretrained`.

This native recipe covers base pretraining. The official GRPO SDE primitives
are retained in the vendor, but PHC+ simulator post-training is not advertised
as a Motius-native training recipe in this release.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Monocular Motion Capture | [Published results](../leaderboards/hf_space_monocular_capture/monocular_capture_results.json) | 3DPW camera/world-space capture metrics |

<!-- MOTIUS_CANONICAL_METRICS:END -->


### Source parity

The [validation record](flowhmr_validation.json) contains checkpoint hashes,
Hub revisions, numerical comparison results and the test boundaries.

The full 0.46B base network was compared against independently imported pinned
upstream classes using the licensed 6890-vertex SMPL-H model. For identical
inputs and RNG states, the training loss, every component loss and 304 gradient
tensors matched exactly (`rtol=0, atol=0`). Both base and latest four-frame,
20-step ODE outputs matched exactly for all nine returned tensor fields.
A 30-frame real-video comparison also matched detection/tracking boxes, VGGT
intrinsics/extrinsics, SAM features, base/latest sampling, floor alignment and
FK joints exactly. Both Hub artifacts were downloaded into a fresh cache and
reloaded through `Pipeline.from_pretrained`; every output matched the local
export exactly.

These checks establish numerical equivalence for the tested inputs, not every
possible input or reproduction of a complete training run.

Small-network tests cover padding, strict checkpoint loading, serialization,
two-process DDP synchronization and official-format WebDataset preprocessing.
The native runner completed two training steps and resumed the full model,
optimizer and scheduler state for the third step.

### Upstream reported results

| Variant | PHC+ tracking success on Wild-4K |
| --- | --- |
| Base | 78.79% |
| Latest | 82.47% |

Source: [pinned upstream Model Zoo](https://github.com/flowhmr/flowhmr/blob/f12e6a2d46a63d771a66dbb6b5598a1be65b9eea/README.md#model-zoo).
These are upstream simulation results, not Motius measurements or joint-error
scores. The training corpus is not public. No full-scale training or Motius
leaderboard result is claimed.

```bash
MOTIUS_FLOWHMR_UPSTREAM=/path/to/pinned/flowhmr pytest -q tests/test_flowhmr.py
python tools/audit_training_release.py
python tools/audit_training_docs.py
```

## Motion Representation

The native 491-D state contains smooth root XZ velocity and Y position, 52 local
joint positions, 52 six-dimensional rotations, 16 shape coefficients and four
foot-contact channels. Video conditions contain 3072 SAM features and nine
camera-rotation values. The native clock is 30 fps and training windows are
360 frames. Hands remain in the native output; the shared capture exposes the
standard 22 body joints and preserves all native rotations.

The shared result contains Y-up world joints and SMPL-H parameters. Camera-space
joints and metric camera-to-motion-world extrinsics are unavailable: VGGT camera
translation is not metrically aligned with the independently reconstructed
motion world. Output availability explicitly records this boundary.

## Citation and License

This implementation originates from [FlowHMR](https://github.com/flowhmr/flowhmr).
Follow the upstream project for publication and citation updates; no publication
identifier is invented for this release. FlowHMR permits educational, research
and non-profit use only. Its terms apply independently of the framework license.

See [ATTRIBUTIONS](../../motius/models/flowhmr/ATTRIBUTIONS.md),
[FlowHMR license](../../motius/models/flowhmr/vendor/UPSTREAM_LICENSE),
[SAM license](../../motius/models/flowhmr/vendor/SAM3D_LICENSE),
[VGGT-Omega license](../../motius/models/flowhmr/vendor/VGGT_OMEGA_LICENSE),
[DINOv3 license](../../motius/models/flowhmr/vendor/DINOV3_LICENSE.md),
[YOLOX license](../../motius/models/flowhmr/vendor/YOLOX_LICENSE) and
[MHR license](../../motius/models/flowhmr/vendor/MHR_LICENSE).
SMPL-H assets require their separate license and are not redistributed.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
