<h1 align="center">HYMotion M2M (MotionCanvas) Model Card</h1>

<p align="center">
  <strong>One 360-frame canvas for generation, completion, spatial control, editing, and repair.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2512.23464">HY-Motion Backbone Paper</a> ·
  <a href="https://github.com/ZeyuLing/Motius">Motius Source</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MotionCanvas-0.46B">Motius Checkpoint</a>
</p>

HYMotion M2M, released as MotionCanvas, is a Motius-native motion-to-motion
model. It extends the HY-Motion
flow-matching backbone with clean motion imputation, arbitrary channel masks,
edit context, sparse-rollout training, and position-constraint projection. The
released implementation, trainer, data pipeline, text encoders, normalization
statistics, bone offsets, and inference pipeline run without another source
checkout.

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
| Text-to-Motion | person walking at a average pace forward, swaying arms and torso with a sense of swagger | <video src="https://github.com/user-attachments/assets/743d0fb5-6494-4c3c-97e7-76ba72d8d0d0" controls></video> | [MP4](https://github.com/user-attachments/assets/743d0fb5-6494-4c3c-97e7-76ba72d8d0d0) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motioncanvas&case=004822) |
| Temporal Motion Completion | Complete a natural walk through the observed footsteps.<br><sub>Sparse foot positions and facing keyframes</sub> | <video src="https://github.com/user-attachments/assets/e36fd6d4-e84a-4c72-a20e-b9a68d3c4c1f" controls></video> | [MP4](https://github.com/user-attachments/assets/e36fd6d4-e84a-4c72-a20e-b9a68d3c4c1f) |
| Kinematic Motion Control | Follow the prescribed root path with natural locomotion.<br><sub>Continuous trajectory plus pelvis-height control</sub> | <video src="https://github.com/user-attachments/assets/cee52e8f-0b1f-4a72-927c-d35b76f0f5fb" controls></video> | [MP4](https://github.com/user-attachments/assets/cee52e8f-0b1f-4a72-927c-d35b76f0f5fb) |
| Motion Editing | Wave with the right hand.<br><sub>Preserve the lower-body motion</sub> | <video src="https://github.com/user-attachments/assets/5cee1790-ee46-4dea-ac91-3e532a58720c" controls></video> | [MP4](https://github.com/user-attachments/assets/5cee1790-ee46-4dea-ac91-3e532a58720c) |
| Motion Repair | Repair drifting, foot sliding, over-smoothing, and jitter while preserving the valid motion. | <video src="https://github.com/user-attachments/assets/67fa7324-7e1a-428c-949e-2757659e1293" controls></video> | [MP4](https://github.com/user-attachments/assets/67fa7324-7e1a-428c-949e-2757659e1293) · [All cases](https://zeyuling-motion-repair-brokenamass-leaderboard.static.hf.space/cases/index.html?method=motioncanvas&case=repair_000) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->

The additional jump-apex edit below exposes source, goal, and generated
trajectory together. All previews are actual Three.js renders with a shared
floor scene at 512 px and 30 fps; the artifact includes the 1080p MP4 sources.

<video src="https://github.com/user-attachments/assets/58f24f63-2b4f-4c83-8883-ec0821f1fd88" controls></video>

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |
| Temporal Motion Completion | `infer_temporal_motion_completion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) |
| Kinematic Motion Control | `infer_kinematic_motion_control` | [Benchmark and examples](../leaderboards/kinematic_motion_control.md) |
| Motion Editing | `infer_motion_editing` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) |
| Motion Repair | `infer_motion_repair` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/motion-repair-brokenamass-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 30 fps on a 360-frame canvas |
| Public preview | 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->

| Item | Value |
| --- | --- |
| Tasks | Text-to-Motion, Temporal Motion Completion, Kinematic Motion Control, Motion Editing, Motion Repair |
| Native representation | MotionCanvas-198, local SMPL-22 rotations, 30 fps |
| Context | 360 frames (12 seconds) |
| Parameters | 0.46B motion transformer |
| Checkpoint | [`ZeyuLing/Motius-MotionCanvas-0.46B`](https://huggingface.co/ZeyuLing/Motius-MotionCanvas-0.46B) |
| Release revision | [`d0b5920`](https://huggingface.co/ZeyuLing/Motius-MotionCanvas-0.46B/tree/d0b5920fc0895a4cd8a05a1ace48015c37dab625) |
| Pipeline | `motius.pipelines.motioncanvas.MotionCanvasPipeline` |
| Release checkpoint | epoch 2354 |

The final training recipe samples 630,000 examples per epoch. Its sampling
mixture is 30% full-mask HYMotion text-to-motion, 60% arbitrary HYMotion
conditions, 5% MotionFix edits, and 5% PerMo edits. Training uses FP32,
batch size 96 per process, AdamW at `1e-5`, and the released sparse-rollout
join objective. The exact configuration is
[`configs/motioncanvas/train_motioncanvas_0p46b.py`](../../configs/motioncanvas/train_motioncanvas_0p46b.py).

## Quick Start

Install the MotionCanvas dependencies and load the complete Hub artifact:

```bash
python -m pip install -e '.[motioncanvas]'
```

```python
import torch
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MotionCanvas-0.46B",
    cache_dir="/path/to/large/huggingface-cache",
    bundle_kwargs={"device": "cuda", "text_dtype": "bf16"},
    num_steps=50,
)

# Native physical-space MotionCanvas-198, shape (B, T, 198).
source = torch.from_numpy(reference_motion)[None].cuda()
generate = torch.zeros(1, source.shape[1], device="cuda")
generate[:, 90:180] = 1  # 1=regenerate, 0=preserve clean evidence

result = pipe.infer_temporal_motion_completion(
    source,
    generate,
    captions="a person turns left and continues walking",
    seed=42,
)
motion_198 = result["motion_198"]
```

The complete artifact is about 20 GB. `cache_dir` is a top-level loader
argument; set `HF_XET_CACHE` as well when the default Xet cache is on a small
system disk.

### Inference Contract

| Task API | Required motion inputs | Optional condition | Primary output |
| --- | --- | --- | --- |
| `infer_text_to_motion` | Caption and output length | seed, guidance scale | generated `motion_198` |
| `infer_temporal_motion_completion` | `source_motion: (B, T, 198)`, `generation_mask: (B, T)` or `(B, T, 198)` | captions, seed, lengths | `motion_198: (B, T, 198)` |
| `infer_kinematic_motion_control` | Same source and mask contract | world-space `PositionConstraint` objects, captions, seed | `motion_198` plus projected `keypoints3d` |
| `infer_motion_editing` | Same source and mask contract | edit caption, seed | edited `motion_198` with preserved cells copied exactly |
| `infer_motion_repair` | Corrupted `source_motion` and optional support mask | captions, detector policy, seed | repaired `motion_198` |

`generation_mask` uses `1` for values to generate and `0` for clean evidence
to preserve. All three APIs also return `rot6d`, `transl`, `keypoints3d`,
normalized `latent`, and valid `lengths`. Inputs and outputs remain on the
Pipeline device; `MotionCanvas-198` uses meters and 30 fps.

The same mask contract drives motion editing. In edit mode, generated cells
also receive the source motion as editable context:

```python
edited = pipe.infer_motion_editing(
    source,
    generate,
    captions="raise the right hand and wave",
    seed=42,
)
```

Frame or channel constraints use the same API. Optional world-space
`PositionConstraint` objects are projected during ODE integration:

```python
controlled = pipe.infer_kinematic_motion_control(
    source,
    generate,
    position_constraints=constraints,
    seed=42,
)
```

`motion_198` is in meters and local SMPL-22 rotation-6D. `rot6d`, `transl`,
`keypoints3d`, normalized `latent`, and valid `lengths` are returned alongside
it. The lower-level `infer_m2m` entry point covers all three modes directly.

### Training

MotionCanvas has a Motius-native `MotionCanvasSparseRolloutJoinTrainer`,
dataset pipeline, and public
[0.46B configuration](../../configs/motioncanvas/train_motioncanvas_0p46b.py).
It warm-starts the Text-to-Motion backbone from
`ZeyuLing/Motius-HYMotion-T2M-1.0-Lite` and trains the MotionCanvas-198 M2M
network on 360-frame, 30 fps crops.

The exact released mixture is 30% HYMotion full-mask generation, 60% HYMotion
arbitrary conditions, 5% MotionFix edits, and 5% PerMo edits. MotionFix and
PerMo can be downloaded through the [Dataset Hub](../datasets/README.md);
HYMotion Data requires separate authorized access. Follow the
[MotionCanvas data and export guide](../../configs/motioncanvas/README.md)
before changing paths in the config.

Train or resume with:

```bash
bash tools/dist_train.sh configs/motioncanvas/train_motioncanvas_0p46b.py 8 \
  --work-dir outputs/training/motioncanvas_0p46b \
  --auto-resume
```

| Training item | Released recipe |
| --- | --- |
| Precision | FP32 |
| Batch size | 96 per process; weighted epoch size 630,000 |
| Optimizer | AdamW, learning rate `1e-5` |
| Objective | Timestep-weighted Smooth L1 flow velocity |
| Geometry terms | FK consistency, joint position/velocity, conditioned-root, transition endpoint/Sobolev, and sparse-rollout join losses |
| Schedule | Up to 10,000 epochs, gradient norm clipped to `2.0` |
| Checkpoints | Every epoch, latest state saved, 30 historical checkpoints retained |
| Outputs | `outputs/training/motioncanvas_0p46b` |

The config enables automatic resume. Use
`--load-from CHECKPOINT --load-scope full` to select an explicit full-state
checkpoint. The [Training Hub](../training/README.md) explains distributed
launch and the distinction between resume and weight-only warm start.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Temporal Motion Completion | [Published results](../leaderboards/hf_space_temporal_condition/temporal_control_results.json) | HumanML3D temporal settings; normalized uTMR FID |
| Kinematic Motion Control | [Published results](../leaderboards/kinematic_motion_control.md) | Native-skeleton protocol; rows remain pending |
| Motion Editing | [Published results](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard) | MotionFix semantic preservation and edit-compliance metrics |
| Motion Repair | [Published results](../leaderboards/hf_space_motion_repair/motion_repair_results.json) | BrokenAMASS pair-validated repair metrics; explicit support tracks |

### Canonical Temporal-Completion Snapshot

#### Temporal Control · Motius normalized space

| Setting | n | R@1 | R@2 | R@3 | Motius FID (normalized) | MM-Dist | Diversity | Constraint error (cm) | Foot skating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| temporal_start_1f | 4,012 | 0.6914 | 0.8410 | 0.8988 | 0.0139 | 27.9444 | 54.0449 | 0.0000 | 0.1380 |
| temporal_pre20 | 4,012 | 0.6931 | 0.8406 | 0.8993 | 0.0091 | 27.8049 | 53.6818 | 0.0000 | 0.0840 |
| temporal_pre20_uncond | 4,012 | 0.3411 | 0.4731 | 0.5532 | 0.1043 | 40.0292 | 52.3568 | 0.0000 | 0.0960 |
| temporal_both_1f | 4,012 | 0.6883 | 0.8380 | 0.8972 | 0.0135 | 28.0997 | 53.9671 | 0.0000 | 0.1840 |
| temporal_mid80 | 4,012 | 0.6941 | 0.8452 | 0.9030 | 0.0052 | 27.7324 | 53.8369 | 0.0000 | 0.1100 |
| temporal_mid80_uncond | 4,012 | 0.4283 | 0.5758 | 0.6562 | 0.0837 | 36.9094 | 52.3345 | 0.0000 | 0.1250 |
| temporal_adaptive_keyframes | 4,012 | 0.6875 | 0.8441 | 0.9031 | 0.0014 | 27.8142 | 53.4264 | 0.0000 | 0.1184 |
| temporal_adaptive_keyframes_uncond | 4,012 | 0.6589 | 0.8218 | 0.8878 | 0.0031 | 28.6949 | 53.6000 | 0.0000 | 0.1358 |

### Canonical Motion-Repair Snapshot

| Support | n | uTMR R@1 | uTMR R@3 | uTMR M2M | MPJPE (cm) | Accel. error | Jitter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| oracle v6 | 299 | 0.9799 | 0.9967 | 0.9810 | 1.6483 | 2.4260 | 12.6213 |

### Canonical HumanML3D Semantic Results

| Evaluator | Variant | n | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HumanML3D Official | 0.46B · 360f | — | — | — | — | — | — | — | Not measured |
| MotionStreamer Evaluator | 0.46B · 360f | 4,042 | 0.7902 | 0.9139 | 0.9521 | 8.2765 | 14.7630 | 27.6745 | Measured |
| Motius Joint-Position Evaluator | 0.46B · 360f | 4,034 | 0.7086 | 0.8562 | 0.9103 | 0.0108 | 27.2810 | 54.1697 | Measured |

<!-- MOTIUS_CANONICAL_METRICS:END -->

The zero condition error reflects clean per-step imputation: observed
coordinates are projected back exactly after each solver step. It is not a
rounded estimate of an unconstrained prediction.

## Motion Representation

MotionCanvas-198 stores one frame as:

| Block | Dimensions | Meaning |
| --- | ---: | --- |
| Root translation | 3 | World-space XYZ in meters |
| Local joint rotations | 132 | 22 SMPL joints × row-major rotation-6D |
| FK joint positions | 63 | Pelvis-relative XYZ for the 21 non-root joints |

The 198-D statistics are the exact HY-Motion-201 statistics with the unused
pelvis-RIC triplet at dimensions 135:138 removed. This preserves the
HY-Motion T2M warm-start scale instead of recomputing an incompatible M2M-only
normalizer. `Mean.npy`, `Std.npy`, their provenance note, and the exact
SMPL-22 rest offsets are part of the checkpoint. Source and target motion are
canonicalized once before this API; Motius does not silently rotate individual
subclips inside `infer_m2m`.

`generation_mask` may be frame-level `(B,T)` or channel-level `(B,T,198)`.
Completion zeroes generated cells in the edit context. Editing retains an
explicit source value in those cells. Both paths fix all known coordinates to
clean evidence at initialization and after every ODE step.

## Citation and License

MotionCanvas is a Motius-native method; no external MotionCanvas paper or
source repository is claimed. Its transformer and text-conditioning design
build on [HY-Motion 1.0](https://arxiv.org/abs/2512.23464), while the M2M
training objective, arbitrary-condition data pipeline, clean-imputation
inference, sparse rollout, and released checkpoint are maintained in
[Motius](https://github.com/ZeyuLing/Motius).

The code follows the Motius repository license. Qwen3, CLIP, HYMotion training
data, MotionHub sources, SMPL body models, and rendered character assets retain
their respective upstream licenses and access terms. The checkpoint contains
text-encoder weights but does not redistribute licensed SMPL body-model files.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
