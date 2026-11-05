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

**Supplied-video cases — Base and Latest, 25 seconds each:**

| Checkpoint | Full result: input with tracking bbox + SMPL-H mesh | Model card |
| --- | --- | --- |
| Base | [Watch Base](../../assets/model_zoo/flowhmr/demo_base_mesh.mp4) | [Hugging Face Base](https://huggingface.co/ZeyuLing/Motius-FlowHMR-Base#visual-results) |
| Latest (PHC+ GRPO) | [Watch Latest](../../assets/model_zoo/flowhmr/demo_latest_mesh.mp4) | [Hugging Face Latest](https://huggingface.co/ZeyuLing/Motius-FlowHMR-Latest#visual-results) |

Both checkpoints run the same [supplied input video](../../assets/model_zoo/flowhmr/demo_input.mp4)
through the public Motius pipeline: 750 frames at 30 fps, seed 0, 20 Euler
steps and CFG 1. They share the exact cached tracking boxes, SAM features and
VGGT camera estimates (9 keyframes at interval 100, interpolated to 750).
Each video is generated from its own checkpoint prediction. Inference safetensors:

- Base: `f8dca504bfe1682ab0a8ec6cb99e555b4c93e77220cc5c80636a87f0492ced46`.
- Latest: `f4ae4168a6f576ffe08a3a19b245e0fd65c0d304b8e93bc2c6ab0a46a6f86912`.

The left panel shows the tracking bbox. The right panel reconstructs the
predicted SMPL-H mesh from saved rotations, shape and translation, with
Three.js smooth normals, depth-tested surfaces, lighting and floor shadows.
Both renders use the same fixed camera, fitted to the union of both sequences.
Horizontal root motion is centered for display; vertical motion is retained.
This is a world view, not a reprojection into the source camera. Original audio
is retained. Weight hashes, shared-input hashes and rendering settings are
recorded in the [Base provenance](../../assets/model_zoo/flowhmr/demo_base_mesh.json)
and [Latest provenance](../../assets/model_zoo/flowhmr/demo_latest_mesh.json).

The older validation clips below use a separate 30-frame input.

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

### Supplied-video previews

| Base | Latest (PHC+ GRPO) |
| --- | --- |
| [![Base: tracking bbox and predicted mesh](../../assets/model_zoo/flowhmr/demo_base_mesh_preview.png)](../../assets/model_zoo/flowhmr/demo_base_mesh.mp4) | [![Latest: tracking bbox and predicted mesh](../../assets/model_zoo/flowhmr/demo_latest_mesh_preview.png)](../../assets/model_zoo/flowhmr/demo_latest_mesh.mp4) |

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

Watch the [Blender product release film](../../assets/model_zoo/flowhmr/blender_demo/motius-flowhmr-product-release-1080p.mp4) and follow the [illustrated first-use guide](../tutorials/flowhmr-blender.md). The film records fresh full-video inference and a complete original-speed 750-frame result, with explicit processing cuts and timing boundaries.

Blender users can install the [FlowHMR add-on](../../blender_extension/README.md),
which provides standard and accelerated inference, native 52-joint animation,
Mixamo retargeting and animated FBX export.

```bash
pip install -e '.[flowhmr,video-to-motion]'
```

### One-command inference

Since 0.3.15, the default `--mode standard` retains official SAM and uses shared
streaming and batching optimizations. `--mode lightweight` (or `--lightweight`)
uses Fast SAM. Both retain every frame and both hands. Use
`--frontend-precision float16` explicitly selects mixed FP16 for the SAM backbone and VGGT aggregator; `float32` is a comparison mode. The default preserves source precision.

Once the licensed **neutral SMPL-H NPZ with 16 shape coefficients** is at
`checkpoints/body_models/smplh/neutral/model.npz`, run:

```bash
python tools/infer_flowhmr.py input.mp4 --lightweight --profile
```

The command downloads `ZeyuLing/Motius-FlowHMR-Latest` and its frontend weights,
transcodes to 30 fps, tracks the person, estimates cameras, extracts full
body/left-hand/right-hand features, and saves `outputs/inference/flowhmr/result.npz`.
`--profile` also writes runtime, CUDA peak allocation and inference settings.
An existing local checkpoint remains supported:

```bash
python tools/infer_flowhmr.py outputs/hf/flowhmr-base input.mp4 \
  --lightweight --body-model /path/to/smplh/neutral/model.npz \
  --work-dir outputs/inference/flowhmr-light --profile
```

`MOTIUS_SMPLH_MODEL` can select the licensed asset once for subsequent commands;
`MOTIUS_BODY_MODEL_DIR` selects a body-model root. Male/female SMPL-H files
cannot replace the checkpoint's neutral model. The joint regressor is optional
for inference. `--cache-dir` controls Hub caching, and `--max-frames` limits a run.
The Python API accepts `lightweight=True` in `Pipeline.from_pretrained(...)`.

The implementation integrates the engineering ideas discussed in
[upstream issue #4](https://github.com/flowhmr/flowhmr/issues/4):

- Query batches of 256 avoid sequence-square attention buffers. Original
  local visibility, cross-stream joint softmax, global token-refiner attention,
  padding and RoPE positions are retained. `--attention-query-chunk-size`
  adjusts memory versus overhead. This path runs only in evaluation without
  gradients; training retains the standard attention implementation.
- Motion weights move to CPU during frontend execution. YOLOX, VGGT and SAM
  release GPU resources between stages; weights return before motion sampling.
- Since 0.3.7, Fast SAM uploads each RGB frame once, performs body/hand image
  crops and normalization on GPU, and runs both hands in a single `[left,right]`
  network batch. It captures the original first body token and both actual hand
  tokens, then exits before unused post-decoder fusion/prompt processing.
  Intermediate pose predictions that condition decoder layers remain intact.
  There are no average hand features or skipped video frames. GPU interpolation
  reproduces OpenCV's fixed-point coordinate table and uint8 rounding; only
  small affine metadata remains on CPU to preserve the reference geometry.
  Use `--sam-backend official` (Python: `sam_backend='official'`) for the
  original SAM model and preprocessing, or `--sam-backend fast` to select Fast.
  Optimized execution avoids unused outputs; `--mode reference` restores the
  reference schedule for comparisons.
  `auto` selects Fast with `--lightweight` and original SAM otherwise.

Since 0.3.11, Fast conditioning caches frozen dummy prompts and initial-pose
embeddings, reuses device ray grids, and omits only unused final geometry and
mesh outputs. All intermediate keypoint predictions and every frame's body/hand
features remain. Set `preprocessor_kwargs={'sam_optimize_conditioning': False}`
to retain the preceding Fast path. Blender 0.2.7 shares scene updates across
independent characters while keeping full deformed-mesh grounding checks.
See the [compute-wait measurements](../architecture/validation/flowhmr-extreme-20261026.md):
the full backend remains above 60 seconds and numerical output differences are reported explicitly.

Since 0.3.10, streamed VGGT prepares images with four ordered CPU workers,
releases AMP cast-weight caches when offloading modules and measures installed
PyTorch/cuDNN exact attention kernels on the actual long-sequence input.
`--vggt-attention-backend torch` preserves the prior kernel; `auto` selects cuDNN
only when measured faster and safely falls back within the attention operation.
No camera frames or global keys are dropped. See the [VGGT full-video report](../architecture/validation/flowhmr-vggt-20261026.md)
for performance, allocation versus driver memory, and output differences.

Since 0.3.9, lightweight Fast SAM batches up to eight independent video frames,
retaining each frame's own selected box and camera intrinsics. Set
`--sam-frame-batch-size 1` to restore the previous per-frame schedule. No frames
or hand features are skipped. See the [full-video batch and progress report](../architecture/validation/flowhmr-batches-20261026.md)
for measured speed, memory and output differences. Blender 0.2.5 displays live
per-stage counts/rate/ETA for tracking, camera encoding/layers and SAM features.

Since 0.3.8, `--lightweight` also selects streamed VGGT camera evaluation.
Independent frame encoders and frame blocks use microbatches; complete global
keys/values stay visible to every query. Only one network module resides on GPU
at a time, and only camera/register prefixes are retained for CameraHead.
`--vggt-memory-mode reference` restores the original camera execution schedule;
`streamed` explicitly selects staging. Original model dtypes and weights remain
unchanged. CPU memory and transfer costs are part of the measured tradeoff.

Camera estimation still uses every frame by default (`--vggt-frame-interval 1`).
Explicit camera subsampling can reduce VGGT memory further but changes the
camera predictions. Long-video frontend memory still grows with frame count;
the sampling measurements below are not full long-video memory guarantees.
Compilation and any *additional* precision reduction are not enabled by lightweight mode.
The original frontend itself uses BF16 where supported; see the
[precision comparison](../architecture/validation/flowhmr-shared-standard-20261027.md).

Historical measurements before the 0.3.6 loading/camera optimizations, on one H200, FP32 motion sampling, 20 Euler steps, seed 17, with query batches of 256:

| Stage | Standard peak | Lightweight peak | Standard / lightweight time |
| --- | --- | --- | --- |
| 30-frame real video, including loading | 9.70 GiB | 7.23 GiB | 45.56 / 44.79 s |
| 2,100-frame sampling | 5.04 GiB | 2.37 GiB | 3.23 / 2.96 s |
| 3,789-frame sampling | 11.68 GiB | 2.81 GiB | 8.73 / 4.76 s |

Long sampling tests repeat measured 30-frame features/cameras and exclude live
frontends. Each timing is one fresh-process run, not a throughput average.
Short sampling can be slower (30 frames: 0.43 / 0.67 s). Peak memory is
PyTorch allocated memory for the measured process, not whole-device usage.

Lightweight results are **not bitwise identical**: the real-video test measured
maximum joint rotation difference **0.0265 degrees** and root translation
difference **0.122 mm**. SAM body-token differences were below 2.9e-6; small
body differences can affect hand crops (minimum hand-token cosine similarity
0.9999175). Long sampling alone differed by at most 0.000141 degrees and
0.031 mm. Latest CFG-2 sampling matched exactly in the tested 30-frame case.
These are measured cases, not accuracy bounds for every video. For the original scheduling use `--mode reference`; the default standard mode
uses official SAM with shared optimizations and is not guaranteed bitwise identical.
The [measurement record](../architecture/validation/flowhmr-lightweight.json)
contains settings, hashes and per-case comparisons. Reproduce paired cases
with `tools/benchmark_flowhmr_lightweight.py` and summarize them using
`tools/verify_flowhmr_lightweight.py`.

### Latency profiling (0.3.6)

Checkpoint loading now constructs the fully checkpointed FlowHMR and VGGT
networks, and the SAM DINO backbone, on the meta device before assigning
weights. Checkpoint dtypes follow the original eager loader; deterministic
nonpersistent buffers and the released SAM zero mask token are materialized.
Training network construction remains available without meta initialization.

Camera extraction skips unused depth/alignment outputs and only retains the
last aggregator layer. Lightweight SAM exits each hand pass after its decoder
token; the original body reconstruction still supplies both original hand boxes.
Per-frame CUDA allocator flushing is removed; stage teardown still releases
frontend resources. Sampling steps, input frames and hand features remain intact.

`--profile` writes model loading and synchronized transcode, tracking, camera,
feature and sampling/decoding timings to `inference_profile.json`. Blender 0.2.2
also writes this file automatically in each job's output directory, including
whether precomputed frontend features were used. These timings exclude Blender
surface import, retargeting, grounding and playback preparation.

See [the latency validation report](../architecture/validation/flowhmr-latency-20261026.md)
for measured improvements, output differences and the limits of H200 results.

### Fast SAM validation (0.3.7)

GPU image preprocessing and merged-hand execution live under
`motius/pipelines/flowhmr/sam`; the original SAM/DINO networks and checkpoint
weights remain under their native vision bundles. There is no runtime dependency
on a Fast SAM repository or a second packaged model tree.

The [full-video Fast SAM report](../architecture/validation/flowhmr-fast-sam-20261026.md)
compares all 750 frames against independent pinned upstream checkouts, including
per-branch tokens, final motion, runtime and memory. The earlier 0.3.6 report
measured a partial token-only path and is not a Fast SAM benchmark.

### Full-frame memory validation (0.3.8)

The [750-frame memory report](../architecture/validation/flowhmr-memory-20261026.md)
measures streamed VGGT with Fast SAM under a 6 GiB H200 allocator limit.
It includes driver memory, CPU RSS, runtime and original-source output differences.
This is not a consumer GPU hardware benchmark or a bitwise equivalence claim.

### Python inference

```python
from motius import Pipeline
from motius.motion.representation.monocular_capture import save_monocular_capture_result

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-FlowHMR-Latest",  # or ZeyuLing/Motius-FlowHMR-Base
    body_model_path="outputs/checkpoints/flowhmr/body_models/smplh/neutral/model.npz",
    bundle_kwargs=dict(device="cuda"),
    lightweight=True,
)
result = pipe.infer_monocular_motion_capture(
    "assets/model_zoo/flowhmr/demo_input.mp4",
    work_dir="outputs/inference/flowhmr", vggt_frame_interval=1, seeds=[0],
)
save_monocular_capture_result(result, "outputs/inference/flowhmr/result.npz")
```

This example preserves all camera frames. The streamed memory measurements
use an H200 with an allocator limit; they do not establish T4/RTX runtime or
compatibility. Render the side-by-side preview with:

```bash
pip install -e '.[flowhmr-render]'
python -m playwright install chromium
python tools/render_flowhmr_demo.py assets/model_zoo/flowhmr/demo_input.mp4 \
  outputs/inference/flowhmr/result.npz outputs/inference/flowhmr/demo.mp4 \
  --body-model outputs/checkpoints/flowhmr/body_models/smplh/neutral/model.npz \
  --checkpoint-label 'FlowHMR Latest'
```

`--preview` writes a five-frame PNG contact sheet for visual inspection before
encoding the video. For a matched Base/Latest comparison, pass each result as
`--camera-reference` when rendering the other result; both cameras then fit the
union of the two sequences. Rendering uses the Three.js module shipped with Motius;
only the licensed body model is supplied separately. The MP4 sidecar records
the inference checkpoint hash, camera, frame count and media hashes.

Provide the official AMASS neutral SMPL-H model (52 joints, 16 shape parameters).
`infer_from_feature(tokens, camera_RT)` accepts precomputed `(T,3072)` tokens
and OpenCV world-to-camera transforms at 30 fps. `infer_v2m` transcodes video,
tracks a person with YOLOX, predicts VGGT cameras and extracts SAM features.
Rotation conditioning uses `R_t @ R_0.T`, matching training and the corrected
upstream inference path. The first frame is identity, and changing the world
coordinate axes does not change conditioning. `camera_R_condition` returns
these relative rotations; `camera_RT` retains the supplied extrinsics.
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

See [ATTRIBUTIONS](../../motius/licenses/flowhmr/ATTRIBUTIONS.md),
[FlowHMR license](../../motius/licenses/flowhmr/UPSTREAM_LICENSE),
[SAM license](../../motius/licenses/flowhmr/SAM3D_LICENSE),
[VGGT-Omega license](../../motius/licenses/flowhmr/VGGT_OMEGA_LICENSE),
[DINOv3 license](../../motius/licenses/flowhmr/DINOV3_LICENSE.md),
[YOLOX license](../../motius/licenses/flowhmr/YOLOX_LICENSE) and
[MHR license](../../motius/licenses/flowhmr/MHR_LICENSE).
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


### Durable Blender jobs and shared standard optimization (Motius 0.3.15 / extension 0.2.9)

Inference now uses an independent persistent queue. Closing Blender or switching
files leaves queued jobs running; results are imported only on explicit request.
Preflight checks CUDA/BF16 execution, available memory, video dimensions, disk and
runtime tools. Low-memory scheduling adjusts SAM/VGGT frame batches and attention
workspaces; CUDA OOM retries use fresh processes without changing frames, model
dtypes, seed or sampling steps. Standard mode retains its original compute path.

The previous 0.3.14 standard mode measured 366.81 s / 49.78 GiB on the complete
750-frame video. Current standard mode retains official SAM while sharing streamed
VGGT and bounded batching; see the [current standard and precision validation](../architecture/validation/flowhmr-shared-standard-20261027.md). See the [productization validation](../architecture/validation/flowhmr-productization-20261027.md)
for timing scope, real OOM recovery and platform limitations. PyTorch remains the shipping backend; the [transferable runtime context](../../deploy/flowhmr/README.md) isolates dependencies without requiring ONNX/TensorRT conversion.

### Optional mixed FP16 frontend (0.3.16)

Use `--frontend-precision float16` to run the SAM backbone and VGGT aggregator in mixed FP16; the upstream precision remains the default. The Blender Options panel exposes the same policy. This does not convert the entire network to FP16 or recover precision already lost in stored BF16 weights. Camera caches are precision-specific and out-of-range SAM backbone checkpoint values are rejected. See the [three-video validation](../architecture/validation/flowhmr-fp16-multivideo-20261027.md) for measured differences, speed, memory and activation-monitoring limits.
