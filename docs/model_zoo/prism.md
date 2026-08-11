<h1 align="center">PRISM Model Card</h1>

<p align="center">
  <strong>Per-unit kinematic motion generation in a structured latent manifold.</strong>
</p>

<p align="center">
  <span>Paper: manuscript</span> ·
  <a href="https://github.com/ZeyuLing/Motius">Motius Implementation</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-prism-1.0-humanml3d">PRISM 1.0 Weights</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-prism-kt-humanml3d">PRISM-KT Weights</a>
</p>

PRISM addresses a representation mismatch in text-to-motion systems: a
holistic frame or clip latent entangles root trajectory, global orientation,
and local articulation, so the generator must rediscover body structure from
anonymous latent channels. PRISM instead uses a causal Motion VAE whose latent
grid retains one addressable token for root motion and every SMPL body unit.
A Kinematic-Unit Flow Transformer (KU-FlowT) performs text-conditioned flow
matching on this time-by-body grid.

The structured grid makes the method's controls concrete. Per-token Diffusion
Forcing keeps observed prefix tokens clean while target tokens are denoised,
so the same generator handles T2M, TP2M, and autoregressive segment chaining.
KT-RoPE replaces arbitrary joint-storage positions with kinematic-tree-derived
coordinates, and KAFS applies a depth-aware, parameter-free inference schedule
that keeps the root on the base schedule while refining distal joints later.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/d78929a3-f7b7-4d60-a4fa-4ed5540b530b" controls></video> | [MP4](https://github.com/user-attachments/assets/d78929a3-f7b7-4d60-a4fa-4ed5540b530b) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=prismkafs) |
| Temporal Motion Completion | Swaggering walk with observed frames / keyframes | <video src="https://github.com/user-attachments/assets/5e55fa78-7171-4d91-a900-93aa8962048c" controls></video> | [MP4](https://github.com/user-attachments/assets/5e55fa78-7171-4d91-a900-93aa8962048c) · [All cases](https://zeyuling-babel-sequential-generation-leaderboard.static.hf.space/cases/index.html?method=prism) |
| Sequential Text-to-Motion | A person walks forward. → A person sits down. → A person rests. → A person stands up. → A person walks back.<br><sub>Five captioned segments; the active caption and segment timeline are embedded in the video</sub> | <video src="https://github.com/user-attachments/assets/ceb83791-6797-4389-805a-ff9135459f97" controls></video> | [MP4](https://github.com/user-attachments/assets/ceb83791-6797-4389-805a-ff9135459f97) · [All cases](https://zeyuling-babel-sequential-generation-leaderboard.static.hf.space/cases/index.html?method=prism&case=val_919) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


### PRISM 1.0

| Input | SMPL Preview |
| ------------------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/158f9df4-6dbc-4b48-84e3-7ceb1e2694d9" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/428fb002-7f68-487d-a779-52e7b3e7fcb3" controls></video> |

### PRISM-KT

| Input | SMPL Preview |
| ------------------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/d78929a3-f7b7-4d60-a4fa-4ed5540b530b" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/52742994-9bd8-4db1-a1c4-cab6ea954f86" controls></video> |

All previews are 512px / 30fps SMPL mesh renders generated with the selected
HumanML3D captions used by the benchmark.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |
| Temporal Motion Completion | `infer_temporal_motion_completion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) |
| Sequential Text-to-Motion | `infer_sequential_text_to_motion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 30 fps (prism_motion138) |
| Public preview | 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Tasks | Text-to-Motion, Temporal Motion Completion, Sequential Text-to-Motion |
| Native representation | `prism_motion138` at 30 fps |
| Skeleton | SMPL-H input, fixed SMPL body-22 subset |
| Generator | 1.4B KU-FlowT with T5-XXL conditioning |
| Pipeline | `motius.pipelines.prism.PRISMPipeline` |

| Variant | Joint coordinate | Inference schedule | Checkpoint |
| ------- | ---------------- | ------------------ | ---------- |
| PRISM 1.0 | Sequential joint-axis RoPE | Shared flow schedule | [`ZeyuLing/motius-prism-1.0-humanml3d`](https://huggingface.co/ZeyuLing/motius-prism-1.0-humanml3d) |
| PRISM-KT | Projected-spectral `spectral_unified` KT-RoPE | Depth-driven KAFS by default | [`ZeyuLing/motius-prism-kt-humanml3d`](https://huggingface.co/ZeyuLing/motius-prism-kt-humanml3d) |

Each repository is self-contained: KU-FlowT, the causal Motion VAE, T5
tokenizer and text encoder, scheduler, motion statistics, and Motius artifact
metadata are versioned together. The released KT checkpoint is the epoch-43
model used for the reported KT/KAFS evaluation; the 1.0 checkpoint is the
iter-15,000 sequential-RoPE baseline.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.prism.PRISMPipeline` |
| Bundle | `motius.models.prism.PRISMBundle` |
| Motion processor | `motius.models.prism.PRISMMotionProcessor` |
| KU-FlowT | `motius.models.prism.network.PrismTransformerMotionModel` |
| Motion VAE | `motius.models.prism.AutoencoderKLPrism2DTK` |

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-prism-kt-humanml3d",
    bundle_kwargs={
        "device": "cuda",
        "transformer_dtype": "bf16",
        "text_dtype": "bf16",
    },
)

t2m = pipe.infer_text_to_motion(
    ["a person takes two steps forward and waves with the right hand"],
    [129],
    seed=42,
)[0]
motion138 = t2m["motion_138"]
motion135 = t2m["motion_135"]
smpl = t2m["smpl"]
```

Prefix-conditioned generation accepts an SMPL `.npz` file or a MotionStreamer
272D `.npy` file:

```python
tp2m = pipe.infer_temporal_motion_completion(
    "the person turns left and begins to run",
    prefix_motion_path="prefix_motion.npz",
    condition_num_frames=5,
    num_frames=129,
)
```

Long motion uses the same clean-context/noisy-target interface:

```python
sequence = pipe.infer_sequential_text_to_motion(
    [
        "a person walks forward",
        "the person stops and looks to the left",
        "the person sits down",
    ],
    segment_frames=[121, 91, 121],
    ar_condition_frames=9,
    seed=42,
)
```

Sequential generation carries 9 causal context frames across subclip
boundaries. The training protocol samples observed prefixes of 1, 5, or 9
frames, so the public API stays within that distribution. TP2M independently
accepts any of those trained observed-prefix lengths.

Load `ZeyuLing/motius-prism-1.0-humanml3d` for the baseline artifact. On the KT
checkpoint, `kafs_mode="none"` disables KAFS for a shared-schedule ablation.
The VAE always runs in fp32; bf16 is used only for KU-FlowT and T5 inference.

### Training

Motius provides a registered `PrismTrainer` and public
[configuration](../../configs/prism/train_prism.py). The released recipe is a
continued-training/fine-tuning path: by default it loads
`ZeyuLing/motius-prism-kt-humanml3d`, keeps the Motion VAE frozen in FP32, and
optimizes the KU-FlowT transformer. Set `MOTIUS_PRISM_PRETRAINED` to another
compatible local snapshot or Hugging Face artifact when training a different
variant.

Training data uses the compact manifest contract with physical-scale
`prism_motion138` arrays at 30 fps. Captions can be encoded online or supplied
as cached T5 features. When cached features and prompt dropout are used,
`MOTIUS_NULL_TEXT_FEATURE` must point to the matching empty-prompt embedding.
See the [data-format guide](../training/prism_tmr_hymotion_t2m.md).

```bash
MOTIUS_DATA_ROOT=/path/to/prism_motion138 \
MOTIUS_TRAIN_MANIFEST=train.json \
MOTIUS_NULL_TEXT_FEATURE=/path/to/null_t5_feature.pt \
bash tools/dist_train.sh configs/prism/train_prism.py 8 \
  --work-dir outputs/training/prism \
  --auto-resume
```

| Training item | Released recipe |
| --- | --- |
| Precision | KU-FlowT and text states in BF16; VAE encoding and loss in FP32 |
| Batch size | 8 per process |
| Optimizer | AdamW, learning rate `1e-4` |
| Conditioning | 10% conditioned examples; observed prefixes sampled from 1, 5, or 9 frames; 10% prompt dropout |
| Objective | Flow MSE split into translation and body-rotation terms with `0.5 / 0.5` weighting |
| Schedule | 100 epochs, gradient norm clipped to `1.0` |
| Checkpoints | Every epoch, latest state saved, five historical checkpoints retained |
| Outputs | `outputs/training/prism` |

Use `--load-from CHECKPOINT --load-scope full` for an explicit full-state
resume. The [Training Hub](../training/README.md) documents multi-node launch,
configuration overrides, and checkpoint semantics.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Temporal Motion Completion | [Published results](../leaderboards/hf_space_temporal_condition/temporal_control_results.json) | HumanML3D temporal settings; normalized uTMR FID |
| Sequential Text-to-Motion | [Published results](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | BABEL semantic and transition metrics; normalized uTMR FID |

### Canonical Temporal-Completion Snapshot

#### TP2M Prefix · MotionStreamer-272 space

| Setting | n | R@1 | R@2 | R@3 | MotionStreamer FID | MM-Dist | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-frame prefix | 4,042 | 0.7798 | 0.9087 | 0.9482 | 16.2756 | 14.9304 | 27.5360 |
| 5-frame prefix | 4,042 | 0.7912 | 0.9107 | 0.9489 | 13.5449 | 14.7900 | 27.4694 |
| 9-frame prefix | 4,042 | 0.7867 | 0.9144 | 0.9529 | 12.5467 | 14.7775 | 27.5227 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


### T2M

Generation uses the fixed selected-caption HumanML3D test protocol. The two
cross-representation evaluators decode through the checked SMPL body-22 route.
Lower FID and MM-Dist are better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: |
| HumanML3D Official | PRISM 1.0 | 3,970 | 0.5560 | 0.7465 | 0.8366 | 0.1992 | 2.8057 | 9.6033 |
| HumanML3D Official | PRISM-KT + KAFS | 3,970 | 0.5448 | 0.7308 | 0.8176 | 0.2081 | 2.9067 | 9.4802 |
| MotionStreamer Evaluator | PRISM 1.0 | 4,042 | 0.7463 | 0.8832 | 0.9241 | 19.0359 | 15.5135 | 27.4151 |
| MotionStreamer Evaluator | PRISM-KT + KAFS | 4,042 | 0.8003 | 0.9137 | 0.9477 | 11.8101 | 14.5998 | 27.3849 |
| Motius Joint-Position Evaluator | PRISM 1.0 | 4,034 | 0.6483 | 0.8065 | 0.8725 | 0.0804 | 30.9335 | 57.4521 |
| Motius Joint-Position Evaluator | PRISM-KT + KAFS | 4,034 | 0.7197 | 0.8669 | 0.9192 | 0.0162 | 27.0247 | 53.8865 |

HumanML3D Official values are means over 20 repeats after converting the
native SMPL body-22 output to unnormalized HumanML3D-263 at 20 fps. The
evaluator uses the same fixed selected captions as generation.

Physical diagnostics on the same 4,042 generated samples:

| Variant | Slide | Float | Jitter | Dynamic | PoseQ |
| ------- | ----: | ----: | -----: | ------: | ----: |
| PRISM 1.0 | 3.6746 | 7.8379 | 6.1307 | 27.7158 | 1.6854 |
| PRISM-KT + KAFS | 3.4524 | 7.7084 | 6.4377 | 27.6258 | 1.6807 |

### TP2M

PRISM-KT uses one model for every prefix length; no TP2M-specific weights are
loaded. The table reports MotionStreamer Evaluator results after conditioning
on the first 1, 5, or 9 frames.

| Prefix | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ------ | ------: | --: | --: | --: | --: | ------: | --------: |
| 1 frame | 3,968 | 0.7467 | 0.8813 | 0.9214 | 48.4399 | 16.2216 | 26.7002 |
| 5 frames | 3,968 | 0.7588 | 0.8957 | 0.9330 | 38.8512 | 15.8599 | 26.8676 |
| 9 frames | 3,968 | 0.7649 | 0.8942 | 0.9367 | 36.4133 | 15.7691 | 26.9763 |

### Sequential Text-to-Motion · BABEL

The BABEL benchmark uses all 1,295 eligible validation episodes and 7,285
captioned subclips. PRISM runs the epoch-18 checkpoint with CFG `1.5`, seed
`42`, and 9 carried context frames, one of the prefix lengths used during
training. Each generated episode has exactly the requested duration; semantic
subclips are independently canonicalized before evaluation. FID values use
L2-normalized uTMR embeddings. R-Precision and MM-Dist use 227 complete
batches of 32 (`7,264` subclips); FID and Diversity use all `7,285` subclips.

| Scope | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ----- | --: | --: | --: | --: | ------: | --------: |
| 7,285 semantic subclips | 0.1933 | 0.3106 | 0.3908 | 0.0680 | 53.5178 | 48.4755 |

| Transition FID | Transition Diversity | Peak Jerk | AUJ Gap |
| -------------: | -------------------: | --------: | ------: |
| 0.0645 | 44.3111 | 249.7794 | 62.3776 |

[Inspect 24 GT/PRISM sequences](https://knights-ser-moment-work.trycloudflare.com/visualization/babel_prism_epoch18_cfg1p5_ar9_full24/),
including 12 regular samples and 12 automatically selected quality-tail
samples. The viewer renders the native SMPL output, segment captions, floor,
trajectory, and first-frame body facing; it does not pass PRISM through the
HumanML3D-to-SMPL fitting route.

## Motion Representation

`prism_motion138` is the model's native decoded tensor:

| Channels | Meaning |
| -------- | ------- |
| `0:3` | Absolute root translation |
| `3:6` | Per-frame root translation delta |
| `6:12` | Global orientation in column-major rotation 6D |
| `12:138` | 21 local body-joint rotations in column-major rotation 6D |

The Motion VAE reshapes this tensor to `[T, 23, 6]`: one translation unit,
one root-orientation unit, and 21 local body units. `motion_135`, SMPL
parameters, and MotionStreamer-272 are output/evaluation adapters rather than
additional native PRISM representations. Hand-pose channels are not generated
by these body-22 checkpoints.

## Citation and License

The PRISM paper is currently an unpublished manuscript. The implementation and
release artifacts are maintained in Motius; no external paper URL or upstream
source repository is claimed.

```bibtex
@article{ling2026prism,
  title={PRISM: Per-unit Kinematic Motion Generation in a Structured Latent Manifold},
  author={Ling, Zeyu and Shuai, Qing and Zhang, Teng and Li, Shiyang and Han, Bo and Zou, Changqing},
  year={2026}
}
```

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
