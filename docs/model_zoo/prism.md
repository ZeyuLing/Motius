<h1 align="center">PRISM Model Card</h1>

<p align="center">
  <strong>Per-unit kinematic motion generation in a structured latent manifold.</strong>
</p>

<p align="center">
  <a href="https://github.com/ZeyuLing/Motius">Motius</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-prism-1.0-humanml3d">PRISM 1.0 Weights</a> |
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

## Preview

### PRISM 1.0

| HumanML3D Sample | Selected Input Text | SMPL Preview |
| ---------------- | ------------------- | ------------ |
| `001840` | someone executes a roundhouse kick with their left foot. | ![PRISM 1.0 HumanML3D 001840](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/prism_1_0/prism_1_0_humanml3d_001840_smpl_mesh_512_30fps.gif) |
| `004545` | a person jumping while raising both hands and moving apart legs. | ![PRISM 1.0 HumanML3D 004545](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/prism_1_0/prism_1_0_humanml3d_004545_smpl_mesh_512_30fps.gif) |
| `006944` | a person moves their right hand left, right, up, and down. | ![PRISM 1.0 HumanML3D 006944](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/prism_1_0/prism_1_0_humanml3d_006944_smpl_mesh_512_30fps.gif) |

### PRISM-KT

| HumanML3D Sample | Selected Input Text | SMPL Preview |
| ---------------- | ------------------- | ------------ |
| `001840` | someone executes a roundhouse kick with their left foot. | ![PRISM-KT HumanML3D 001840](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/prism_kt/prism_kt_humanml3d_001840_smpl_mesh_512_30fps.gif) |
| `004545` | a person jumping while raising both hands and moving apart legs. | ![PRISM-KT HumanML3D 004545](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/prism_kt/prism_kt_humanml3d_004545_smpl_mesh_512_30fps.gif) |
| `006944` | a person moves their right hand left, right, up, and down. | ![PRISM-KT HumanML3D 006944](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/prism_kt/prism_kt_humanml3d_006944_smpl_mesh_512_30fps.gif) |

All previews are 512px / 30fps SMPL mesh renders generated with the selected
HumanML3D captions used by the benchmark.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Tasks | T2M, TP2M, Sequential Generation |
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

## Usage

```python
from motius.pipelines.prism import PRISMPipeline

pipe = PRISMPipeline.from_pretrained(
    "kt",
    bundle_kwargs={
        "device": "cuda",
        "transformer_dtype": "bf16",
        "text_dtype": "bf16",
    },
)

t2m = pipe.text_to_motion(
    "a person takes two steps forward and waves with the right hand",
    num_frames=129,
    seed=42,
)
motion138 = t2m["motion_138"]
motion135 = t2m["motion_135"]
smpl = t2m["smpl"]
```

Prefix-conditioned generation accepts an SMPL `.npz` file or a MotionStreamer
272D `.npy` file:

```python
tp2m = pipe.temporal_condition(
    "the person turns left and begins to run",
    prefix_motion_path="prefix_motion.npz",
    condition_num_frames=5,
    num_frames=129,
)
```

Long motion uses the same clean-context/noisy-target interface:

```python
sequence = pipe.sequential_generation(
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

Set `model_name="1.0"` or pass `"1.0"` to load the baseline. On the KT
checkpoint, `kafs_mode="none"` disables KAFS for a shared-schedule ablation.
The VAE always runs in fp32; bf16 is used only for KU-FlowT and T5 inference.

## Evaluation Results

### T2M

Generation uses the fixed selected-caption HumanML3D test protocol. The two
cross-representation evaluators decode through the checked SMPL body-22 route.
Lower FID and MM-Dist are better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: |
| HumanML3D Official | PRISM 1.0 | 3,970 | 0.5560 | 0.7465 | 0.8366 | 0.1992 | 2.8057 | 9.6033 |
| HumanML3D Official | PRISM-KT + KAFS | 3,970 | 0.5448 | 0.7308 | 0.8176 | 0.2081 | 2.9067 | 9.4802 |
| MotionStreamer Evaluator | PRISM 1.0 | 4,042 | 0.7463 | 0.8832 | 0.9241 | 19.0359 | 15.5135 | 27.4151 |
| MotionStreamer Evaluator | PRISM-KT + KAFS | 4,042 | 0.7408 | 0.8619 | 0.9050 | 19.9682 | 15.8072 | 27.2536 |
| Motius Joint-Position Evaluator | PRISM 1.0 | 4,034 | 0.6483 | 0.8065 | 0.8725 | 146.1266 | 30.9335 | 57.4521 |
| Motius Joint-Position Evaluator | PRISM-KT + KAFS | 4,034 | 0.6629 | 0.8105 | 0.8656 | 123.8378 | 30.5957 | 56.6863 |

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

### Sequential Generation

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

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.prism.PRISMPipeline` |
| Bundle | `motius.models.prism.PRISMBundle` |
| Motion processor | `motius.models.prism.PRISMMotionProcessor` |
| KU-FlowT | `motius.models.prism.network.PrismTransformerMotionModel` |
| Motion VAE | `motius.models.prism.AutoencoderKLPrism2DTK` |

## Citation

```bibtex
@article{ling2026prism,
  title={PRISM: Per-unit Kinematic Motion Generation in a Structured Latent Manifold},
  author={Ling, Zeyu and Shuai, Qing and Zhang, Teng and Li, Shiyang and Han, Bo and Zou, Changqing},
  year={2026}
}
```
