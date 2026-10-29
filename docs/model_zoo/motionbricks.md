<h1 align="center">MotionBricks Model Card</h1>

<p align="center">
  <strong>Real-time Unitree G1 motion primitives with modular latent generators.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2604.24833">Paper</a> ·
  <a href="https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks">Original GitHub</a>
</p>

MotionBricks is NVIDIA's real-time whole-body control stack released inside
GR00T-WholeBodyControl. It combines a VQVAE motion tokenizer, a pose model, and
a root model to stream controllable Unitree G1 motion primitives.

Motius implements the encoder, decoder, quantizer, pose network and root network
in `MotionBricksBundle`. Native stage Trainers own augmentation and objectives;
`MotionBricksPipeline` owns sampling, primitives, controllers and MuJoCo qpos
assembly. Body geometry, representations and assets live outside the model package.
Pretrained weights remain external artifacts.

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
| No registered public task | Native capability input | <video src="https://github.com/user-attachments/assets/425a675c-416f-4643-b7ac-9e93a79470d6" controls></video> | — |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


The inline preview is rendered from a deterministic G1 qpos rollout produced
by `Pipeline.from_pretrained(...).infer_g1_qpos_generation(...)`. MotionBricks
does not yet have a canonical Motius benchmark task, so the animation is a
runtime capability check rather than a leaderboard result.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Not registered | No canonical task API | See the capability boundary below |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 30 fps (Unitree G1) |
| Public preview | 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | Modular latent generative model plus smart primitives |
| Task status | Not registered |
| Native representation | MotionBricks G1 global 414D, local 413D, dual-root 418D |
| Robot skeleton | Unitree G1 29-DOF MuJoCo model |
| Public output | G1 qpos-36 stream |
| Checkpoint | [`ZeyuLing/motius-motionbricks-g1`](https://huggingface.co/ZeyuLing/motius-motionbricks-g1) |
| Pipeline | `motius.pipelines.motionbricks.MotionBricksPipeline` |

MotionBricks is a Model Zoo integration, but it is not registered under a
canonical Motius task and therefore does not appear in the Task Index. The
repository provides native navigation and representation components. The public
qpos capability is not a benchmark contract.

### Checkpoints

The released artifact contains the three neural checkpoints, G1 primitive clip,
stage configuration and normalization statistics. Native exports save the current
five neural roots and essential pipeline assets.
`motius.configuration.motionbricks.validate_checkpoint_layout` checks legacy
checkpoint completeness and unresolved LFS pointers before construction.

## Quick Start

Install optional runtime dependencies:

```bash
pip install -e ".[motionbricks]"
```

Run a headless qpos rollout:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-motionbricks-g1",
    bundle_kwargs={"device": "cuda"},
    controller="random",
)

result = pipe.infer_g1_qpos_generation(steps=240, seed=1234)
qpos = result["qpos"]          # (T, 36), Unitree G1 MuJoCo qpos
fps = result["fps"]            # 30

navigation = pipe.infer_g1_realtime_navigation(steps=240)
```

Use `controller="wasd"` for the interactive controller and `controller="random"`
for automated smoke tests or offline previews.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| No registered public task | Not applicable | See the capability boundary |

<!-- MOTIUS_CANONICAL_METRICS:END -->


No canonical Motius benchmark is registered for MotionBricks yet. The
released weights have been compared against independent original source at
commit `45c03facbf848e0c743091b7d9c17952096ff908`: neural outputs, two complete
160-frame random-controller rollouts, and all three original training-stage
objectives, active gradients and VQ EMA updates. These checks use synthetic
normalized training inputs; the full dataset and optimizer protocol remains
unverified. It is not assigned a leaderboard score.

## Motion Representation

MotionBricks and ARDY both target Unitree G1, but their tensors are different:

| Representation | Shape | Meaning |
| -------------- | ----: | ------- |
| `motionbricks_g1_414` | 414D | Global-root subset used by the root model |
| `motionbricks_g1_413` | 413D | Local-root subset used by pose/tokenizer modules |
| `motionbricks_g1_418` | 418D | Full dual-root feature tensor |
| `g1_38` | 38D | Motius compact G1 representation |
| `g1_qpos` | 36D | MuJoCo root pose plus 29-DOF robot state |

Motion representation transforms live in `motius.motion.representation.motionbricks`;
G1 qpos conversion and body geometry live in `motius.motion.body_models.motionbricks`.
The model package contains only neural modules.

## Citation and License

- Paper: [MotionBricks: Modular Motion Generation for Whole-Body
  Control](https://arxiv.org/abs/2604.24833)
- Original source:
  [NVlabs/GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl/tree/main/motionbricks)
- License and source provenance:
  [`motius/licenses/motionbricks/ATTRIBUTIONS.md`](../../motius/licenses/motionbricks/ATTRIBUTIONS.md)
  (source code Apache-2.0; original checkpoint and dataset terms preserved).

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
