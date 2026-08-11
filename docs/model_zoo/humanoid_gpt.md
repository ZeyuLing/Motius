<h1 align="center">HumanoidGPT Model Card</h1>

<p align="center">
  <strong>A billion-scale pretrained Unitree G1 controller for zero-shot motion tracking.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2606.03985">Paper</a> ·
  <a href="https://github.com/GalaxyGeneralRobotics/Humanoid-GPT">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-HumanoidGPT-G1">Motius Checkpoint</a>
</p>

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
| Motion Tracking | Method-native reference motion and controller state | <video src="https://github.com/user-attachments/assets/d44a17dc-2caa-49db-be9e-645d911c03e1" controls></video> | [MP4](https://github.com/user-attachments/assets/d44a17dc-2caa-49db-be9e-645d911c03e1) · [All cases](https://zeyuling-motion-tracking-mujoco-leaderboard.static.hf.space/cases/index.html?method=humanoid_gpt) · [Leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) |

The policy-step API and physical rollout are evaluated under the registered MuJoCo or Isaac Lab protocol stated in the Model Card.

<!-- MOTIUS_TASK_DEMOS:END -->

The preview compares the kinematic reference with the persisted physical
MuJoCo rollout. The public all-case viewer uses the same rollout artifacts as
the leaderboard metrics.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Motion Tracking | `infer_motion_tracking` | [Task and runtime contract](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 50 Hz G1 control; official training code is not released |
| Public preview | 30 fps visualization sampled from the 50 Hz physical rollout |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->

| Item | Value |
| --- | --- |
| Method | HumanoidGPT non-privileged 216 |
| Tasks | Motion Tracking |
| Task status | MuJoCo metrics complete; all-case physical rollouts public |
| Robot | Unitree G1-5010, 29 actuated joints |
| Native control rate | 50 Hz |
| Checkpoint | [`ZeyuLing/Motius-HumanoidGPT-G1`](https://huggingface.co/ZeyuLing/Motius-HumanoidGPT-G1) |
| Pipeline APIs | `infer_motion_tracking`, `rollout_motion_tracking` |

### Checkpoint

[`ZeyuLing/Motius-HumanoidGPT-G1`](https://huggingface.co/ZeyuLing/Motius-HumanoidGPT-G1)
contains the official non-privileged `pns_wo_priv216.onnx` policy, a
self-describing Motius manifest, and the complete Unitree G1-5010 MJCF and mesh
tree. Loading it does not download or import the upstream repository.

### Release Boundary

The upstream project releases inference, deployment, and pretrained weights.
Its training code and training data are not public, so Motius does not expose a
HumanoidGPT trainer or claim a from-scratch training reproduction.

## Quick Start

HumanoidGPT requires Python 3.10 or newer and ONNX Runtime 1.18 or newer
because the official graph uses ONNX IR 10 / opset 20.

```bash
pip install -e ".[motion-tracking-mujoco]"
```

Run one policy step with the official 136D observation:

```python
import numpy as np
from motius import Pipeline

pipeline = Pipeline.from_pretrained(
    "ZeyuLing/Motius-HumanoidGPT-G1"
)
result = pipeline.infer_motion_tracking(
    np.zeros((1, 136), dtype=np.float32)
)
print(result["continuous_actions"].shape)  # (1, 29)
print(result["motor_targets"].shape)       # (1, 29)
```

Run the complete method-native observation adapter and physical controller:

```python
rollout = pipeline.rollout_motion_tracking(
    "checkpoints/datasets/lafan1_g1/walk1_subject1.npz",
    simulator="mujoco",
    max_steps=1000,
    output_path="outputs/motion_tracking/humanoid_gpt_walk1.npz",
)
print(rollout.metrics)
```

The rollout API canonicalizes the G1 reference clock, assembles every policy
observation field, applies the released gains and motor-target transform, and
persists execution qpos plus physical metrics.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Measured physical rollout.** Scores use the registered MuJoCo protocol `mujoco-g1-reference-tracking-50hz-v1` at 50 Hz.

| Setting | Coverage | Success ↑ | Completion ↑ | Local MPJPE ↓ | Joint MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MuJoCo · LAFAN1-G1 | 40 / 40 | 80.0% | 88.5% | 27.85 mm | 0.0700 rad |

[Canonical result pack](../leaderboards/hf_space_motion_tracking_mujoco/motion_tracking_results.json)

<!-- MOTIUS_CANONICAL_METRICS:END -->


**Protocol:** all 40 LAFAN1-G1 references, first 1,000 control steps, 50 Hz,
MuJoCo, reference-relative termination. GT is excluded from controller ranks.

| Coverage | Success ↑ | Completion ↑ | Local MPJPE ↓ | Joint MAE ↓ | Root drift ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 40 / 40 | 80.0% | 88.47% | 27.85 mm | 0.0700 rad | 651.11 mm |

These are fresh Motius measurements under the same engine and horizon as the
other MuJoCo rows. They are distinct from the older local 600-frame diagnostic
run and are backed by 40 persisted rollout files.

## Motion Representation

The released ONNX graph consumes `obs[B, 136]` in this exact order:

| Field | Dim |
| --- | ---: |
| Pelvis gyroscope and gravity | 6 |
| Current joint-position delta and velocity | 58 |
| Previous normalized action | 29 |
| Next reference joint-position delta | 29 |
| Reference pelvis height, gravity, and spatial velocity | 10 |
| Heading error `[cos, sin]` and heading-frame XY error | 4 |

The policy returns a normalized 29-DOF action. Motius applies the official
`nominal + action × 0.25 × torque_limit / stiffness` transform and simulates a
Unitree G1-5010. Root quaternions use `wxyz`; persisted trajectories use MuJoCo
G1 qpos `[root_xyz, root_quat_wxyz, 29 joints]`.

## Citation and License

```bibtex
@article{humanoidgpt2026,
  title={Humanoid-GPT: Humanoid Generative Pre-Training for Zero-Shot Motion Tracking},
  author={Qi, Zekun and Chen, Xuchuan and others},
  journal={arXiv preprint arXiv:2606.03985},
  year={2026}
}
```

HumanoidGPT is released under Apache-2.0. The checkpoint artifact preserves the
upstream license and the bundled Unitree G1 asset license.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
