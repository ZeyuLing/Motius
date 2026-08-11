<h1 align="center">Any2Track Model Card</h1>

<p align="center">
  <strong>An OpenTrack G1 generalist that tracks all 40 LAFAN1 motions.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2509.13833">Paper</a> ·
  <a href="https://github.com/GalaxyGeneralRobotics/OpenTrack">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2">Motius Checkpoint</a>
</p>

Motius packages the official Any2Track LAFAN1 generalist v2 ONNX policy and
training config. The release uses the paper/project name Any2Track; OpenTrack is
the upstream training and deployment repository.

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
| Motion Tracking | Method-native reference motion and controller state | <video src="https://github.com/user-attachments/assets/dbf58602-9523-4377-8b28-f52a711954d7" controls></video> | [MP4](https://github.com/user-attachments/assets/dbf58602-9523-4377-8b28-f52a711954d7) · [All cases](https://zeyuling-motion-tracking-mujoco-leaderboard.static.hf.space/cases/index.html?method=any2track) · [Leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) |

The policy-step API and physical rollout are evaluated under the registered MuJoCo or Isaac Lab protocol stated in the Model Card.

<!-- MOTIUS_TASK_DEMOS:END -->


All 40 official OpenTrack LAFAN1-G1 motions have been evaluated in the shared
MuJoCo setting. Isaac Lab is maintained as a separate leaderboard.

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
| Training motion | 50 Hz G1 control after reference-motion preprocessing |
| Public preview | 50 Hz MuJoCo rollout; public media encoded at 30 fps |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| --- | --- |
| Method | Any2Track LAFAN1 generalist v2 |
| Tasks | Motion Tracking |
| Task status | MuJoCo metrics complete; synchronized rollout video public |
| Robot | Unitree G1, 29 actuated joints |
| Reference set | LAFAN1, 40 motions |
| Native control rate | 50 Hz |
| Checkpoint | [`ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2`](https://huggingface.co/ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2) |
| Pipeline APIs | `infer_motion_tracking`, `rollout_motion_tracking` |

The official files are preserved byte-for-byte. Their SHA-256 values are
`1320bddd4b981876e84243c0d336c8eaef57e202499b694cfbdc626fab3972c1`
for `model.onnx` and
`3dac8c821f0b2326fc1644a8d4626e84a27b23cf74762f7d04881bf8b7082c86`
for `config.json`.

## Quick Start

```bash
pip install -e ".[motion-tracking-mujoco]"
```

```python
import numpy as np
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2"
)
result = pipe.infer_motion_tracking(
    np.zeros((1, 156), dtype=np.float32)
)
action = result["continuous_actions"]  # [1, 29]
```

The API also accepts a mapping of the nine named components in `config.json`;
Motius concatenates them in the official order and rejects missing, extra, or
wrong-width input.

Run a closed-loop physical rollout with the same pipeline:

```python
rollout = pipe.rollout_motion_tracking(
    "data/LAFAN1/Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
    simulator="mujoco",
    max_steps=1000,
    output_path="outputs/motion_tracking/any2track_walk1.npz",
)
print(rollout.metrics)
```

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Measured physical rollout.** Scores use the registered MuJoCo protocol `mujoco-g1-reference-tracking-50hz-v1` at 50 Hz.

| Setting | Coverage | Success ↑ | Completion ↑ | Local MPJPE ↓ | Joint MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MuJoCo · LAFAN1-G1 | 40 / 40 | 100.0% | 100.0% | 14.76 mm | 0.0632 rad |

[Canonical result pack](../leaderboards/hf_space_motion_tracking_mujoco/motion_tracking_results.json)

<!-- MOTIUS_CANONICAL_METRICS:END -->


The canonical source is the [MuJoCo result pack](../leaderboards/hf_space_motion_tracking_mujoco/motion_tracking_results.json).
Errors are weighted over completed physical steps; GT is excluded from policy
ranking. The official ONNX IR v10 graph also passes direct inference and
returns a finite `[1, 29]` action for a `[1, 156]` observation.

## Motion Representation

The 156D observation is assembled from joint-position/velocity errors, pelvis
gravity and gyroscope signals, current joint position/velocity, previous motor
targets, four reference foot heights, and reference root height. The action is
a 29D residual target interpreted by the official G1 environment config.

## Citation and License

See the [Any2Track paper](https://arxiv.org/abs/2509.13833),
[official OpenTrack source](https://github.com/GalaxyGeneralRobotics/OpenTrack),
and [attribution note](../../motius/models/any2track/ATTRIBUTIONS.md). OpenTrack
and the redistributed artifact are Apache-2.0 licensed.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
