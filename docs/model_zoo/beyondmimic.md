<h1 align="center">BeyondMimic Model Card</h1>

<p align="center">
  <strong>Official BeyondMimic training and deployment in one Motius package.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2508.08241">Paper</a> ·
  <a href="https://github.com/HybridRobotics/whole_body_tracking">Original GitHub</a> ·
  <a href="https://github.com/HybridRobotics/motion_tracking_controller">Deployment GitHub</a>
</p>

Motius packages the official BeyondMimic Isaac Lab/RSL-RL trainer together
with its ONNX input, output, and metadata contract. It does not import an
external checkout. The upstream release does not provide a named pretrained
policy, so artifacts are exported from the user's own training run.

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
| Motion Tracking | Method-native reference motion and controller state | <video src="https://github.com/user-attachments/assets/d6191b1a-e8f2-4626-b9ba-bb0b8f29d013" controls></video> | [MP4](https://github.com/user-attachments/assets/d6191b1a-e8f2-4626-b9ba-bb0b8f29d013) · [All cases](https://zeyuling-motion-tracking-isaaclab-leaderboard.static.hf.space/cases/index.html?method=beyondmimic) · [Leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-tracking-isaaclab-leaderboard) |

The policy-step API and physical rollout are evaluated under the registered MuJoCo or Isaac Lab protocol stated in the Model Card.

<!-- MOTIUS_TASK_DEMOS:END -->


No redistributable general-purpose pretrained policy is available. The public
rollouts are result artifacts from authorized per-reference experiment exports;
Motius does not relabel them as one reusable checkpoint.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Motion Tracking | `infer_motion_tracking` | [Task and runtime contract](https://huggingface.co/spaces/ZeyuLing/motion-tracking-isaaclab-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 50 Hz G1 control for the official deployment recipe |
| Public preview | Canonical qpos playback at 30 fps |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| --- | --- |
| Method | BeyondMimic whole-body tracking |
| Tasks | Motion Tracking |
| Task status | Trainable user-checkpoint integration |
| Robot | Unitree G1 |
| Native control rate | 50 Hz |
| Public pretrained checkpoint | Not released upstream |
| Pipeline API | `infer_motion_tracking` |

The ONNX graph embeds the complete reference motion. Each call consumes policy
observation `obs` plus `time_step`, then returns action, reference joint state,
and reference body state. Joint order, gains, defaults, action scale,
observation terms, anchor, and body names are read from ONNX metadata.

## Quick Start

Export `policy.onnx` with the official BeyondMimic exporter, then create a
local self-describing artifact:

```bash
python tools/export_motion_tracking_hf.py beyondmimic \
  --source /path/to/policy.onnx \
  --output outputs/checkpoints/beyondmimic
```

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained("outputs/checkpoints/beyondmimic")
result = pipe.infer_motion_tracking(observation, time_step=0)
action = result["actions"]
reference_joint_position = result["joint_pos"]
```

This local artifact can be uploaded to a private Hugging Face repository after
checking the motion/checkpoint licenses.

### Training

The official BeyondMimic environment, PPO loop, observation terms, rewards,
terminations, checkpointing, and ONNX exporter are source-pinned under
[`motius/trainers/beyondmimic`](../../motius/trainers/beyondmimic) at upstream
commit `cd65172032893724b445448818c34165846d847d`. Training uses one
official-format G1 reference motion per policy.

| Item | Training contract |
| --- | --- |
| Objective | Clipped PPO with value loss and entropy regularization; global anchor pose, relative body pose, body velocity, action-rate, joint-limit, and undesired-contact terms |
| Precision | FP32 optimization with the official TF32 CUDA matmul/cuDNN settings; AMP is not enabled |
| Checkpoints | Complete RSL-RL `model_*.pt` state every 500 iterations; the released config runs 30,000 iterations |

Install Isaac Sim 4.5 and Isaac Lab 2.1 first, then install the Motius
training dependencies and verified Unitree G1 description:

```bash
pip install -e ".[motion-tracking-train-beyondmimic]"
python tools/download_beyondmimic_assets.py
```

The trainer consumes the official maximum-coordinate NPZ with `fps`,
`joint_pos`, `joint_vel`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and
`body_ang_vel_w`. Convert a retargeted G1 CSV using the packaged official
Isaac Lab preprocessing path:

```bash
python -m motius.trainers.beyondmimic.prepare_motion \
  --input_file /path/to/lafan1_g1.csv \
  --input_fps 30 \
  --output_fps 50 \
  --output_file outputs/training/beyondmimic/motions/lafan1_g1.npz \
  --headless
```

Launch the checked-in configuration:

```bash
export BEYONDMIMIC_MOTION_FILE="$PWD/outputs/training/beyondmimic/motions/lafan1_g1.npz"
python tools/train_motion_tracking.py \
  configs/motion_tracking/beyondmimic_g1_lafan1.yaml
```

The config includes `--auto-resume`; the wrapper finds the newest
`model_*.pt` under the same experiment directory and restores policy,
optimizer, normalizer, and iteration state. All run files remain below
`outputs/training/beyondmimic/`.

Export the official metadata-bearing ONNX from a saved run:

```bash
python -m motius.trainers.beyondmimic.export_policy \
  --motion_file "$BEYONDMIMIC_MOTION_FILE" \
  --load_run RUN_DIRECTORY \
  --checkpoint model_30000.pt \
  --headless
```

Then package it for `Pipeline.from_pretrained`:

```bash
python tools/export_motion_tracking_hf.py beyondmimic \
  --source outputs/training/beyondmimic/logs/rsl_rl/g1_flat/RUN_DIRECTORY/exported/policy.onnx \
  --output outputs/checkpoints/beyondmimic
```

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Measured physical rollout.** Scores use the registered Isaac Lab protocol `gentrack-fall-only-30fps-v020` at 50 Hz.

| Setting | Coverage | Success ↑ | Completion ↑ | Local MPJPE ↓ | Joint MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isaac Lab · LAFAN1-G1 | 40 / 40 | 87.5% | 100.0% | 51.26 mm | 0.1009 rad |
| Isaac Lab · AMASS-test-G1 | 100 / 138 | 93.0% | 100.0% | 53.60 mm | 0.1287 rad |

> Per-reference optimization upper bound; unranked and coverage-disclosed.

[Canonical result pack](../leaderboards/hf_space_motion_tracking_isaaclab/motion_tracking_results.json)

<!-- MOTIUS_CANONICAL_METRICS:END -->


The runtime is tested against the official export signature: inputs `obs` and
`time_step`; outputs `actions`, joint position/velocity, and world-space body
position, quaternion, linear velocity, and angular velocity. The published
GenTrack v0.20 scores and persisted qpos30 rollouts retain their per-reference
provenance and are excluded from controller ranking.

## Motion Representation

BeyondMimic's ONNX checkpoint couples a learned policy with one embedded G1
reference trajectory. The trajectory stores 29D joint position/velocity and
world-space body states; the controller aligns its initial reference anchor to
the current robot position and heading before constructing observations.

## Citation and License

See the [BeyondMimic paper](https://arxiv.org/abs/2508.08241),
[official training source](https://github.com/HybridRobotics/whole_body_tracking),
[deployment source](https://github.com/HybridRobotics/motion_tracking_controller),
and [attribution note](../../motius/models/beyondmimic/ATTRIBUTIONS.md).

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
