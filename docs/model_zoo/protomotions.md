<h1 align="center">ProtoMotions Model Card</h1>

<p align="center">
  <strong>A deployment-ready G1 motion tracker trained on BONES-SEED.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2507.14320">Paper</a> ·
  <a href="https://github.com/NVlabs/ProtoMotions">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-ProtoMotions-G1-BONES-SEED">Motius Checkpoint</a>
</p>

Motius packages ProtoMotions' official G1 BONES-SEED unified ONNX pipeline and
deployment YAML. Observation construction, policy inference, action decoding,
PD targets, and gains remain in the released graph/contract.

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
| Motion Tracking | Method-native reference motion and controller state | <video src="https://github.com/user-attachments/assets/9cca8d5c-0572-4157-9683-f9136d85480a" controls></video> | [MP4](https://github.com/user-attachments/assets/9cca8d5c-0572-4157-9683-f9136d85480a) · [All cases](https://zeyuling-motion-tracking-mujoco-leaderboard.static.hf.space/cases/index.html?method=protomotions) · [Leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) |

The policy-step API and physical rollout are evaluated under the registered MuJoCo or Isaac Lab protocol stated in the Model Card.

<!-- MOTIUS_TASK_DEMOS:END -->


The official BONES-SEED deployment policy has been evaluated on all 40
OpenTrack LAFAN1-G1 references in the shared MuJoCo setting.

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
| Training motion | 50 Hz G1 control with BONES-SEED reference motion |
| Public preview | 30 FPS media playback; unified ONNX runtime remains 50 Hz |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| --- | --- |
| Method | ProtoMotions G1 BONES-SEED deployment tracker |
| Tasks | Motion Tracking |
| Task status | MuJoCo metrics complete; synchronized rollout video public |
| Robot | Unitree G1, 29 actuated joints |
| Training | PPO tracking with AMP and transfer-oriented randomization |
| Native control rate | 50 Hz |
| Checkpoint | [`ZeyuLing/Motius-ProtoMotions-G1-BONES-SEED`](https://huggingface.co/ZeyuLing/Motius-ProtoMotions-G1-BONES-SEED) |
| Pipeline APIs | `infer_motion_tracking`, `rollout_motion_tracking` |

## Quick Start

```bash
pip install -e ".[motion-tracking-mujoco]"
```

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-ProtoMotions-G1-BONES-SEED"
)
result = pipe.infer_motion_tracking(
    {
        "current_anchor_rot": current_anchor_rot,
        "current_dof_pos": current_dof_pos,
        "current_dof_vel": current_dof_vel,
        "current_root_local_ang_vel": root_angular_velocity,
        "historical_processed_actions": previous_action[:, None],
        "mimic_future_anchor_rot": future_anchor_rot,
        "mimic_future_dof_pos": future_joint_pos,
        "mimic_future_dof_vel": future_joint_vel,
    }
)
joint_targets = result["joint_pos_targets"]
```

Future reference samples correspond to control steps `[1, 2, 4, 8]`, or
`[0.02, 0.04, 0.08, 0.16]` seconds. Quaternions use `xyzw`, as declared by the
official YAML.

```python
rollout = pipe.rollout_motion_tracking(
    "data/LAFAN1/Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
    simulator="mujoco",
    max_steps=1000,
    output_path="outputs/motion_tracking/protomotions_walk1.npz",
)
```

### Training

The native `ProtoMotionsPPOTrainer` (also exported as `ProtoMotionsTrainer`)
runs through the shared AccelerateRunner. Reference motion, expert observations,
rollout collection, PPO/AMP objectives and Isaac Lab physics are separate
framework components; no copied upstream application package is imported.

| Item | Training contract |
| --- | --- |
| Objective | Original PPO, L2C2, task/style value objectives and AMP discriminator with gradient penalty |
| Precision | FP32, with matmul precision matching the configured runtime |
| Checkpoints | Native Accelerate model/optimizer/RNG/progress state under `outputs/training/protomotions/`; `--auto-resume` restores native runs |

```bash
pip install -e ".[motion-tracking-train-protomotions]"
python tools/download_protomotions_assets.py
export G1_MOTION_FILE=/path/to/bones_seed_g1_train.motion
export PROTOMOTIONS_CHECKPOINT=/path/to/released_g1/last.ckpt
python tools/train_motion_tracking.py \
  configs/motion_tracking/protomotions_g1_bones_seed.yaml
```

Install Isaac Sim 5.1 and Isaac Lab 2.3.2 separately in their compatible Python
environment. The original checkpoint must be accompanied by its resolved
training and inference YAML/PT files. Original checkpoint loading starts with
released neural weights and fresh optimizers; native `--auto-resume` restores
complete framework checkpoints. The final current-weight Pipeline is exported
to the run's `exported` directory and loads through `Pipeline.from_pretrained`.

See [native training and validation details](../tasks/protomotions_native_training.md).
Four real G1 environments passed two 32x4 rollout/update iterations against
original AMP.fit, including actual reference-library expert observations and
all optimizer states. This uses a short synthetic reference and controlled
FP32/eager execution; it does not establish full-dataset convergence or
compiled-runtime equivalence. Published MuJoCo metrics below describe the
existing released deployment artifact.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Measured physical rollout.** Scores use the registered MuJoCo protocol `mujoco-g1-reference-tracking-50hz-v1` at 50 Hz.

| Setting | Coverage | Success ↑ | Completion ↑ | Local MPJPE ↓ | Joint MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| MuJoCo · LAFAN1-G1 | 40 / 40 | 45.0% | 72.8% | 48.01 mm | 0.1918 rad |

[Canonical result pack](../leaderboards/hf_space_motion_tracking_mujoco/motion_tracking_results.json)

<!-- MOTIUS_CANONICAL_METRICS:END -->


The canonical source is the [MuJoCo result pack](../leaderboards/hf_space_motion_tracking_mujoco/motion_tracking_results.json).
Errors are weighted over completed physical steps; GT is excluded from policy
ranking. This public checkpoint was trained on BONES-SEED rather than LAFAN1,
which is disclosed because it materially affects cross-domain tracking.

## Motion Representation

The policy tracks G1 joint-space reference motion. Current state contains
anchor rotation, 29D joint position/velocity, root angular velocity, and one
processed-action history frame. Reference state contains four future samples
of anchor rotation and 29D joint position/velocity. The artifact's
`unified_pipeline.yaml` is the source of truth for joint names, gains, timing,
and tensor order.

## Citation and License

See the [ProtoMotions paper](https://arxiv.org/abs/2507.14320),
[official source](https://github.com/NVlabs/ProtoMotions), and
[attribution note](../../motius/licenses/protomotions/ATTRIBUTIONS.md).
ProtoMotions and the redistributed artifact are Apache-2.0 licensed.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
