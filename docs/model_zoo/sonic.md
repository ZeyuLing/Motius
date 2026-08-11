<h1 align="center">SONIC Model Card</h1>

<p align="center">
  <strong>Universal-token whole-body tracking for the Unitree G1.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2511.07820">Paper</a> ·
  <a href="https://github.com/NVlabs/GR00T-WholeBodyControl">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-SONIC-G1">Motius Checkpoint</a>
</p>

Motius packages the official default SONIC encoder, decoder, and observation
configuration behind a self-contained `SONICPipeline`. No upstream checkout or
remote code execution is required.

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
| Motion Tracking | Method-native reference motion and controller state | <video src="https://github.com/user-attachments/assets/19b1ef53-773f-4362-b53c-4da3068dcd97" controls></video> | [MP4](https://github.com/user-attachments/assets/19b1ef53-773f-4362-b53c-4da3068dcd97) · [All cases](https://zeyuling-motion-tracking-isaaclab-leaderboard.static.hf.space/cases/index.html?method=sonic) · [Leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-tracking-isaaclab-leaderboard) |

The policy-step API and physical rollout are evaluated under the registered MuJoCo or Isaac Lab protocol stated in the Model Card.

<!-- MOTIUS_TASK_DEMOS:END -->


Motius maintains separate MuJoCo and Isaac Lab settings. SONIC is assigned to
the Isaac Lab setting because publishing its score through a convenient MuJoCo
port would change the method-native physical protocol.

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
| Training motion | 50 Hz G1 control with 20 ms reference sampling |
| Public preview | 30 FPS media playback; encoder/decoder runtime remains 50 Hz |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| --- | --- |
| Method | SONIC universal motion tracker |
| Tasks | Motion Tracking |
| Task status | Runtime integration |
| Robot | Unitree G1, 29 actuated joints |
| Native control rate | 50 Hz |
| Checkpoint | [`ZeyuLing/Motius-SONIC-G1`](https://huggingface.co/ZeyuLing/Motius-SONIC-G1) |
| Pipeline API | `infer_motion_tracking` |

The encoder maps a 1,762D reference/command observation to a 64D universal
motion token. Motius prepends that token to the 930D decoder state exactly as
the official export expects, producing the 994D decoder input and a 29D action.

## Quick Start

```bash
pip install -e ".[motion-tracking]"
```

```python
import numpy as np
from motius import Pipeline

pipe = Pipeline.from_pretrained("ZeyuLing/Motius-SONIC-G1")
result = pipe.infer_motion_tracking(
    encoder_observation=np.zeros((1, 1762), dtype=np.float32),
    decoder_observation=np.zeros((1, 930), dtype=np.float32),
)
action = result["action"]  # [1, 29]
```

The zero tensors only demonstrate the API. A physical rollout must assemble
the documented reference/history observations at 50 Hz and apply the action
with the matching G1 controller and safety limits.

### Training

SONIC's released PPO/TRL trainer is packaged under
[`motius/trainers/sonic`](../../motius/trainers/sonic). It uses the official
Isaac Lab environment, rewards, optimizer, universal-token auxiliary losses,
checkpoint cadence, and Hydra config without importing another repository.
Use Python 3.10 or newer, install Isaac Lab first, then the Python training
dependencies:

| Item | Training contract |
| --- | --- |
| Objective | Clipped PPO policy/value objectives, entropy regularization, physical tracking rewards, G1 reconstruction, and cross-representation latent MSE losses |
| Precision | FP32 by default; AMP is enabled only when `algo.config.use_amp=true` is explicitly supplied |
| Checkpoints | Full `last.pt` state every 50 global steps and numbered `model_step_*.pt` snapshots every 2,000 global steps |

```bash
pip install -e ".[motion-tracking-train-sonic]"
```

Set the two processed BONES-SEED motion roots and launch the checked-in
configuration:

```bash
export G1_MOTION_DIR=/path/to/bones_seed/g1_motion_lib
export SMPL_MOTION_DIR=/path/to/bones_seed/smpl_motion_lib
python tools/train_motion_tracking.py \
  configs/motion_tracking/sonic_g1_bones_seed.yaml
```

Use `--num-processes 64` to override the configured process count. Hydra
overrides can follow the config path. To continue an interrupted run, add
`resume=true experiment_dir=outputs/training/sonic/TRL_G1_Track/<run>`.
Checkpoints and logs remain under `outputs/training/sonic/`.

Export a native checkpoint into a complete Pipeline artifact:

```bash
python -m motius.trainers.sonic.export_policy \
  --checkpoint outputs/training/sonic/TRL_G1_Track/<run>/last.pt \
  --output outputs/checkpoints/sonic
```

This command runs the method-native encoder/decoder export, records the
resolved observation config, and writes `sonic_config.json` plus
`model_index.json`. Encoder and decoder dimensions are read from the exported
graphs rather than assumed from the public pretrained checkpoint.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Measured physical rollout.** Scores use the registered Isaac Lab protocol `gentrack-fall-only-30fps-v020` at 50 Hz.

| Setting | Coverage | Success ↑ | Completion ↑ | Local MPJPE ↓ | Joint MAE ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Isaac Lab · LAFAN1-G1 | 40 / 40 | 87.5% | 99.8% | 40.06 mm | 0.1348 rad |
| Isaac Lab · AMASS-test-G1 | 138 / 138 | 78.3% | 99.4% | 46.54 mm | 0.1616 rad |

[Canonical result pack](../leaderboards/hf_space_motion_tracking_isaaclab/motion_tracking_results.json)

<!-- MOTIUS_CANONICAL_METRICS:END -->


The released artifact passes an exact tensor-contract test and real ONNX
forward test: encoder output `[1, 64]`, decoder input `[1, 994]`, and finite
action output `[1, 29]`. The [Isaac Lab leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-tracking-isaaclab-leaderboard)
publishes the complete metric set and all-case persisted rollout viewer.

## Motion Representation

SONIC consumes controller observations rather than a standalone animation
tensor. Its G1 reference mode uses future 29-DOF joint positions, velocities,
and anchor orientations; the policy state uses measured joint/history, action,
angular-velocity, and gravity terms. The included `observation_config.yaml`
defines the authoritative concatenation order.

## Citation and License

See the [SONIC paper](https://arxiv.org/abs/2511.07820),
[official source](https://github.com/NVlabs/GR00T-WholeBodyControl), and
[attribution note](../../motius/models/sonic/ATTRIBUTIONS.md). The model remains
subject to the NVIDIA Open Model License included with the checkpoint.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
