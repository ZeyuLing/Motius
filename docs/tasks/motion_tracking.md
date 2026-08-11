# Motion Tracking

Motius separates controller inference from physical rollout while exposing both
through one pipeline. `infer_motion_tracking` runs one method-native policy
step. `rollout_motion_tracking` owns reference loading, observation assembly,
closed-loop simulation, termination, metrics, and persisted trajectories.

<p align="center">
  <a href="README.md">Task Registry</a> ·
  <a href="../model_zoo/README.md">Model Zoo</a> ·
  <a href="../datasets/README.md">Dataset Hub</a> ·
  <a href="https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard">MuJoCo Leaderboard</a> ·
  <a href="https://huggingface.co/spaces/ZeyuLing/motion-tracking-isaaclab-leaderboard">Isaac Lab Leaderboard</a>
</p>

## Engine-Isolated Benchmarks

MuJoCo and Isaac Lab are separate settings. Physics, contacts, robot assets,
actuators, and numerical integration differ, so their rows are never mixed or
ranked against one another.

| Setting | Dataset | Protocol | Publication status |
| --- | --- | --- | --- |
| [MuJoCo](https://huggingface.co/spaces/ZeyuLing/motion-tracking-mujoco-leaderboard) | Official OpenTrack LAFAN1 Unitree G1 retarget, all 40 motions | 50 Hz; fixed first 1,000 steps; reference-relative termination | GT + Any2Track + ProtoMotions + HumanoidGPT metrics and all-case rollouts public |
| [Isaac Lab](https://huggingface.co/spaces/ZeyuLing/motion-tracking-isaaclab-leaderboard) | LAFAN1-G1 (40) + licensed AMASS-test-G1 (138) | GenTrack v0.20; one 30 FPS resample; fall-only success gate | GT + SONIC complete; BeyondMimic per-reference upper bound disclosed |

GT is shown first as a kinematic calibration row. It is excluded from
controller ranking because it does not execute a physical policy.

## Released Runtimes

| Method | Artifact | Policy input | Physical backend |
| --- | --- | --- | --- |
| [Any2Track](../model_zoo/any2track.md) | [`Motius-Any2Track-G1-LAFAN1-v2`](https://huggingface.co/ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2) | 156D observation or nine named components | MuJoCo measured |
| [ProtoMotions](../model_zoo/protomotions.md) | [`Motius-ProtoMotions-G1-BONES-SEED`](https://huggingface.co/ZeyuLing/Motius-ProtoMotions-G1-BONES-SEED) | Current state + four future references | MuJoCo measured |
| [HumanoidGPT](../model_zoo/humanoid_gpt.md) | [`Motius-HumanoidGPT-G1`](https://huggingface.co/ZeyuLing/Motius-HumanoidGPT-G1) | 136D proprioception, next-reference pose, and root command | MuJoCo measured |
| [SONIC](../model_zoo/sonic.md) | [`Motius-SONIC-G1`](https://huggingface.co/ZeyuLing/Motius-SONIC-G1) | 1,762D reference observation + 930D decoder state | Isaac Lab measured |
| [BeyondMimic](../model_zoo/beyondmimic.md) | Per-reference user-exported ONNX | Observation + reference time step | Isaac Lab measured upper bound; unranked |

All policy runtimes are self-contained and do not import an upstream checkout.
The HumanoidGPT artifact includes its complete G1-5010 MJCF and mesh tree.

## Training Support

SONIC, ProtoMotions, and BeyondMimic include their official simulator-owned
PPO training loops as fixed, attributed source snapshots inside Motius. They
are launched from public configs and write only below `outputs/training/`:

```bash
python tools/train_motion_tracking.py \
  configs/motion_tracking/sonic_g1_bones_seed.yaml
python tools/train_motion_tracking.py \
  configs/motion_tracking/protomotions_g1_bones_seed.yaml
python tools/train_motion_tracking.py \
  configs/motion_tracking/beyondmimic_g1_lafan1.yaml
```

See the [Training Hub](../training/README.md) and each Model Card for dataset
environment variables, Isaac Lab setup, distributed launch, and resume
behavior. HumanoidGPT is inference-only because its official release does not
include a trainer.

The complete GenTrack generator–tracker post-training implementation is
available separately from these controller baselines. It includes the
immutable-G0 Flow-GRPO update, lagged judge clock, replay exchange,
equal-budget controls, ablations, unified evaluator, and runtime attestation.
See the [GenTrack reproduction guide](../training/gentrack.md).

## MuJoCo Rollout

```bash
pip install -e ".[motion-tracking-mujoco]"
```

```python
from motius import Pipeline

pipeline = Pipeline.from_pretrained(
    "ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2"
)
rollout = pipeline.rollout_motion_tracking(
    "data/LAFAN1/Lafan1/mocap/UnitreeG1/walk1_subject1.npz",
    simulator="mujoco",
    max_steps=1000,
    output_path="outputs/motion_tracking/walk1_subject1.npz",
)
print(rollout.metrics)
```

Run the complete deterministic evaluation from the command line:

```bash
python tools/evaluate_motion_tracking.py \
  --model ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2 \
  --simulator mujoco \
  --reference 'data/LAFAN1/Lafan1/mocap/UnitreeG1/*.npz' \
  --window-steps 1000 --windows first \
  --output outputs/evaluation/motion_tracking/any2track_mujoco
```

The MuJoCo backend packages the exact G1 MJCF, observation/action adapters,
method-specific gains, 50 Hz controller loop, and physical diagnostics. Each
rollout artifact stores simulated qpos, reference qpos, actions, termination
reason, and metric metadata.

## Published Results

| Engine / split | Method | Coverage | Success | Completion | Local MPJPE |
| --- | --- | ---: | ---: | ---: | ---: |
| MuJoCo / LAFAN1-G1 | Any2Track | 40 / 40 | **100.0%** | **100.0%** | **14.76 mm** |
| MuJoCo / LAFAN1-G1 | ProtoMotions | 40 / 40 | 45.0% | 72.8% | 48.01 mm |
| MuJoCo / LAFAN1-G1 | HumanoidGPT | 40 / 40 | 80.0% | 88.47% | 27.85 mm |
| Isaac Lab / LAFAN1-G1 | SONIC | 40 / 40 | 87.5% | 99.8% | 40.06 mm |
| Isaac Lab / LAFAN1-G1 | BeyondMimic (upper bound) | 40 / 40 | 87.5% | 100.0% | 51.26 mm |
| Isaac Lab / AMASS-test-G1 | SONIC | 138 / 138 | 78.3% | 99.4% | 46.54 mm |
| Isaac Lab / AMASS-test-G1 | BeyondMimic (upper bound) | 100 / 138 | 93.0% | 100.0% | 53.60 mm |

GT is pinned first on both public pages and excluded from ranking. BeyondMimic
is also excluded because each policy is optimized for one reference. Its
AMASS result and viewer expose the actual 100/138 retained-fit coverage rather
than filling the missing 38 cases.

## Isaac Lab Result Lifecycle

The public Isaac Lab rows are migrated from the frozen GenTrack v0.20
fall-only experiment artifacts. The result pack records this provenance, and
the all-case viewer renders the persisted canonical qpos30 reference and
execution trajectories. It never substitutes MuJoCo scores or visualization
rollouts. Licensed AMASS-derived qpos files remain in the public visualization
dataset only under the applicable source-data terms.

The lightweight `Pipeline.from_pretrained` policy-step contract remains
separate from Isaac Sim startup. Reproducing the physical rollout requires the
method-native Isaac Lab environment; importing the checkpoint does not launch
Isaac Sim implicitly.

## Protocol Metrics

| Metric | Definition |
| --- | --- |
| Success | Episode reaches the fixed horizon without a registered termination |
| Completion | Completed physical trajectory frames divided by scheduled frames |
| Local MPJPE | Protocol-defined root-relative G1 body-position error |
| Joint MAE | Mean absolute 29-DOF joint-position error |
| Global MPJPE / root drift | World-space tracking error, reported separately from local pose quality |
| Foot slip | Horizontal foot speed while the corresponding foot is near the floor |
| Mechanical power | Sum of absolute joint torque multiplied by joint velocity |

MuJoCo pose and control errors are weighted by completed physical steps.
GenTrack continuous errors include both successful and failed trajectories.
Success is averaged over references. GT is calibration-only and never receives
a controller rank.

## Reference Data

- [LAFAN1-G1](../datasets/README.md#lafan1) uses the public OpenTrack retarget:
  40 named 23-DOF trajectories at 40 Hz. Motius inserts the six absent G1
  joints as zero and resamples to 50 Hz with cubic translation/joint
  interpolation plus quaternion SLERP.
- [AMASS](../datasets/README.md#amass) is user licensed. Motius does not make a
  public AMASS-G1 derivative; retargeted files stay under `outputs/` unless the
  user has distribution rights.

## Train And Export BeyondMimic

BeyondMimic does not publish a named pretrained policy. Motius includes the
source-pinned official trainer and packages an authorized ONNX export:

```bash
export BEYONDMIMIC_MOTION_FILE=/path/to/official_format_motion.npz
python tools/train_motion_tracking.py \
  configs/motion_tracking/beyondmimic_g1_lafan1.yaml

python tools/export_motion_tracking_hf.py beyondmimic \
  --source outputs/training/beyondmimic/logs/rsl_rl/g1_flat/RUN/exported/policy.onnx \
  --output outputs/checkpoints/beyondmimic
```

See the [BeyondMimic Model Card](../model_zoo/beyondmimic.md#training) for the
verified robot-asset download, official motion preprocessing, automatic
resume, and checkpoint-to-ONNX commands.

## Validation

```bash
pytest -q \
  tests/test_motion_tracking_artifacts.py \
  tests/test_motion_tracking_simulators.py \
  tests/test_motion_tracking_trainers.py \
  tests/test_motion_tracking_reproduction.py
python tools/audit_pipeline_task_apis.py
python tools/audit_leaderboards.py
```

`tools/verify_motion_tracking_reproduction.py --require-complete` is the
release gate for downloaded artifact inventories, finite policy outputs,
fresh-versus-baseline physical replays, full trainer update/resume/export
evidence, and viewer/result case alignment. The tests also cover reference
resampling, joint-name conversion, and a real MuJoCo closed-loop smoke rollout.
