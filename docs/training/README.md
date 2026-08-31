<!-- MOTIUS_DOCS_NAV:START -->
<p><a href="../../README.md"><strong>Motius</strong></a> / <a href="../README.md">Documentation</a> / <strong>Training Hub</strong></p>
<p>
  <a href="../getting_started.md">Quickstart</a> ·
  <a href="../tasks/README.md">Tasks</a> ·
  <a href="../datasets/README.md">Datasets</a> ·
  <a href="../model_zoo/README.md">Models</a> ·
  <strong>Training</strong> ·
  <a href="../evaluator_zoo/README.md">Evaluators</a> ·
  <a href="../leaderboards/README.md">Benchmarks</a> ·
  <a href="../motion/README.md">Motion I/O</a>
</p>
<!-- MOTIUS_DOCS_NAV:END -->

# Motius Training Hub

Motius exposes two explicit training runtime families. Generative and
evaluator models use `ModelBundle` + method `Trainer` + `AccelerateRunner`.
Physical controllers keep their simulator-owned PPO loop because environment
stepping, rollout storage, and optimizer cadence form one algorithmic unit.
Both families use checked-in configs, write below `outputs/training/`, and
resume complete training state.

## Training Support

Only the entries below currently expose a Motius-native trainer and a public
configuration. Every other Model Zoo package is checkpoint/inference support
unless its card is promoted into this table.

| Package | Support | Public config | Trainer | Output | Card |
| --- | --- | --- | --- | --- | --- |
| HY-Motion T2M / G1 | Full flow-matching training in HY-Motion-201 or native G1-38; optional weight warm start | [T2M config](../../configs/hymotion_t2m/train_hymotion_t2m.py) · [G1 config](../../configs/hymotion_t2m/train_hymotion_g1.py) | `HyMotionT2MTrainer` | `outputs/training/hymotion_t2m` · `outputs/training/hymotion_g1` | [Training section](../model_zoo/hymotion_t2m.md#training) |
| GenTrack | Online G1 generator post-training with a lagged physical tracker, immutable-G0 control variate, Flow-GRPO replay, and tracker replay exchange | [`configs/gentrack/train_gentrack_g1.py`](../../configs/gentrack/train_gentrack_g1.py) · [offline control](../../configs/gentrack/train_gentrack_offline_g1.py) | `GenTrackFlowGRPOTrainer` | `outputs/training/gentrack_g1` | [GenTrack reproduction guide](gentrack.md) |
| PRISM | Continued training/fine-tuning from a released PRISM artifact | [`configs/prism/train_prism.py`](../../configs/prism/train_prism.py) | `PrismTrainer` | `outputs/training/prism` | [Training section](../model_zoo/prism.md#training) |
| MotionCanvas | Full M2M training from a HY-Motion T2M warm start | [`configs/motioncanvas/train_motioncanvas_0p46b.py`](../../configs/motioncanvas/train_motioncanvas_0p46b.py) | `MotionCanvasSparseRolloutJoinTrainer` | `outputs/training/motioncanvas_0p46b` | [Training section](../model_zoo/motioncanvas.md#training) |
| ProtoMotions | G1 physical motion tracking with native PPO | [`configs/motion_tracking/protomotions_g1_bones_seed.yaml`](../../configs/motion_tracking/protomotions_g1_bones_seed.yaml) | `ProtoMotionsTrainer` | `outputs/training/protomotions` | [Training section](../model_zoo/protomotions.md#training) |
| SONIC | G1 universal-token tracking with PPO and auxiliary losses | [`configs/motion_tracking/sonic_g1_bones_seed.yaml`](../../configs/motion_tracking/sonic_g1_bones_seed.yaml) | `SonicTrainer` | `outputs/training/sonic` | [Training section](../model_zoo/sonic.md#training) |
| BeyondMimic | Per-reference G1 tracking with the official Isaac Lab/RSL-RL PPO loop | [`configs/motion_tracking/beyondmimic_g1_lafan1.yaml`](../../configs/motion_tracking/beyondmimic_g1_lafan1.yaml) | `BeyondMimicTrainer` | `outputs/training/beyondmimic` | [Training section](../model_zoo/beyondmimic.md#training) |
| Motius Joint-Position Evaluator | TMR reconstruction and contrastive training from scratch | [`configs/tmr/train_tmr_smpl22.py`](../../configs/tmr/train_tmr_smpl22.py) | `TMRTrainer` | `outputs/training/tmr_smpl22` | [Evaluator training section](../evaluator_zoo/motius_joint_position.md#training) |

This table describes training support in Motius, not whether an upstream
repository contains its own training script.

## Launch

### Bundle Trainers

Single process:

```bash
python tools/train.py CONFIG \
  --work-dir outputs/training/RUN_NAME \
  --auto-resume
```

One machine with eight GPUs:

```bash
bash tools/dist_train.sh CONFIG 8 \
  --work-dir outputs/training/RUN_NAME \
  --auto-resume
```

Multi-node launch uses the same command on every node:

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 \
bash tools/dist_train.sh CONFIG 8 \
  --work-dir outputs/training/RUN_NAME \
  --auto-resume
```

Set `NODE_RANK=1` on the second node. `GPUS` is the number of processes per
node; total world size is `NNODES * GPUS`.

## Resume And Warm Start

| Mode | Command | Restored state |
| --- | --- | --- |
| Automatic resume | `--auto-resume` | Latest model, optimizer, scheduler, epoch/iteration, and distributed state under the work directory |
| Explicit full resume | `--load-from PATH --load-scope full` | Full training state from the named checkpoint |
| Weight warm start | `--load-from PATH --load-scope model` | Model parameters only; optimizer and counters start fresh |

Do not combine checkpoints from different representations or normalizers.
Automatic resume is scoped to one explicit work directory so an unrelated run
cannot be selected accidentally.

## Configuration Overrides

MMEngine-style dotted options allow small launch-time changes without copying
a config:

```bash
bash tools/dist_train.sh configs/prism/train_prism.py 8 \
  --work-dir outputs/training/prism_large_batch \
  --auto-resume \
  --cfg-options \
    train_dataloader.batch_size=16 \
    train_dataloader.num_workers=12 \
    train_cfg.max_epochs=200 \
    optimizer.lr=5e-5
```

### Physical Controller Trainers

SONIC, ProtoMotions, and BeyondMimic are launched through the
simulator-training dispatcher:

```bash
python tools/train_motion_tracking.py \
  configs/motion_tracking/sonic_g1_bones_seed.yaml
```

```bash
python tools/train_motion_tracking.py \
  configs/motion_tracking/protomotions_g1_bones_seed.yaml
```

```bash
export BEYONDMIMIC_MOTION_FILE=/path/to/official_format_motion.npz
python tools/train_motion_tracking.py \
  configs/motion_tracking/beyondmimic_g1_lafan1.yaml
```

Use `--dry-run` to inspect the exact command and `--num-processes N` to
override distributed world size. These adapters invoke source snapshots under
`motius/trainers/{method}/vendor`; they never import an external checkout.
Isaac Lab remains a separately installed simulator dependency.

GenTrack coordinates the HYMotion-G1 bundle trainer and the vendored
ProtoMotions tracker rather than hiding either optimization loop. See the
[GenTrack reproduction guide](gentrack.md) for artifact inputs, the lagged
judge clock, single-round launch, multi-round launch, ablations, and validation.

BeyondMimic additionally needs the source-pinned Unitree G1 description:

```bash
python tools/download_beyondmimic_assets.py
```

The batch size is per process. Effective global batch size is:

```text
batch_size_per_process * world_size * gradient_accumulation_steps
```

Mixed precision is declared by each config under `accelerator`. Do not
override it casually: PRISM keeps its VAE path in FP32 while training the
transformer in BF16, HY-Motion T2M uses BF16, and MotionCanvas and TMR use
FP32 in their released recipes.

## Data

Download public sources through the [Dataset Hub](../datasets/README.md). The
compact PRISM/HY-Motion manifest and materialized TMR layouts are documented in
[PRISM, TMR, and HY-Motion data formats](prism_tmr_hymotion_t2m.md).

MotionCanvas uses the exact mixture declared in its config:

- 30% HYMotion full-mask Text-to-Motion examples.
- 60% HYMotion arbitrary-condition examples.
- 5% MotionFix instruction edits.
- 5% PerMo style edits.

MotionFix and PerMo are public Motius dataset releases. HYMotion Data is not
currently distributed through the public Dataset Hub, so reproducing the exact
released MotionCanvas mixture requires separately authorized HYMotion access.
The trainer remains usable with a replacement dataset mixture that emits the
same MotionCanvas-198 batch contract, but that is a different training run and
must not be presented as an exact reproduction.

## Outputs

Training outputs always live below `outputs/training/`. A typical run contains:

```text
outputs/training/RUN_NAME/
  YYYYMMDD_HHMMSS/
    train.log
    config.py
  checkpoint-epoch_*/
```

The exact checkpoint names are hook-defined. Public configs save every epoch,
retain a bounded history, and save the final state. Dataset caches,
pre-extracted embeddings, and body-model assets belong under `data/` or
`checkpoints/`, never under the training output directory.

## Package Contract

A method is advertised as trainable only when all of the following exist:

1. A registered trainer under `motius/trainers/{method}/`.
2. A public config under `configs/{method}/`.
3. A Model Card training section with data, precision, loss, launch, resume,
   checkpoint, and output details.
4. A test that imports the config and resolves its bundle and trainer.

Upstream-only training code or an inference configuration does not satisfy this
contract.

## Validation

Run the training release gates before publishing a trainer or changing its
documentation:

```bash
python tools/audit_training_release.py
python tools/audit_training_docs.py
pytest -q tests/test_training_release.py tests/test_training_docs.py
```
