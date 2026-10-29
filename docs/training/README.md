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

Motius composes neural ModelBundles, method Trainers and AccelerateRunner.
Online controller rollouts, physics and reference-data sampling are separate
framework components. BeyondMimic, ProtoMotions and SONIC use this native path.

## Training Support

The entries below expose public training launch configurations. Additional
trainer components under migration are listed separately. A runnable launcher
alone does not establish source parity or a complete reproduction protocol.

| Package | Support | Public config | Trainer | Output | Card |
| --- | --- | --- | --- | --- | --- |
| HY-Motion T2M / G1 | Full flow-matching training in HY-Motion-201 or native G1-38; optional weight warm start | [T2M config](../../configs/hymotion_t2m/train_hymotion_t2m.py) · [G1 config](../../configs/hymotion_t2m/train_hymotion_g1.py) | `HyMotionT2MTrainer` | `outputs/training/hymotion_t2m` · `outputs/training/hymotion_g1` | [Training section](../model_zoo/hymotion_t2m.md#training) |
| MotionStreamer | Native TAE, two-forward AR and streaming objectives; source component parity, CLI resume and two-GPU synchronization passed; full-dataset convergence pending | [`configs/motionstreamer/train_motionstreamer.py`](../../configs/motionstreamer/train_motionstreamer.py) | `MotionStreamerTrainer` | `outputs/training/motionstreamer_` + stage | [Training guide](motionstreamer.md) |
| FlowHMR | Official 491-D base flow-matching training | [`configs/flowhmr/train_flowhmr.py`](../../configs/flowhmr/train_flowhmr.py) | `FlowHMRTrainer` | `outputs/training/flowhmr` | [FlowHMR model card](../model_zoo/flowhmr.md) |
| GenTrack | Online G1 generator post-training with a lagged physical tracker, immutable-G0 control variate, Flow-GRPO replay, and tracker replay exchange | [`configs/gentrack/train_gentrack_g1.py`](../../configs/gentrack/train_gentrack_g1.py) · [offline control](../../configs/gentrack/train_gentrack_offline_g1.py) | `GenTrackFlowGRPOTrainer` | `outputs/training/gentrack_g1` | [GenTrack reproduction guide](gentrack.md) |
| PRISM | Continued training/fine-tuning from a released PRISM artifact | [`configs/prism/train_prism.py`](../../configs/prism/train_prism.py) | `PrismTrainer` | `outputs/training/prism` | [Training section](../model_zoo/prism.md#training) |
| AnimateCanvas | Full M2M training from a HY-Motion T2M warm start | [`configs/animatecanvas/train_animatecanvas_0p46b.py`](../../configs/animatecanvas/train_animatecanvas_0p46b.py) | `AnimateCanvasSparseRolloutJoinTrainer` | `outputs/training/animatecanvas_0p46b` | [Training section](../model_zoo/animatecanvas.md#training) |
| ProtoMotions | G1 physical motion tracking with native PPO | [`configs/motion_tracking/protomotions_g1_bones_seed.yaml`](../../configs/motion_tracking/protomotions_g1_bones_seed.yaml) | `ProtoMotionsPPOTrainer` | `outputs/training/protomotions` | [Training section](../model_zoo/protomotions.md#training) |
| SONIC | Native G1 physics/data, online PPO/auxiliary Trainer and shared Runner; bounded original-source physical/online checks and CLI resume passed | [`configs/motion_tracking/sonic_g1_bones_seed.yaml`](../../configs/motion_tracking/sonic_g1_bones_seed.yaml) | `SONICOnlinePPOTrainer` | `outputs/training/sonic` | [Training section](../model_zoo/sonic.md#training) |
| BeyondMimic | Native per-reference G1 PPO with AccelerateRunner; source physical rollout and update comparison passed | [`configs/motion_tracking/beyondmimic_g1_lafan1.yaml`](../../configs/motion_tracking/beyondmimic_g1_lafan1.yaml) | `BeyondMimicTrainer` | `outputs/training/beyondmimic` | [Training section](../model_zoo/beyondmimic.md#training) |
| Motius Joint-Position Evaluator | TMR reconstruction and contrastive training from scratch | [`configs/tmr/train_tmr_smpl22.py`](../../configs/tmr/train_tmr_smpl22.py) | `TMRTrainer` | `outputs/training/tmr_smpl22` | [Evaluator training section](../evaluator_zoo/motius_joint_position.md#training) |

This table describes training support in Motius, not whether an upstream
repository contains its own training script.

## Components under architecture migration

These native trainer components are being validated and are not promoted into
the protocol-complete release table above. Their precise acceptance status is
tracked in [the architecture migration](../architecture/native-methods.md).

| Component | Implementation | Current boundary |
| --- | --- | --- |
| `MotionMillionTrainer` | `motius/trainers/motionmillion/trainer.py` | Original real 7B/T5/FSQ inference, corrupted-token CE, reconstruction/velocity/acceleration losses and all gradients passed; full data and optimizer protocol pending |
| `TM2TTrainer` | `motius/trainers/tm2t/trainer.py` | Released-source beam decoding, summed caption CE, label smoothing and all gradients passed; tokenizer/GAN stage and full dataset protocol pending |
| `T2MGPTTrainer` | `motius/trainers/t2mgpt/trainer.py` | Native corrupted-token CE and VQ reconstruction/commitment objectives; released-source inference/loss/all-gradient/EMA checks passed; dataset protocol pending |
| `MotionCLIPTrainer` | `motius/trainers/motion_clip/trainer.py` | Contrastive objective and gradient gather outside the neural model; synthetic pre-migration GPU loss/gradient comparisons passed; independent released-weight and dataset validation pending |
| `GVHMRTrainer` | `motius/trainers/gvhmr/trainer.py` | Published masked regression, camera reprojection, root trajectory, vertices and contact objectives; dataset integration pending |
| `FlowMDMTrainer` | `motius/trainers/flowmdm/trainer.py` | Released masked x-start objective; complete dataset protocol pending |
| `MaskControlTrainer` | `motius/trainers/maskcontrol/trainer.py` | Released six-anchor logits regularizer objective; full dataset protocol pending |
| `GemSmplTrainer` / `GemXTrainer` | `motius/trainers/gem_smpl/trainer.py`, `motius/trainers/gem_x/trainer.py` | Native regression/diffusion and joint graphs; GEM-SMPL source objectives/all gradients passed; GEM-X source training and complete data/optimizer protocols pending |
| `MotionBricksPoseTrainer`, `MotionBricksRootTrainer`, `MotionBricksVQTrainer` | `motius/trainers/motionbricks/pose.py`, `motius/trainers/motionbricks/root.py`, `motius/trainers/motionbricks/vqvae.py` | All three original stage objectives, augmentation, active gradients and VQ EMA updates exact on synthetic normalized G1 inputs; full data/optimizer protocol pending |
| `PromptHMRVideoTrainer` | `motius/trainers/prompthmr/trainer.py` | All original temporal weighted losses, condition dropout and active gradients exact (520 comparisons on synthetic supervised inputs). Image training source and full dataset protocol unavailable |
| `VermoTrainer` | `motius/trainers/vermo/trainer.py` | Multimodal preparation and chunked weighted causal objectives outside the neural bundle; full released-weight migration losses/gradients and tied-head two-process checks passed; full protocol pending |
| `DARTTrainer` | `motius/trainers/dart/trainer.py` | Native primitive-denoiser and VAE objectives; full body/rollout protocol validation pending |
| `ViMoGenTrainer` | `motius/trainers/vimogen/trainer.py` | Native weighted/masked flow matching and reference corruption; source objective/all gradients and real two-GPU checkpointed training passed; dataset integration pending |
| `UniMuMoTrainer` | `motius/trainers/unimumo/trainer.py` | Native dual-stream token CE and caption objectives on encoded conditions; source losses/all gradients and two-process training passed; raw-data protocol and codec stage pending |
| `MotionGPT3Trainer` | `motius/trainers/motiongpt3/trainer.py` | Native modality CE, diffusion-head and continuous-VAE objectives; released-source losses/gradients passed; dataset and distributed VAE integration pending |
| `MotionGPTTrainer` | `motius/trainers/motiongpt/trainer.py` | Native instruction language-model and VQ reconstruction objectives; released-source loss/gradient comparison passed; dataset and distributed VQ integration pending |
| `MoGenTSTrainer` | `motius/trainers/mogents/trainer.py` | Native auxiliary and spatial-temporal base/residual token objectives; VQ and dataset integration pending |
| `MoMaskTrainer` | `motius/trainers/momask/trainer.py` | Base and residual token objectives; VQ stage pending |
| `BailandoTrainer` | [motius/trainers/bailando/trainer.py](../../motius/trainers/bailando/trainer.py) | Native supervised token and root-velocity decoder stages; actor-critic reinforcement protocol pending |
| `InterGenTrainer` | [motius/trainers/intergen/trainer.py](../../motius/trainers/intergen/trainer.py) | Native denoising/geometry objective; two-process smoke test; complete dataset protocol and original training comparison pending |
| `InterMaskTrainer` | [motius/trainers/intermask/trainer.py](../../motius/trainers/intermask/trainer.py) | Native masked-token objective and step-unroll stage; VQ training and Inter-X validation pending |
| `MoGenDiTTrainer` | [motius/trainers/mogendit/trainer.py](../../motius/trainers/mogendit/trainer.py) | Native restoration objective and [dedicated training entrypoint](../tasks/mogendit_training.md); full original trainer archive comparison pending |
| `YOLOXTrainer` | [motius/trainers/vision/yolox.py](../../motius/trainers/vision/yolox.py) | Shared vision objective extracted from the FlowHMR frontend; original training-gradient validation pending |

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
override distributed world size. All three use native neural Bundles, PPO Trainers and AccelerateRunner.
Their acceptance records distinguish physical deployment, objective/online
comparisons and distributed integration from full-dataset convergence.
Isaac Lab is a separately installed simulator dependency.

GenTrack coordinates the HYMotion-G1 bundle trainer and the native
ProtoMotions tracker through their framework entrypoints. See the
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
transformer in BF16, HY-Motion T2M uses BF16, and AnimateCanvas and TMR use
FP32 in their released recipes.

## Data

Download public sources through the [Dataset Hub](../datasets/README.md). The
compact PRISM/HY-Motion manifest and materialized TMR layouts are documented in
[PRISM, TMR, and HY-Motion data formats](prism_tmr_hymotion_t2m.md).

AnimateCanvas uses the exact mixture declared in its config:

- 30% HYMotion full-mask Text-to-Motion examples.
- 60% HYMotion arbitrary-condition examples.
- 5% MotionFix instruction edits.
- 5% PerMo style edits.

MotionFix and PerMo are public Motius dataset releases. HYMotion Data is not
currently distributed through the public Dataset Hub, so reproducing the exact
released AnimateCanvas mixture requires separately authorized HYMotion access.
The trainer remains usable with a replacement dataset mixture that emits the
same AnimateCanvas-198 batch contract, but that is a different training run and
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
