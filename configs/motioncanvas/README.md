# MotionCanvas Training

`train_motioncanvas_0p46b.py` is the release recipe for the 0.46B
MotionCanvas checkpoint. It uses the same 30 fps, 360-frame, 198-D motion
space as the public artifact.

See the [Motius Training Hub](../../docs/training/README.md) for distributed
launch, resume semantics, supported-package boundaries, and output rules.

## Data Mix

Each 630,000-sample training epoch draws:

| Source | Weight | Supervision |
| --- | ---: | --- |
| HYMotion full-mask data | 30% | Text-to-motion |
| HYMotion arbitrary conditions | 60% | Completion and control |
| MotionFix | 5% | Motion editing |
| PerMo | 5% | Motion editing |

The dataset paths in the config are deployment-specific. Set them to the
materialized HYMotion and MotionHub roots on the training cluster.

The exact MotionCanvas-198 `Mean.npy`, `Std.npy`, normalization provenance,
and 22-joint FK rest offsets are small tracked files under
`checkpoints/models/motioncanvas/`. The training recipe references them
explicitly and does not depend on normalizer or body-model files under
`data/`.

## Train

```bash
bash tools/dist_train.sh configs/motioncanvas/train_motioncanvas_0p46b.py 8 \
  --work-dir outputs/training/motioncanvas_0p46b \
  --auto-resume
```

The config enables automatic resume. Logs and checkpoints are kept below
`outputs/training/motioncanvas_0p46b`; no training output is written to the
repository root.

The released run uses FP32, AdamW with a `1e-5` learning rate, and batch size
96 per process. Distributed launch and process count are supplied by the
cluster launcher rather than hard-coded in this config.

## Export

```bash
python tools/export_motioncanvas_hf.py \
  --checkpoint outputs/training/motioncanvas_0p46b/checkpoint-epoch_2354 \
  --stats-dir /path/to/motioncanvas_198_stats \
  --bone-offsets /path/to/bone_offsets_22.pt \
  --llm-path /path/to/Qwen3-8B \
  --sentence-encoder-path /path/to/clip-vit-large-patch14 \
  --output-dir outputs/releases/motioncanvas_0p46b
```

The exporter fails on missing or unexpected model keys and packages the
motion transformer, learned condition parameters, normalization statistics,
rest offsets, text encoders, configuration, and inference demos into one
`Pipeline.from_pretrained` artifact.

Verify exact tensor identity before publishing:

```bash
python tools/audit_motioncanvas_checkpoint.py \
  --source outputs/training/motioncanvas_0p46b/checkpoint-epoch_2354 \
  --artifact outputs/releases/motioncanvas_0p46b \
  --stats-dir /path/to/motioncanvas_198_stats \
  --bone-offsets /path/to/bone_offsets_22.pt
```
