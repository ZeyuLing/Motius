# MoGenDiT / MoreDiff training

This is the dedicated training implementation recovered from the supplied
MoGenDiT snapshot. Its original launcher constructs `get_MoreDiff_model(...)`
and `MoGenDitDistributedTrainer`; Motius exposes the equivalent objective as
`motius.trainers.mogendit.MoGenDiTTrainer`, operating on `MoGenDiTBundle`.
It trains the MoreDiff RoPE network, rather than the legacy non-RoPE MoGenDiT
classes also present in the archive.

## Start training

```bash
pip install -e '.[mogendit]'

python tools/train_mogendit.py configs/mogendit/train_morediff_0p1b.json \
  --data-root /path/to/amass_hq2 /path/to/academic_hq2 \
  --work-dir outputs/training/morediff_0p1b
```

For multiple GPUs:

```bash
torchrun --standalone --nproc_per_node=4 tools/train_mogendit.py \
  configs/mogendit/train_morediff_0p1b.json \
  --data-root /path/to/amass_hq2 /path/to/academic_hq2 \
  --work-dir outputs/training/morediff_0p1b
```

`--batch-size` is **per rank**. `--max-steps` is the total number of optimizer
updates, not epochs or additional steps. The supplied config uses the original
0.1B architecture, batch size 128, AdamW learning rate 1e-4, weight decay 1e-4,
gradient clipping at 1, and 224-frame crops. Training is float32. Change
`model.version` to `0.03B` or `0.3B` for other original presets.

The dedicated entrypoint registers only the required model, dataset and trainer;
it does not require the unrelated model dependencies imported by `tools/train.py`.
The trainer inherits Motius `BaseTrainer` and is registered in `TRAINERS`, the
dataset in `DATASETS`, and the bundle declares its trainable/saveable `net`.
Optimizer stepping is outside `train_step`, following the framework contract.
For the original EMA warm-up and exact cursor/RNG resume behavior, use the
entrypoint above; the generic runner's EMA/checkpoint hooks have different
semantics and are not a substitute for these training checkpoints.

## Input data

Each dataset directory is searched recursively for `.pt` records containing:

| Key | Shape | Convention |
| --- | --- | --- |
| `pose` | `(T,22,3,3)` | Local SMPL-22 rotation matrices |
| `joint` | `(T,22,3)` | Pelvis-relative joints, Y-up, metres |
| `trans` | `(T,3)` | Pelvis translation, Y-up, metres |

Store these values as torch tensors. Dataset loading uses `weights_only=True`;
old pickled NumPy/object records need conversion to tensor dictionaries first.
No body-model mesh file is needed for this preprocessed data format. This loader
does not convert raw AMASS axis-angle or SMPL-H/SMPL-X files.

The original dataset behavior is retained: clips shorter than 30 frames are
excluded, the random start leaves at least 60 frames when available, up to 224
frames are encoded to Occam-201, heading/origin/ground height are normalized,
and shorter crops receive zero padding. `global_pose` determines whether encoding
converts local rotations to global rotations; it must match the model artifact.
The default native corruption is defined for 30 fps, so input must be resampled
before loading. Shapes and finite values are checked; malformed clips fail
explicitly rather than disappearing silently from a cached file list.

## Method-specific training logic

- Uniform sampling over the original 1000-step cosine diffusion schedule;
  predict clean x0 with observed-channel imputation during forward noising.
- Original ordered mask allocation: random frames 0.2, random phrases 0.2,
  start/end frames 0.2, translation inpainting 0.1, joints-only 0.1,
  unconditional 0.2. Per-mode batch counts use the original floor/remainder
  rule, so very small batches can allocate all examples to the final mode.
- Corrupt half the examples by default, preserving a randomly chosen initial
  1–10 frames. Keep **clean** motion as the target. The seven native corruption
  families and original identity probability are retained.
- Sum separate valid-element pose, joint-position, root-translation and global
  joint-velocity losses. Velocity is a per-frame difference, with no added fps
  multiplier. Original L1/L2 weights and timestep importance weights are retained.
- Enable skeleton-offset consistency only when the number of previously completed
  steps is strictly greater than `consis_start_step` (default 10000).
- EMA uses decay 0.999 and the original delayed update starting at completed step
  1000. Before this warm-up, the EMA snapshot is still the initial network: use
  the ordinary `model/` artifact for early smoke-test inference.

The original `loss-second-moment` sampler was instantiated but never updated by
the trainer. The migrated trainer explicitly supports the effective uniform
policy; it does not claim to reproduce adaptive loss-aware sampling.

Corrections made during migration:

- Remove wall-clock reseeding from mask/corruption example selection. Python,
  NumPy and torch RNG now follow the configured per-rank seed and checkpoint.
- Place end-frame observations at each example's real sequence end, not its
  padded end; never mark padding as observed.
- Apply native corruption only to each example's valid prefix; zero-padded
  rotation vectors must not become synthetic skeletons. This changes corruption
  batching relative to upstream but preserves clean targets and native operators.
- Give empty velocity selections a zero loss. Avoid calculating unused
  consistency before its activation threshold.
- Replace the asynchronous original producer thread and hardcoded paths with
  synchronous, configurable loading. This favors exact resumption; background
  data workers/prefetch are not implemented in this entrypoint.

## Resume, fine-tune and use the result

Full resumption:

```bash
python tools/train_mogendit.py configs/mogendit/train_morediff_0p1b.json \
  --data-root /path/to/amass_hq2 /path/to/academic_hq2 \
  --work-dir outputs/training/morediff_0p1b \
  --resume outputs/training/morediff_0p1b/latest.pt
```

Use the same `torchrun` world size when resuming a distributed run. Model,
training settings, data paths/file metadata and world size are checked. You may
change only `max_steps`, `save_interval`, and `log_interval`; `max_steps` must
exceed the saved step. `latest.pt` is written atomically and includes model,
optimizer, EMA, step, epoch, sampler cursor and each rank's Python/NumPy/torch/
CUDA RNG states. Data order is generated separately from crop/noise RNG.
Changing file contents without changing size/mtime is not detected by the data
fingerprint. Cross-hardware bit-exactness is not guaranteed.

For weights-only initialization, use `--init-weights /path/to/model.pth` with a
fresh output directory. Both original `{"model": state_dict}` checkpoints
(including EMA files) and raw state dictionaries are accepted. This starts fresh
optimizer/EMA/step state; it is not a full resume of the old standalone trainer.

On completion, `model/` and `ema/` contain self-describing inference artifacts:

```python
from motius import Pipeline

pipeline = Pipeline.from_pretrained("outputs/training/morediff_0p1b/model").to("cuda")
result = pipeline.infer_motion_repair(motion201)
```

## Verification

```bash
python -m pytest tests/test_mogendit_training.py tests/test_mogendit.py tests/test_auto_pipeline.py -q
python tools/verify_mogendit_training_parity.py --source /path/to/extracted/MoGenDiT
```

Tests exercise real optimization, tiny-set loss reduction, all seven native
corruption families, padding/observation semantics, consistency activation,
legacy weight loading, and exact single-process interrupted/resumed training.
The parity tool extracts the loss methods from the supplied original trainer:
tested totals are bit-exact; prediction gradients differ by at most 1.9e-9 in
the tested active-consistency case. Small CPU/Gloo two-rank training and full
resume also match uninterrupted model, optimizer and EMA states bit-for-bit.
A single-CUDA-device training smoke test and loading its exported weights for
repair inference also passed. Multi-GPU NCCL training has not been exercised
because this environment exposes one CUDA device.
These are implementation checks on synthetic clips, not a trained-model quality
or benchmark-reproduction claim.
