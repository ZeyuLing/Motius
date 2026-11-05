# Native prepared-motion training

MDM, CondMDI, OmniControl, ProjFlow, MotionCLR, MLD and MotionLCM share the existing `AccelerateRunner` through
`configs/motion_diffusion/prepared_motion.py`. Their Bundles own neural modules;
Trainers own the objectives and noise; Pipelines own iterative inference and CFG.

This entry point consumes prepared tensors. It does not reproduce a new raw-data
augmentation, condition-selection, EMA or mixed-precision training protocol.
Original-source objective/inference certificates and separate native distributed
checks are linked from [the architecture report](../architecture/native-methods.md).

## Data contract

MotionCLR accepts either its verified official raw checkpoint layout or a current
Motius artifact. Its mean/std keep the original float64 values.

A JSON list or JSONL manifest contains records such as:

```json
{"path": "walk_001.npz", "text": "A person walks forward."}
```

Paths are relative to the manifest. NPZ files contain:

| Field | Shape | Meaning |
| --- | --- | --- |
| `motion` | `[T,263]`; ProjFlow `[T,22,3]` | Already normalized with the selected checkpoint's own training statistics; CondMDI uses absolute-root features |
| `length` | scalar integer | Number of valid frames; records in one batch have equal padded `T` |
| `obs_mask` | `[T,263]`, bool | Required by CondMDI; observed features, excluding padding |
| `obs_x0` | `[T,263]` | Optional CondMDI observations in the same normalized absolute representation; defaults to `motion` |
| `hint` | `[T,66]` | Required by OmniControl; normalized joint-position hints with the original zero sentinel |

Data preparation owns normalization, padding and the original random observation
policy. The reader validates shapes and preserves frame/feature alignment.

## Run and resume

```bash
export MOTIUS_MOTION_METHOD=mdm  # condmdi, omnicontrol, projflow, motionclr, mld, motionlcm
export MOTIUS_MOTION_ARTIFACT=/path/to/local/artifact
export MOTIUS_MOTION_MANIFEST=/path/to/prepared/train.jsonl
export MOTIUS_MOTION_WORK_DIR=work_dirs/mdm_native
export MOTIUS_MOTION_BATCH_SIZE=8
export MOTIUS_MOTION_STEPS=100000

torchrun --standalone --nproc_per_node=2 tools/train.py \
  configs/motion_diffusion/prepared_motion.py

# Continue from the latest complete shared-Runner checkpoint.
torchrun --standalone --nproc_per_node=2 tools/train.py \
  configs/motion_diffusion/prepared_motion.py --auto-resume
```

The recipe uses FP32, math attention and frozen CLIP, matching the objective
certificates. Set `MOTIUS_MOTION_LR`, `MOTIUS_MOTION_WORKERS`,
`MOTIUS_MOTION_SAVE_EVERY` and `MOTIUS_MOTION_SEED` as needed. These optimizer
settings are an executable prepared-data entry point, not a claim of the complete
original training schedule.

For MLD, set `MOTIUS_MLD_STAGE=vae` or `diffusion` and use a complete MLD artifact.
Each stage has its own optimizer and work directory; automatic resume continues
the same stage. Native loading preserves either the official CLIP or existing
SentenceT5 conditioning specified by that artifact. The reader trims stored
padding to each minibatch's maximum valid length, as in the original collator.

For MotionLCM distillation, set `MOTIUS_MOTION_METHOD=motionlcm`, use a complete
MotionLCM artifact for its VAE/text architecture, and set
`MOTIUS_MOTIONLCM_TEACHER=/path/to/mld_humanml_v1.ckpt` to the matching teacher.
The Trainer initializes fresh students from that teacher before distributed
preparation; the original guidance projection starts at zero. Its objective owns
teacher CFG, the DDIM consistency target and Huber loss. It also owns the optimizer
step so target EMA runs after the student update and before checkpoint saving.
Use `MOTIUS_MOTION_LR=0.0002` for the source learning rate. This prepared-data
recipe uses accumulation one and FP32; it does not supply the original cosine
warmup schedule or raw-corpus augmentation. Checkpoints include the frozen
VAE/text/teacher, online student, EMA target and optimizer; `--auto-resume`
restores these roots before continuing distillation.

`bundle.save_pretrained(directory)` exports the current denoiser, frozen CLIP,
configuration and actual statistics; reload with `motius.Pipeline.from_pretrained`.
For CondMDI, pass `return_absolute=True` for original absolute-root features or
`return_joints=True` for the original world-space SMPL-22 joints.
The default standard-HumanML conversion cannot represent arbitrary initial global
translation/yaw; compare original absolute-root outputs in the same representation.
