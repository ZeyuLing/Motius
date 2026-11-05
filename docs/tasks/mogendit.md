# MoGenDiT / MoreDiff motion repair

The recovered MoGenDiT snapshot is integrated as `MoGenDiTBundle` and
`MoGenDiTPipeline`. The bundle runs **MoreDiff** (RoPE network) in the original
0.03B, 0.1B or 0.3B configuration. Legacy `MoGenDiT_V2` and `MoGenDiT_CatMask`
network classes are also retained, but their weights must not be loaded into a
MoreDiff bundle.

## Load an original checkpoint directory

```python
import torch
from motius.pipelines.mogendit import MoGenDiTPipeline

pipeline = MoGenDiTPipeline.from_pretrained(
    "/path/to/MoreDiff-0.1B",
    bundle_kwargs={"use_ema": True},
).to("cuda")

# motion201: torch.Tensor shaped (T, 201) or (B, T, 201).
result = pipeline.infer_motion_repair(
    motion201,
    mode="ada_denoise",
    step=10,
    imputation_mode="skip_last",
    restore_world=True,
)
repaired = result["motion"]
```

The directory needs `train_args.json` with `model_version` (`"0.1B"`, etc.) and
`global_pose` (boolean), and an original `ema_model_<step>.pth` checkpoint.
The largest numeric step is selected. Set `use_ema=False` to select
`model_<step>.pth`, or pass `checkpoint="ema_model_0000060000.pth"` in
`bundle_kwargs` to select a specific file. Checkpoints may contain a raw state
dictionary or `{"model": state_dict}`; loading is strict and CPU-mapped.
There is no silent EMA fallback and no bundled/downloaded pretrained weight.

Install the lightweight runtime extra with `pip install -e '.[mogendit]'`.
Inference uses float32. Optional ground simulation requires
`pip install -e '.[mogendit-physics]'` and `flat_ground_trans=True` with windowing.
The recovered simulator uses a fixed 30 fps assumption; leave it disabled for
other frame rates. This optional solver path has not been numerically validated.

## Representation contract

Input must use the archive's **Occam-201**, Y-up coordinates and metres:

| Channels | Meaning |
| --- | --- |
| 0:132 | 22 joint rotations, six values each, in the archive's 6D convention |
| 132:198 | 22 pelvis-relative joint positions, three values each |
| 198:201 | Pelvis translation |

`global_pose` must match the checkpoint. Use `pipeline.bundle.motion_rep.encode`
to convert **local rotation matrices** `(T,22,3,3)`, pelvis-relative joints
`(T,22,3)`, and translation `(T,3)`; it performs global FK when configured.
Do not directly supply HML263, HyMotion198, or another implementation's flattened
rotation-6D ordering. No implicit frame-rate or axis conversion is performed.
The optional skeleton condition must have shape `(B,1,66)`; omitted conditions
use the original network's zero-condition behavior.

Output is a dictionary containing `motion`, local rotation matrices `poses`,
pelvis-relative `joints`, and `transl`, preserving the input's batch convention.
By default, the original normalization resets heading, horizontal origin and
ground height. `restore_world=True` aligns the output's first pelvis and heading
to the source with the original stitching routine.

Modes: `denoise`, `ada_denoise` (two passes with the original 0.1 residual
threshold), and `trans_regen`. For denoising, `step=10` means noising to index 10
then reversing indices 9 through 0; it does **not** mean ten evenly spaced steps
across all 1000 timesteps. Translation regeneration uses the original custom
10-index schedule, or all 1000 indices with `fast_sampling=False`.

Windowing defaults to 224 frames with one preceding frame of overlap.
`imputation_mode="all"` also restores observed channels after the final sampling
step; `skip_last` preserves the upstream behavior allowing final-step changes.
The upstream window stitch can additionally rotate/translate a sampled window.

## Save a self-describing Motius artifact

```python
from motius import Pipeline

pipeline.bundle.save_pretrained("/path/to/motius-mogendit")
restored = Pipeline.from_pretrained("/path/to/motius-mogendit")
```

This writes `model.pt`, `mogendit_config.json`, and `model_index.json`.
A minimal configuration example is in `configs/mogendit/mogendit_config.json`.
Local artifacts are supported; this integration does not publish a Hub model.

## Validation and scope

CPU tests cover representation round trips, all three modes, both window paths,
mask preservation, strict legacy/EMA loading, artifact round trips, batching,
world-frame restoration and network backward propagation. An additional
comparison against the supplied original source verified bit-exact network
forward, seeded denoising and representation encoding/normalization on small
CPU inputs. No pretrained weights were included, so repair quality and full-size
GPU inference have not been validated. Dedicated training, including masks, corruption, losses, EMA and resumption,
is now integrated; see [MoGenDiT training](mogendit_training.md).

Reproduce the source comparison after extracting the supplied archive:

```bash
python tools/verify_mogendit_parity.py --source /path/to/extracted/MoGenDiT
python -m pytest tests/test_mogendit.py tests/test_auto_pipeline.py -q
```

The 29 targeted tests passed in the integration environment. Broader core
registration and task-API audit runs were blocked by missing existing
`transformers` and `clip` dependencies, respectively.
