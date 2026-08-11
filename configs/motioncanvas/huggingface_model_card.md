---
library_name: motius
tags:
  - motion-generation
  - motion-completion
  - motion-control
  - motion-editing
  - smpl
---

# Motius MotionCanvas 0.46B

MotionCanvas is the Motius-native HYMotion M2M checkpoint for temporal motion
completion, kinematic control, and text-guided motion editing. This epoch 2354
artifact is self-contained: it includes the motion transformer, learned M2M
condition parameters, MotionCanvas-198 normalization statistics, SMPL-22 rest
offsets, Qwen3, and CLIP.

## Load

```bash
python -m pip install git+https://github.com/ZeyuLing/Motius.git
python -m pip install torchdiffeq einops mmcv-lite
```

```python
import torch
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MotionCanvas-0.46B",
    cache_dir="/path/to/large/huggingface-cache",
    bundle_kwargs={"device": "cuda", "text_dtype": "bf16"},
    num_steps=50,
)

source = torch.from_numpy(reference_motion_198)[None].cuda()
generation_mask = torch.zeros(1, source.shape[1], device="cuda")
generation_mask[:, 90:180] = 1

result = pipe.infer_temporal_motion_completion(
    source,
    generation_mask,
    captions="a person turns left and continues walking",
    seed=42,
)
motion_198 = result["motion_198"]
```

The complete artifact is about 20 GB. `cache_dir` must be a top-level loader
argument; set `HF_XET_CACHE` too when the default Xet cache is on a small
system disk.

`generation_mask` uses `1=generate` and `0=preserve`. It can have shape
`(B,T)` or `(B,T,198)`. The same artifact exposes:

- `infer_temporal_motion_completion`
- `infer_motion_inbetweening`
- `infer_keyframe_motion_control`
- `infer_kinematic_motion_control`
- `infer_motion_editing`
- `infer_motion_repair`
- `infer_m2m`

## Demos

| Temporal completion | Kinematic control |
| --- | --- |
| ![Temporal completion](demos/keyframe_control_512_30fps.gif) | ![Kinematic control](demos/trajectory_control_512_30fps.gif) |

| Motion editing | Trajectory editing |
| --- | --- |
| ![Motion editing](demos/instruction_editing_512_30fps.gif) | ![Trajectory editing](demos/motion_editing_512_30fps.gif) |

The `demos/` directory also contains the original 1080p MP4 renders.

## Representation

MotionCanvas-198 runs at 30 fps on a 360-frame canvas:

- 3 world-space root translation values
- 132 row-major local rotation-6D values for 22 SMPL joints
- 63 pelvis-relative FK joint-position values for 21 non-root joints

`Mean.npy` and `Std.npy` are the exact HY-Motion-201 statistics with the unused
pelvis-RIC triplet removed. See `normalization_provenance.md`.

## Evaluation

All rows use the 4,012-motion HumanML3D official test split. FID is computed
in per-sample L2-normalized uTMR embedding space.

| Setting | Text | R@1 | R@2 | R@3 | FID | MM-Dist | Condition cm | Foot skating |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First frame | yes | 0.6914 | 0.8410 | 0.8988 | 0.0139 | 27.9444 | 0.0000 | 0.1380 |
| Prefix 20% | yes | 0.6931 | 0.8406 | 0.8993 | 0.0091 | 27.8049 | 0.0000 | 0.0840 |
| Prefix 20% | no | 0.3411 | 0.4731 | 0.5532 | 0.1043 | 40.0292 | 0.0000 | 0.0960 |
| First + last frame | yes | 0.6883 | 0.8380 | 0.8972 | 0.0135 | 28.0997 | 0.0000 | 0.1840 |
| First + last 10% | yes | 0.6941 | 0.8452 | 0.9030 | 0.0052 | 27.7324 | 0.0000 | 0.1100 |
| First + last 10% | no | 0.4283 | 0.5758 | 0.6562 | 0.0837 | 36.9094 | 0.0000 | 0.1250 |
| Adaptive keyframes | yes | 0.6875 | 0.8441 | 0.9031 | 0.0014 | 27.8142 | 0.0000 | 0.1184 |
| Adaptive keyframes | no | 0.6589 | 0.8218 | 0.8878 | 0.0031 | 28.6949 | 0.0000 | 0.1358 |

The complete protocol and physical metrics are maintained in the
[MotionCanvas Model Card](https://github.com/ZeyuLing/Motius/blob/main/docs/model_zoo/motioncanvas.md)
and the
[Temporal Condition Leaderboard](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard).

## Provenance

MotionCanvas is implemented and trained in
[Motius](https://github.com/ZeyuLing/Motius). Its transformer and
text-conditioning design build on
[HY-Motion 1.0](https://arxiv.org/abs/2512.23464). The M2M objective,
arbitrary-condition data pipeline, clean-imputation solver, sparse rollout,
and this checkpoint are Motius-native.
