# OmniControl Model Card

OmniControl generates HumanML3D motion from text while controlling selected
3D joint positions at selected frames. The Motius integration vendors the
official MIT inference runtime and uses the released HumanML3D checkpoint.

## Native Control Contract

- Input motion representation: physical-scale HumanML3D-263.
- Control evidence: world-space XYZ positions for any subset of the 22 joints
  at any subset of frames.
- Temporal completion: select all joints at the required prefix, boundary, or
  keyframes.
- Root trajectory: select the pelvis across dense or sparse frames.
- Local joint rotations are not a native OmniControl control input.

## Usage

```python
from motius.pipelines.omnicontrol import OmniControlPipeline

pipe = OmniControlPipeline.from_pretrained(
    "/path/to/model_humanml3d.pt",
    device="cuda",
)
outputs = pipe.infer_control(
    captions=["a person walks forward"],
    motions=[ground_truth_hml263],
    control_mode="first_last",
)
```

The runtime caches the CLIP text embedding once per batch. Motion and spatial
normalization statistics are packaged with Motius so execution does not depend
on an external OmniControl checkout or a current working directory.

## Temporal Condition Benchmark

All eight settings use the complete 4,012-sample HumanML3D official test split.
Retrieval uses batches of 32 and a single deterministic repeat. FID is measured
in the normalized Motius joint-position evaluator space; condition error is
pelvis-relative SMPL-22 error on constrained frames.

| Condition | Text | R@1 | R@2 | R@3 | FID | MM-Dist | Error (cm) | Fail@20 | Fail@50 | Foot skate | Diversity |
| --------- | :--: | --: | --: | --: | --: | ------: | ---------: | ------: | ------: | ---------: | --------: |
| First frame | On | 0.4345 | 0.5785 | 0.6548 | 0.1626 | 38.5584 | 12.5645 | 0.1563 | 0.0057 | 0.0912 | 57.6540 |
| First 20% | On | 0.3915 | 0.5363 | 0.6122 | 0.1809 | 40.0146 | 5.4204 | 0.0285 | 0.0030 | 0.1381 | 55.6316 |
| First 20% | Off | 0.2040 | 0.3127 | 0.3887 | 0.4044 | 48.6065 | 6.6972 | 0.0695 | 0.0036 | 0.0730 | 50.2717 |
| First + last frame | On | 0.3820 | 0.5292 | 0.6122 | 0.1913 | 39.9154 | 9.5008 | 0.0955 | 0.0032 | 0.2494 | 56.3401 |
| First + last 10% | On | 0.4020 | 0.5410 | 0.6270 | 0.1591 | 38.7989 | 7.9063 | 0.0697 | 0.0025 | 0.3430 | 56.1377 |
| First + last 10% | Off | 0.1898 | 0.2878 | 0.3610 | 0.2627 | 47.2423 | 8.9497 | 0.1023 | 0.0041 | 0.3828 | 54.0509 |
| Adaptive sparse frames | On | 0.4265 | 0.5795 | 0.6630 | 0.1818 | 38.3565 | 11.1176 | 0.1217 | 0.0042 | 0.4764 | 56.2984 |
| Adaptive sparse frames | Off | 0.2427 | 0.3485 | 0.4295 | 0.2969 | 45.7091 | 13.2468 | 0.1913 | 0.0145 | 0.5418 | 54.5594 |

The canonical result records are maintained in the
[Temporal Condition leaderboard](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard).

## Body-Part Position Benchmark

All settings use the complete 4,012-sample HumanML3D official test split and
OmniControl's native world-space, per-axis joint evidence. Retrieval uses 125
complete batches of 32; FID is computed over all samples in the normalized
Motius joint-position evaluator space. Control error is pelvis-relative
SMPL-22 error on constrained channels.

| Condition | Evidence | R@1 | R@2 | R@3 | FID | MM-Dist | Error (cm) | Hit@5cm | Hit@10cm | Foot skate | Diversity |
| --------- | -------- | --: | --: | --: | --: | ------: | ---------: | ------: | -------: | ---------: | --------: |
| Upper body, dense | XYZ | 0.5293 | 0.6897 | 0.7689 | 0.0865 | 33.3632 | 12.69 | 0.3640 | 0.6099 | 0.5857 | 55.0567 |
| Lower body, dense | XYZ | 0.5716 | 0.7327 | 0.8058 | 0.0438 | 31.9896 | 8.29 | 0.5734 | 0.7667 | 0.6682 | 55.0192 |
| Both wrists, sparse | XYZ | 0.6082 | 0.7709 | 0.8402 | 0.0339 | 30.7173 | 20.62 | 0.1144 | 0.3315 | 0.7039 | 54.9739 |
| Both wrists, dense | XYZ | 0.5935 | 0.7592 | 0.8305 | 0.0427 | 31.1537 | 19.52 | 0.1562 | 0.3775 | 0.7472 | 54.7597 |
| Both elbows, sparse | XYZ | 0.5194 | 0.6839 | 0.7666 | 0.0649 | 34.0466 | 18.33 | 0.1339 | 0.3737 | 0.4820 | 55.7278 |
| Both elbows, dense | XYZ | 0.4660 | 0.6264 | 0.7103 | 0.1181 | 37.0355 | 19.70 | 0.1289 | 0.3467 | 0.4430 | 55.7839 |
| Both feet, sparse | XZ | 0.5499 | 0.7117 | 0.7875 | 0.0523 | 32.7509 | 15.60 | 0.2310 | 0.4807 | 0.9572 | 55.4945 |
| Both feet, dense | XZ | 0.4758 | 0.6275 | 0.7038 | 0.1369 | 36.5298 | 17.39 | 0.2033 | 0.4376 | 0.8570 | 55.7148 |
| Both knees, sparse | XYZ | 0.4950 | 0.6554 | 0.7379 | 0.0811 | 35.3168 | 13.78 | 0.2755 | 0.5218 | 0.5815 | 56.1932 |
| Both knees, dense | XYZ | 0.4367 | 0.5907 | 0.6756 | 0.1446 | 38.3829 | 14.92 | 0.2530 | 0.5008 | 0.4474 | 56.0380 |

Lower is better for FID, MM-Dist, control error, and foot skate; higher is
better for retrieval and hit rate, while diversity is interpreted relative to
GT.

## Provenance

- Paper: [OmniControl: Control Any Joint at Any Time for Human Motion Generation](https://arxiv.org/abs/2310.08580)
- Official code: [neu-vi/OmniControl](https://github.com/neu-vi/OmniControl)
- Vendored license: `motius/models/omnicontrol/LICENSE`
