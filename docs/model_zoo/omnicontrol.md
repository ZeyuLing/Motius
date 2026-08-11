<h1 align="center">OmniControl Model Card</h1>

<p align="center">
  <strong>Text-to-motion generation with sparse, per-axis control over any SMPL body joint.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2310.08580">Paper</a> ·
  <a href="https://neu-vi.github.io/omnicontrol/">Project Page</a> ·
  <a href="https://github.com/neu-vi/OmniControl">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-omnicontrol-humanml3d">Motius Checkpoint</a>
</p>

OmniControl generates HumanML3D motion from text while controlling selected
3D joint positions at selected frames. The Motius integration vendors the
official MIT inference runtime and uses the released HumanML3D checkpoint.

**Tasks:** Text-to-Motion, Temporal Motion Completion, Kinematic Motion Control.

[Kinematic Motion Control benchmark protocol](../leaderboards/kinematic_motion_control.md)

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
| Text-to-Motion | Native task input | <video src="https://github.com/user-attachments/assets/41dc67e4-aca8-4d33-9fb5-0203690e4c14" controls></video> | [MP4](https://github.com/user-attachments/assets/41dc67e4-aca8-4d33-9fb5-0203690e4c14) |
| Temporal Motion Completion | person walking at a average pace forward, swaying arms and torso with a sense of swagger<br><sub>Adaptive sparse keyframes; 12 observed frames over a 300-frame clip</sub> | <video src="https://github.com/user-attachments/assets/4e6e1699-1a23-41d6-8d25-dfca67851af3" controls></video> | [MP4](https://github.com/user-attachments/assets/4e6e1699-1a23-41d6-8d25-dfca67851af3) · [All cases](https://zeyuling-temporal-condition-leaderboard.static.hf.space/cases/adaptive_keyframes/index.html?method=omnicontrol&case=004822) |
| Kinematic Motion Control | Text plus spatial motion constraints | <video src="https://github.com/user-attachments/assets/288899b9-e744-4c91-9192-d74cffe56c8d" controls></video> | [MP4](https://github.com/user-attachments/assets/288899b9-e744-4c91-9192-d74cffe56c8d) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


- [Temporal completion all-case viewer](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard)
- [Kinematic control protocol and examples](../leaderboards/kinematic_motion_control.md)

Both routes retain the supplied world-space evidence and use the shared SMPL
Mesh viewer for qualitative comparison. The constrained frames and joints are
shown separately from generated regions.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |
| Temporal Motion Completion | `infer_temporal_motion_completion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) |
| Kinematic Motion Control | `infer_kinematic_motion_control` | [Benchmark and examples](../leaderboards/kinematic_motion_control.md) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 20 fps (HumanML3D) |
| Public preview | 30 fps, duration-preserving 20→30 fps resampling |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->

| Item | Value |
| --- | --- |
| Tasks | Text-to-Motion, Temporal Motion Completion, Kinematic Motion Control |
| Native representation | HumanML3D-263, 20 FPS |
| Control evidence | World-space XYZ for selected joints, axes, and frames |
| Checkpoint | [`ZeyuLing/motius-omnicontrol-humanml3d`](https://huggingface.co/ZeyuLing/motius-omnicontrol-humanml3d) |
| Pipeline | `motius.pipelines.omnicontrol.OmniControlPipeline` |
| License | MIT |

- Input motion representation: physical-scale HumanML3D-263.
- Control evidence: world-space XYZ positions for any subset of the 22 joints
  at any subset of frames.
- Temporal completion: select all joints at the required prefix, boundary, or
  keyframes.
- Root trajectory: select the pelvis across dense or sparse frames.
- Local joint rotations are not a native OmniControl control input.

### Checkpoint

- [Motius OmniControl · HumanML3D](https://huggingface.co/ZeyuLing/motius-omnicontrol-humanml3d)

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-omnicontrol-humanml3d",
    bundle_kwargs={"device": "cuda"},
)

generated = pipe.infer_text_to_motion(
    ["a person walks forward"],
    [120],
)

outputs = pipe.infer_temporal_motion_completion(
    captions=["a person walks forward"],
    motions=[ground_truth_hml263],
    control_mode="first_last",
)

controlled = pipe.infer_kinematic_motion_control(
    captions=["a person follows the supplied wrist path"],
    motions=[control_reference_hml263],
    joint_indices=[20, 21],
    axes="xyz",
    control_mode="trajectory",
)
```

The artifact contains the released model, OpenAI CLIP ViT-B/32, and both
motion and spatial normalization statistics. After the snapshot is available,
inference requires no OmniControl checkout or secondary model download.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Temporal Motion Completion | [Published results](../leaderboards/hf_space_temporal_condition/temporal_control_results.json) | HumanML3D temporal settings; normalized uTMR FID |
| Kinematic Motion Control | [Published results](../leaderboards/kinematic_motion_control.md) | Native-skeleton protocol; rows remain pending |

### Canonical Temporal-Completion Snapshot

#### Temporal Control · Motius normalized space

| Setting | n | R@1 | R@2 | R@3 | Motius FID (normalized) | MM-Dist | Diversity | Constraint error (cm) | Foot skating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| temporal_start_1f | 4,012 | 0.4345 | 0.5785 | 0.6548 | 0.1626 | 38.5584 | 57.6540 | 12.5645 | 0.0912 |
| temporal_pre20 | 4,012 | 0.3915 | 0.5363 | 0.6122 | 0.1809 | 40.0146 | 55.6316 | 5.4204 | 0.1381 |
| temporal_pre20_uncond | 4,012 | 0.2040 | 0.3127 | 0.3887 | 0.4044 | 48.6065 | 50.2717 | 6.6972 | 0.0730 |
| temporal_both_1f | 4,012 | 0.3820 | 0.5292 | 0.6122 | 0.1913 | 39.9154 | 56.3401 | 9.5008 | 0.2494 |
| temporal_mid80 | 4,012 | 0.4020 | 0.5410 | 0.6270 | 0.1591 | 38.7989 | 56.1377 | 7.9063 | 0.3430 |
| temporal_mid80_uncond | 4,012 | 0.1898 | 0.2878 | 0.3610 | 0.2627 | 47.2423 | 54.0509 | 8.9497 | 0.3828 |
| temporal_adaptive_keyframes | 4,012 | 0.4265 | 0.5795 | 0.6630 | 0.1818 | 38.3565 | 56.2984 | 11.1176 | 0.4764 |
| temporal_adaptive_keyframes_uncond | 4,012 | 0.2427 | 0.3485 | 0.4295 | 0.2969 | 45.7091 | 54.5594 | 13.2468 | 0.5418 |

### Canonical HumanML3D Semantic Results

| Evaluator | Variant | n | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| HumanML3D Official | Default | — | — | — | — | — | — | — | Not measured |
| MotionStreamer Evaluator | Default | — | — | — | — | — | — | — | Not measured |
| Motius Joint-Position Evaluator | Default | — | — | — | — | — | — | — | Not measured |

<!-- MOTIUS_CANONICAL_METRICS:END -->


### Temporal Motion Completion · HumanML3D

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
[Temporal Motion Completion · HumanML3D benchmark](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard).

### Part-Level Motion Control · HumanML3D

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

## Motion Representation

OmniControl consumes denormalized HumanML3D-263 together with spatial evidence
expressed as world-space positions over the shared 22-joint SMPL body skeleton.
Its internal spatial normalizer is packaged with the checkpoint. The model does
not accept local joint rotations as native control input; SMPL Mesh output is
materialized after HumanML3D recovery for evaluation and visualization.

## Citation and License

- Paper: [OmniControl: Control Any Joint at Any Time for Human Motion Generation](https://arxiv.org/abs/2310.08580)
- Official code: [neu-vi/OmniControl](https://github.com/neu-vi/OmniControl)
- Vendored license: `motius/models/omnicontrol/LICENSE`

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
