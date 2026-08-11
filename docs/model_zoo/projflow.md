<h1 align="center">ProjFlow Model Card</h1>

<p align="center">
  <strong>Training-free projection sampling for exact Cartesian motion constraints.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2602.22742">Paper</a> ·
  <a href="https://akihisa-watanabe.github.io/projflow.github.io/">Project Page</a> ·
  <a href="https://github.com/Akihisa-Watanabe/ProjFlow">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-projflow-humanml3d">Motius Checkpoint</a>
</p>

ProjFlow is the CVPR 2026 method *ProjFlow: Projection Sampling with Flow
Matching for Zero-Shot Exact Spatial Motion Control*. Motius includes a native
ACMDM Flow implementation and projection sampler; inference does not import or
clone the upstream repository.

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
| Temporal Motion Completion | person walking at an average pace forward, swaying arms and torso with a sense of swagger<br><sub>First and last full-body frames observed</sub> | <video src="https://github.com/user-attachments/assets/9f754147-2da3-4cc1-b8e4-3b091ba5c4a4" controls></video> | [MP4](https://github.com/user-attachments/assets/9f754147-2da3-4cc1-b8e4-3b091ba5c4a4) |
| Kinematic Motion Control | person walking at an average pace forward, swaying arms and torso with a sense of swagger<br><sub>Sparse pelvis XYZ trajectory every 20 frames</sub> | <video src="https://github.com/user-attachments/assets/e23e01a8-d239-4860-89b0-8b4b5e6605fe" controls></video> | [MP4](https://github.com/user-attachments/assets/e23e01a8-d239-4860-89b0-8b4b5e6605fe) |
| Part-Level Motion Control | person walking at an average pace forward, swaying arms and torso with a sense of swagger<br><sub>Left-wrist XYZ position observed at every frame</sub> | <video src="https://github.com/user-attachments/assets/d8b22e4f-05ff-481b-b410-4eae70e3884b" controls></video> | [MP4](https://github.com/user-attachments/assets/d8b22e4f-05ff-481b-b410-4eae70e3884b) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Temporal Motion Completion | `infer_temporal_motion_completion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) |
| Kinematic Motion Control | `infer_kinematic_motion_control` | [Benchmark and examples](../leaderboards/kinematic_motion_control.md) |
| Part-Level Motion Control | `infer_part_level_motion_control` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 20 fps (ACMDM Flow prior on HumanML3D) |
| Public preview | 20 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->

| Item | Value |
| --- | --- |
| Tasks | Temporal Motion Completion, Kinematic Motion Control, Part-Level Motion Control |
| Native representation | HumanML3D SMPL-22 joint positions, 20 fps |
| Motion prior | ACMDM Raw Flow S PatchSize22 |
| Sampling | ProjFlow, 100 steps, CFG 2.5 |
| Maximum length | 196 frames |
| Checkpoint | [`ZeyuLing/motius-projflow-humanml3d`](https://huggingface.co/ZeyuLing/motius-projflow-humanml3d) |
| Pipeline | `motius.pipelines.projflow.ProjFlowPipeline` |

The artifact contains the ACMDM model, OpenAI CLIP ViT-B/32, joint
normalization statistics, configuration, and provenance. Forward and sampler
parity against upstream revision `9550501a439964a73063505b7a52e574ae11a43c`
are bit-exact on the audited test inputs.

## Quick Start

```python
import numpy as np
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-projflow-humanml3d",
    bundle_kwargs={"device": "cuda"},
)

# True means generate; False means preserve the observed frame.
generation_mask = np.ones(len(source_joints), dtype=bool)
generation_mask[[0, -1]] = False
completed = pipe.infer_temporal_motion_completion(
    source_joints,
    generation_mask=generation_mask,
    captions=["a person walks forward"],
)

trajectory = pipe.infer_kinematic_motion_control(
    ["a person follows the supplied path"],
    source_joints,
    control_mode="trajectory",
    joint_indices=[0],
    axes="xyz",
)

left_wrist = pipe.infer_part_level_motion_control(
    ["a person walks while following the wrist path"],
    source_joints,
    control_mode="dense",
    joint_indices=[20],
    axes="xyz",
)
```

Inputs may be `(T, 22, 3)` joint positions or `(T, 263)` HumanML3D features.
The default output is a list of metric-space `(T, 22, 3)` arrays. Pass
`return_format="hml263"` when a downstream HumanML3D feature tensor is needed.
Lengths above 196 raise an error rather than being silently truncated.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Temporal Motion Completion | [Published results](../leaderboards/hf_space_temporal_condition/temporal_control_results.json) | HumanML3D temporal settings; normalized uTMR FID |
| Kinematic Motion Control | [Published results](../leaderboards/kinematic_motion_control.md) | Native-skeleton protocol; rows remain pending |
| Part-Level Motion Control | [Published results](https://huggingface.co/spaces/ZeyuLing/body-part-condition-humanml3d-leaderboard) | HumanML3D part-control metrics; normalized uTMR FID |

### Canonical Temporal-Completion Snapshot

#### Temporal Control · Motius normalized space

| Setting | n | R@1 | R@2 | R@3 | Motius FID (normalized) | MM-Dist | Diversity | Constraint error (cm) | Foot skating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| temporal_start_1f | 4,012 | 0.5857 | 0.7493 | 0.8227 | 0.0239 | 30.9837 | 54.5344 | 1.3281 | 0.1228 |
| temporal_pre20 | 4,012 | 0.6308 | 0.7910 | 0.8600 | 0.0125 | 29.4352 | 53.9720 | 2.0006 | 0.1116 |
| temporal_pre20_uncond | 4,012 | 0.2795 | 0.4078 | 0.4861 | 0.2199 | 43.1426 | 49.9231 | 1.9978 | 0.0581 |
| temporal_both_1f | 4,012 | 0.6112 | 0.7725 | 0.8448 | 0.0191 | 30.1218 | 54.1275 | 8.0387 | 0.1380 |
| temporal_mid80 | 4,012 | 0.6518 | 0.8095 | 0.8752 | 0.0093 | 28.8443 | 53.7430 | 7.5732 | 0.1233 |
| temporal_mid80_uncond | 4,012 | 0.3595 | 0.4942 | 0.5756 | 0.1763 | 39.8629 | 51.9529 | 7.0210 | 0.0803 |
| temporal_adaptive_keyframes | 4,012 | 0.6787 | 0.8377 | 0.8993 | 0.0075 | 28.4312 | 53.9327 | 7.0198 | 0.1685 |
| temporal_adaptive_keyframes_uncond | 4,012 | 0.5916 | 0.7652 | 0.8428 | 0.0280 | 31.0818 | 52.9138 | 7.2312 | 0.1650 |

<!-- MOTIUS_CANONICAL_METRICS:END -->

All eight Temporal Motion Completion settings use the 4,012-case HumanML3D
official test split, selected captions, retrieval batches of 32, and one
deterministic evaluation pass. Motius FID is computed in per-sample
L2-normalized joint-position evaluator space. ProjFlow is not reported on
rotation-only Body Part settings because its native constraint is Cartesian
position, not local rotation.

The three release demos additionally verify the hard constraints directly;
the maximum constrained-joint error is below `1.3e-7 m`. This single-case API
check is not presented as a ranked benchmark result.

## Motion Representation

ProjFlow operates on world-space HumanML3D SMPL-22 joint positions in meters,
with Y up and 20 fps. It applies the released per-axis statistics
`mean=[0.00016345, 0.37663162, 0.31019124]` and
`std=[0.5147618, 0.6101322, 0.8715584]`. A Boolean `(B,T,22,3)` mask selects
the exact frames, joints, and axes to project during sampling.

The native joint output is preferred for control analysis. HML263 export
re-canonicalizes the generated joints and derives rotations, velocities, and
contacts, so it is a lossy convenience representation rather than ProjFlow's
native state.

## Citation and License

```bibtex
@inproceedings{watanabe2026projflow,
  title={ProjFlow: Projection Sampling with Flow Matching for Zero-Shot Exact Spatial Motion Control},
  author={Akihisa Watanabe and Qing Yu and Edgar Simo-Serra and Kent Fujiwara},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2026}
}
```

The original [ProjFlow repository](https://github.com/Akihisa-Watanabe/ProjFlow)
did not contain a repository-level license file at the audited revision. Review
the upstream terms before redistribution or commercial use. Vendored-source
provenance and file-level notices are recorded in
[`motius/models/projflow/ATTRIBUTIONS.md`](../../motius/models/projflow/ATTRIBUTIONS.md).
OpenAI CLIP retains its own license and attribution.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
