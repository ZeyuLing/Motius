<h1 align="center">KIMODO Model Card</h1>

<p align="center">
  <strong>Controllable human and humanoid motion generation through text and kinematic constraints.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2603.15546">Paper</a> ·
  <a href="https://research.nvidia.com/labs/sil/projects/kimodo/">Project Page</a> ·
  <a href="https://github.com/nv-tlabs/kimodo">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-KIMODO-SOMA-RP">SOMA-RP Checkpoint</a>
</p>

KIMODO is NVIDIA's kinematic motion diffusion model for text-driven and
constraint-driven motion authoring. This Motius release packages the native
KIMODO runtime, skeleton assets, motion-representation utilities, and a unified
pipeline facade for text-to-motion, multi-prompt transitions, full-body
keyframes, end-effector controls, root paths, and prefix-conditioned TP2M.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/078ea591-3110-4ebe-88b1-fe102a1e7e0e" controls></video> | [MP4](https://github.com/user-attachments/assets/078ea591-3110-4ebe-88b1-fe102a1e7e0e) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=kimodo) |
| Temporal Motion Completion | person walking at a average pace forward, swaying arms and torso with a sense of swagger<br><sub>Prediction: first 20%; condition frames 0-59 of 300</sub> | <video src="https://github.com/user-attachments/assets/ad327700-9523-4c1c-b0a8-75eb5c14bd19" controls></video> | [MP4](https://github.com/user-attachments/assets/ad327700-9523-4c1c-b0a8-75eb5c14bd19) · [All cases](https://zeyuling-temporal-condition-leaderboard.static.hf.space/cases/pre20/index.html?method=kimodo&case=004822) |
| Sequential Text-to-Motion | “A person walks forward, then lifts.” → “A person walks back.” | <video src="https://github.com/user-attachments/assets/1ad2661c-58d0-4a39-8c8d-ceaa35c7fd7d" controls></video> | [MP4](https://github.com/user-attachments/assets/1ad2661c-58d0-4a39-8c8d-ceaa35c7fd7d) |
| Kinematic Motion Control | Text plus spatial motion constraints | <video src="https://github.com/user-attachments/assets/60483c7c-344d-4dab-a812-dd354b86125e" controls></video> | [MP4](https://github.com/user-attachments/assets/60483c7c-344d-4dab-a812-dd354b86125e) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/078ea591-3110-4ebe-88b1-fe102a1e7e0e" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/ec7d7aaa-b747-4946-ba7b-e388c3126f1a" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/9a90d84d-c4fe-473c-90d2-54e48d21b323" controls></video> |

512px / 30fps H.264 video previews rendered from released HumanML3D test outputs.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |
| Temporal Motion Completion | `infer_temporal_motion_completion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) |
| Sequential Text-to-Motion | `infer_sequential_text_to_motion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) |
| Kinematic Motion Control | `infer_kinematic_motion_control` | [Benchmark and examples](../leaderboards/kinematic_motion_control.md) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 30 fps for released SOMA, SMPL-X, and G1 checkpoints |
| Public preview | 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | KIMODO, two-stage kinematic motion diffusion |
| Tasks | Text-to-Motion, Temporal Motion Completion, Sequential Text-to-Motion, Kinematic Motion Control |
| Motion representations | SOMA, Unitree G1, SMPL-X, plus `motion_135` TP2M bridge |
| Text encoder | LLM2Vec / Meta-Llama-3 local encoder tree |
| Default model | `Kimodo-SOMA-RP-v1` |
| Pipeline | `motius.pipelines.kimodo.KIMODOPipeline` |

[Kinematic Motion Control benchmark protocol](../leaderboards/kinematic_motion_control.md)

Processed checkpoints:

| Variant | Native Skeleton | Checkpoint |
| ------- | --------------- | ---------- |
| SOMA-RP | SOMA | [`ZeyuLing/Motius-KIMODO-SOMA-RP`](https://huggingface.co/ZeyuLing/Motius-KIMODO-SOMA-RP) |
| G1-RP | Unitree G1 | [`ZeyuLing/Motius-KIMODO-G1-RP`](https://huggingface.co/ZeyuLing/Motius-KIMODO-G1-RP) |
| G1-SEED | Unitree G1 | [`ZeyuLing/Motius-KIMODO-G1-SEED`](https://huggingface.co/ZeyuLing/Motius-KIMODO-G1-SEED) |
| SMPLX-RP | SMPL-X | [`ZeyuLing/Motius-KIMODO-SMPLX-RP`](https://huggingface.co/ZeyuLing/Motius-KIMODO-SMPLX-RP) |

Each artifact contains its own complete LLM2Vec / Meta-Llama text-encoder tree,
so every variant remains independently loadable after one snapshot download.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.kimodo.KIMODOPipeline` |
| Bundle | `motius.models.kimodo.KIMODOBundle` |
| Runtime | `motius.models.kimodo.network` |

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-KIMODO-SOMA-RP",
    device="cuda",
)

motion = pipe.infer_text_to_motion(
    "a person walks forward and waves.",
    num_frames=150,
)
```

The G1 checkpoints use the identical task API and keep inference in the native
Unitree G1 skeleton space:

```python
g1_pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-KIMODO-G1-RP",
    device="cuda",
)
g1_motion = g1_pipe.infer_text_to_motion(
    "a humanoid walks forward and waves its right hand.",
    num_frames=150,
)
```

Use the same pipeline for constraint-based generation:

```python
root_path = pipe.root2d_constraint(
    frame_indices=[0, 30, 60, 90],
    smooth_root_2d=[[0.0, 0.0], [0.5, 0.2], [1.0, 0.2], [1.5, 0.0]],
)

motion = pipe.infer_kinematic_motion_control(
    "a person follows a curved walking path",
    num_frames=120,
    constraints=[root_path],
)
```

Temporal completion accepts either a native KIMODO result or a `motion_135`
prefix. Native inputs stay in SOMA space; `motion_135` inputs retain the
evaluation bridge:

```python
reference = pipe.infer_text_to_motion(
    "a person keeps walking forward",
    num_frames=120,
)
samples = pipe.infer_temporal_motion_completion(
    ["a person keeps walking forward"],
    [reference],
    condition_frames=30,
)
```

Ordered prompts use the same artifact:

```python
sequence = pipe.infer_sequential_text_to_motion(
    ["walk forward", "turn right", "sit down"],
    [60, 60, 90],
)
```

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Temporal Motion Completion | [Published results](../leaderboards/hf_space_temporal_condition/temporal_control_results.json) | HumanML3D temporal settings; normalized uTMR FID |
| Sequential Text-to-Motion | [Published results](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | BABEL semantic and transition metrics; normalized uTMR FID |
| Kinematic Motion Control | [Published results](../leaderboards/kinematic_motion_control.md) | Native-skeleton protocol; rows remain pending |

### Canonical Temporal-Completion Snapshot

#### Temporal Control · Motius normalized space

| Setting | n | R@1 | R@2 | R@3 | Motius FID (normalized) | MM-Dist | Diversity | Constraint error (cm) | Foot skating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| temporal_start_1f | 4,012 | 0.4246 | 0.5868 | 0.6786 | 0.2964 | 41.5010 | 56.8706 | 0.0000 | 0.1780 |
| temporal_pre20 | 4,012 | 0.4510 | 0.6053 | 0.6891 | 0.1686 | 38.5412 | 56.3010 | 0.0000 | 0.0870 |
| temporal_pre20_uncond | 4,012 | 0.2881 | 0.4050 | 0.4772 | 0.3350 | 46.6152 | 55.0981 | 0.0000 | 0.0630 |
| temporal_both_1f | 4,012 | 0.4715 | 0.6266 | 0.7087 | 0.2137 | 38.8853 | 56.7591 | 0.0000 | 0.1540 |
| temporal_mid80 | 4,012 | 0.5393 | 0.7014 | 0.7772 | 0.0910 | 34.4845 | 55.3450 | 0.0000 | 0.0850 |
| temporal_mid80_uncond | 4,012 | 0.4059 | 0.5482 | 0.6276 | 0.1906 | 40.0989 | 54.2403 | 0.0000 | 0.0740 |
| temporal_adaptive_keyframes | 4,012 | 0.6132 | 0.7822 | 0.8560 | 0.0694 | 31.8709 | 55.6616 | 1.8341 | 0.1851 |
| temporal_adaptive_keyframes_uncond | 4,012 | 0.5914 | 0.7614 | 0.8381 | 0.0616 | 32.3919 | 54.9968 | 1.7745 | 0.1654 |

#### TP2M Prefix · MotionStreamer-272 space

| Setting | n | R@1 | R@2 | R@3 | MotionStreamer FID | MM-Dist | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-frame prefix | 3,968 | 0.5250 | 0.6900 | 0.7690 | 82.5600 | 19.3010 | 26.1580 |
| 5-frame prefix | 3,968 | 0.5380 | 0.6990 | 0.7750 | 80.3810 | 19.1990 | 26.1540 |
| 9-frame prefix | 3,968 | 0.5310 | 0.7040 | 0.7720 | 79.1220 | 19.1660 | 26.2020 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | SMPL-X RP | 3,970 | 0.292 | 0.452 | 0.558 | 1.520 | 4.573 | 8.875 | Measured |
| MotionStreamer Evaluator | SMPL-X RP | 4,042 | 0.3646 | 0.4998 | 0.5818 | 117.0279 | 21.4102 | 25.3629 | Measured |
| Motius Joint-Position Evaluator | SMPL-X RP | 4,034 | 0.3033 | 0.4638 | 0.5570 | 0.4881 | 47.8189 | 54.4397 | Measured |

### TP2M Results

Protocol: HumanML3D TP2M official-test selected-caption splits scored with
MotionStreamer-272. Each row uses the standard min/max length filter.

| Condition Frames | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ----------------: | ------: | --: | --: | --: | --: | ------: | --------: |
| 1 | 3,968 | 0.525 | 0.690 | 0.769 | 82.560 | 19.301 | 26.158 |
| 5 | 3,968 | 0.538 | 0.699 | 0.775 | 80.381 | 19.199 | 26.154 |
| 9 | 3,968 | 0.531 | 0.704 | 0.772 | 79.122 | 19.166 | 26.202 |

## Motion Representation

KIMODO operates in native skeleton spaces rather than HumanML3D-263. The Motius
runtime exposes native arrays such as `local_rot_mats`, `global_rot_mats`,
`posed_joints`, `root_positions`, `smooth_root_pos`, `foot_contacts`, and
`global_root_heading`.

For TP2M and cross-method evaluation, Motius also supports a `motion_135` bridge:
root translation `(3)` plus 22 local joint rotations in row-major 6D `(132)`.
Native SOMA prefix completion does not pass through this bridge. When
`motion_135` is supplied explicitly, the public pipeline uses the same
row-major 6D convention for prefix constraints and generated bridge output.

## Citation and License

```bibtex
@article{rempe2026kimodo,
  title={Kimodo: Scaling Controllable Human Motion Generation},
  author={Rempe, Davis and Petrovich, Mathis and Yuan, Ye and Zhang, Haotian and Peng, Xue Bin and Jiang, Yifeng and Wang, Tingwu and Iqbal, Umar and Minor, David and de Ruyter, Michael and Li, Jiefeng and Tessler, Chen and Lim, Edy and Jeong, Eugene and Wu, Sam and Hassani, Ehsan and Huang, Michael and Yu, Jin-Bey and Chung, Chaeyeon and Song, Lina and Dionne, Olivier and Kautz, Jan and Yuen, Simon and Fidler, Sanja},
  journal={arXiv preprint arXiv:2603.15546},
  year={2026}
}
```

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
