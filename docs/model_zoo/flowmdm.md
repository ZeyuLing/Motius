<h1 align="center">FlowMDM Model Card</h1>

<p align="center">
  <strong>Seamless multi-prompt human motion composition, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2402.15509">Paper</a> ·
  <a href="https://barquerogerman.github.io/FlowMDM/">Project Page</a> ·
  <a href="https://github.com/BarqueroGerman/FlowMDM">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-FlowMDM-HumanML3D">HumanML3D Checkpoint</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-flowmdm-babel">BABEL Checkpoint</a>
</p>

FlowMDM is the motion composition baseline from *Seamless Human Motion
Composition with Blended Positional Encodings* (Barquero et al., CVPR 2024).
This Motius release packages the MDM-style diffusion model, blended positional
encoding sampler, HumanML3D statistics, and text-to-motion / multi-prompt
pipeline methods without requiring the original checkout.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/a5234eca-e75c-4379-bfe8-a53eb76e866b" controls></video> | [MP4](https://github.com/user-attachments/assets/a5234eca-e75c-4379-bfe8-a53eb76e866b) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=flowmdm&case=001840) |
| Temporal Motion Completion | person walking at a average pace forward, swaying arms and torso with a sense of swagger<br><sub>Prediction: first 20%; condition frames 0-59 of 300</sub> | <video src="https://github.com/user-attachments/assets/3c164739-3e53-449c-9989-ad1b4141a1b8" controls></video> | [MP4](https://github.com/user-attachments/assets/3c164739-3e53-449c-9989-ad1b4141a1b8) · [All cases](https://zeyuling-temporal-condition-leaderboard.static.hf.space/cases/pre20/index.html?method=flowmdm&case=004822) |
| Sequential Text-to-Motion | A person walks forward. → A person sits down. → A person rests. → A person stands up. → A person walks back.<br><sub>Five captioned segments; the active caption and segment timeline are embedded in the video</sub> | <video src="https://github.com/user-attachments/assets/2488a222-d543-4215-867c-14329c79bbea" controls></video> | [MP4](https://github.com/user-attachments/assets/2488a222-d543-4215-867c-14329c79bbea) · [All cases](https://zeyuling-babel-sequential-generation-leaderboard.static.hf.space/cases/index.html?method=flowmdm&case=val_919) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/5633a083-e15e-48cc-a0af-2af75f1f8920" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/40093579-ad4c-40fa-a978-74502d3d0211" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/4256ec99-694c-44b9-a259-1588ffba1bb8" controls></video> |

512px / 30fps H.264 video previews rendered from released HumanML3D test outputs.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |
| Temporal Motion Completion | `infer_temporal_motion_completion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) |
| Sequential Text-to-Motion | `infer_sequential_text_to_motion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | HumanML3D checkpoint: 20 fps; BABEL checkpoint: 30 fps |
| Public preview | T2M/temporal: 30 fps after duration-preserving 20→30 fps resampling; sequential BABEL: 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | FlowMDM, diffusion with blended positional encodings |
| Tasks | Text-to-Motion, Temporal Motion Completion, Sequential Text-to-Motion |
| Venue | CVPR 2024 |
| Motion representation | HumanML3D-263 at 20 fps; BABEL-135 at 30 fps |
| Checkpoints | [`HumanML3D`](https://huggingface.co/ZeyuLing/Motius-FlowMDM-HumanML3D), [`BABEL`](https://huggingface.co/ZeyuLing/motius-flowmdm-babel) |
| Pipeline | `motius.pipelines.flowmdm.FlowMDMPipeline` |

The HumanML3D artifact contains `model000500000.pt`, `args.json`, `Mean.npy`,
`Std.npy`, and `model_index.json`. The BABEL artifact contains the official
`model001300000.pt`, `args.json`, BABEL normalization statistics, license, and
Motius model index.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.flowmdm.FlowMDMPipeline` |
| Bundle | `motius.models.flowmdm.FlowMDMBundle` |
| Runtime | `motius.models.flowmdm.network` |

The SMPL visualizer branch from the original implementation is stubbed for T2M
inference because the released HumanML3D checkpoint predicts HumanML3D-263
features directly.

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-FlowMDM-HumanML3D",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward then sits down"],
    [120],
)
```

Sequential generation is exposed through the BABEL checkpoint with the same
pipeline class:

```python
babel_pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-flowmdm-babel",
    bundle_kwargs={"device": "cuda"},
    device="cuda",
)
motions = babel_pipe.infer_sequential_text_to_motion(
    [["a person walks forward", "then turns around"]],
    [[80, 80]],
)
```

The HumanML3D artifact also exposes observed-prefix completion:

```python
completed = pipe.infer_temporal_motion_completion(
    ["a person continues walking and turns left"],
    [160],
    [reference_hml263],
    condition_num_frames=5,
)
```

`motions` is a list of NumPy arrays. HumanML3D outputs have shape `(T, 263)`;
BABEL outputs have shape `(T, 135)`. Both are returned in physical scale.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Temporal Motion Completion | [Published results](../leaderboards/hf_space_temporal_condition/temporal_control_results.json) | HumanML3D temporal settings; normalized uTMR FID |
| Sequential Text-to-Motion | [Published results](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | BABEL semantic and transition metrics; normalized uTMR FID |

### Canonical Temporal-Completion Snapshot

#### Temporal Control · Motius normalized space

| Setting | n | R@1 | R@2 | R@3 | Motius FID (normalized) | MM-Dist | Diversity | Constraint error (cm) | Foot skating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| temporal_start_1f | 4,012 | 0.4666 | 0.6397 | 0.7295 | 0.0608 | 34.7605 | 55.3023 | 1.4378 | 0.1330 |
| temporal_pre20 | 4,012 | 0.5403 | 0.7106 | 0.7923 | 0.0349 | 32.2914 | 55.1279 | 1.6983 | 0.1140 |
| temporal_pre20_uncond | 4,012 | 0.2639 | 0.3827 | 0.4592 | 0.3000 | 45.0807 | 49.5066 | 1.6986 | 0.0490 |
| temporal_both_1f | 4,012 | 0.4825 | 0.6575 | 0.7477 | 0.0519 | 34.0014 | 55.1943 | 4.6053 | 0.1530 |
| temporal_mid80 | 4,012 | 0.5534 | 0.7280 | 0.8098 | 0.0292 | 31.6779 | 54.5792 | 4.3873 | 0.1270 |
| temporal_mid80_uncond | 4,012 | 0.2891 | 0.4100 | 0.4848 | 0.2689 | 43.7286 | 50.3749 | 3.6646 | 0.0630 |

#### TP2M Prefix · MotionStreamer-272 space

| Setting | n | R@1 | R@2 | R@3 | MotionStreamer FID | MM-Dist | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-frame prefix | 3,968 | 0.4490 | 0.6300 | 0.7060 | 83.7730 | 19.8720 | 26.3650 |
| 5-frame prefix | 3,968 | 0.4810 | 0.6540 | 0.7290 | 75.8530 | 19.4560 | 26.4670 |
| 9-frame prefix | 3,968 | 0.4900 | 0.6640 | 0.7420 | 71.3380 | 19.2620 | 26.6250 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Default | 3,970 | 0.439 | 0.636 | 0.744 | 0.327 | 3.387 | 9.942 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.4737 | 0.6496 | 0.7312 | 36.3767 | 20.0018 | 25.1783 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.4390 | 0.6153 | 0.7111 | 0.1239 | 37.4096 | 55.5127 | Measured |

### BABEL Sequential Results

Protocol: 1,295 eligible episodes from the processed official BABEL validation
split, containing 7,285 LLM-rewritten action intervals and 5,990 paired
transitions. Short actions are merged to at least 30 frames. Generated and GT
motions use the same neutral zero-beta SMPL-22 skeleton.

The previous 64-composition result used raw FlowMDM composition prompts and
independent reference pools, so it has been withdrawn. The corrected full-split
result is:

| Method | Episodes | Segments | R@1 | R@2 | R@3 | Normalized Semantic FID | MM-Dist | Normalized Transition FID | AUJ Gap |
| ------ | -------: | -------: | --: | --: | --: | -----------: | ------: | -------------: | ------: |
| BABEL GT | 1,295 | 7,285 | 0.3947 | 0.5515 | 0.6327 | 0.0000 | 44.5941 | 0.0000 | 0.0000 |
| FlowMDM BABEL | 1,295 | 7,285 | 0.2958 | 0.4217 | 0.5018 | 0.0843 | 46.7698 | 0.1092 | 34.4040 |

R-Precision uses official BABEL `act_cat` action-group multi-positive batches
of 32 and therefore scores 7,264 paired segments; normalized uTMR FID and diversity use all
7,285 segments. The evaluator encoder forward batch is also 32 in this measured
run, but it does not define the R-Precision candidate set. GT is a calibration
row and is excluded from method ranking.

Full protocol and diagnostic statistics are maintained on the
[`Sequential Text-to-Motion · BABEL` benchmark](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard).

### TP2M Results

FlowMDM also supports prefix-conditioned TP2M evaluation under the published
[`Temporal Motion Completion · HumanML3D` benchmark](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard)
protocol. These results are separate from the multi-prompt BABEL benchmark
above.

| Condition Frames | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ----------------: | ------: | --: | --: | --: | --: | ------: | --------: |
| 1 | 3,968 | 0.449 | 0.630 | 0.706 | 83.773 | 19.872 | 26.365 |
| 5 | 3,968 | 0.481 | 0.654 | 0.729 | 75.853 | 19.456 | 26.467 |
| 9 | 3,968 | 0.490 | 0.664 | 0.742 | 71.338 | 19.262 | 26.625 |

## Motion Representation

FlowMDM generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

## Citation and License

```bibtex
@inproceedings{barquero2024seamless,
  title={Seamless Human Motion Composition with Blended Positional Encodings},
  author={Barquero, German and Escalera, Sergio and Palmero, Cristina},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
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
