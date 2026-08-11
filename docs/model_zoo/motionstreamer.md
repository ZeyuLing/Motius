<h1 align="center">MotionStreamer Model Card</h1>

<p align="center">
  <strong>Streaming text-to-motion generation with a causal latent motion space.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2503.15451">Paper</a> ·
  <a href="https://zju3dv.github.io/MotionStreamer/">Project Page</a> ·
  <a href="https://github.com/zju3dv/MotionStreamer">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MotionStreamer-HumanML272">Motius Checkpoint</a>
</p>

MotionStreamer is a streaming text-to-motion model that combines a causal TAE,
a LLaMA-style autoregressive transformer, and a per-token diffusion head. This
Motius release packages the TAE, AR model, diffusion sampler, normalization
statistics, and task-facing pipeline without requiring the original checkout.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/74e3ccee-2f4c-4896-9124-0abaaccd7852" controls></video> | [MP4](https://github.com/user-attachments/assets/74e3ccee-2f4c-4896-9124-0abaaccd7852) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motionstreamer) |
| Temporal Motion Completion | Swaggering walk with observed frames / keyframes | <video src="https://github.com/user-attachments/assets/5effea2b-0934-40fd-a273-b69532ab193b" controls></video> | [MP4](https://github.com/user-attachments/assets/5effea2b-0934-40fd-a273-b69532ab193b) · [All cases](https://zeyuling-babel-sequential-generation-leaderboard.static.hf.space/cases/index.html?method=motionstreamer) |
| Sequential Text-to-Motion | A person walks forward. → A person sits down. → A person rests. → A person stands up. → A person walks back.<br><sub>Five captioned segments; the active caption and segment timeline are embedded in the video</sub> | <video src="https://github.com/user-attachments/assets/fe644de3-bee5-43e3-914a-5a659d950a73" controls></video> | [MP4](https://github.com/user-attachments/assets/fe644de3-bee5-43e3-914a-5a659d950a73) · [All cases](https://zeyuling-babel-sequential-generation-leaderboard.static.hf.space/cases/index.html?method=motionstreamer&case=val_919) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/74e3ccee-2f4c-4896-9124-0abaaccd7852" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/f3f15afd-dc10-4e8d-adff-5b232fd4abdd" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/e9ae790c-5eb4-4a13-89cb-fc029a6f688a" controls></video> |

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
| Training motion | 30 fps (MotionStreamer-272) |
| Public preview | 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | MotionStreamer, diffusion autoregression in causal latent space |
| Tasks | Text-to-Motion, Temporal Motion Completion, Sequential Text-to-Motion |
| Venue | ICCV 2025 |
| Motion representation | MotionStreamer-272, 30 fps |
| Text encoder | SentenceT5-XXL |
| Checkpoint | [`ZeyuLing/Motius-MotionStreamer-HumanML272`](https://huggingface.co/ZeyuLing/Motius-MotionStreamer-HumanML272) |
| Pipeline | `motius.pipelines.motionstreamer.MotionStreamerPipeline` |

The checkpoint artifact contains `tae.safetensors`, `ar.safetensors`,
`ms_config.json`, `Mean.npy`, `Std.npy`, `model_index.json`, and the frozen
`sentence-t5-xxl` encoder under `text_encoder/`. Loading the pipeline does not
resolve or download a second model.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motionstreamer.MotionStreamerPipeline` |
| Bundle | `motius.models.motionstreamer.MotionStreamerBundle` |
| Runtime | `motius.models.motionstreamer.network` |

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MotionStreamer-HumanML272",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person squats down and jumps up quickly"],
    [120],
)
```

Sequential streaming generation is exposed through the same pipeline:

```python
motions = pipe.infer_sequential_text_to_motion(
    [["a person walks forward", "then raises both hands"]],
    [[80, 80]],
    exact_lengths=True,
)
```

`exact_lengths=True` is intended for fixed-boundary protocols such as BABEL.
It supports actions longer than one 77-token autoregressive block and returns
every requested segment boundary exactly. Since one latent token spans four
frames, the pipeline decodes each segment with at most three alignment frames
and linearly resamples it back to the exact requested length.

TP2M uses a MotionStreamer-272 prefix:

```python
motions = pipe.infer_temporal_motion_completion(
    ["a person continues walking forward"],
    [160],
    [gt_motion_272],
    condition_num_frames=5,
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 272)` and is
denormalized to MotionStreamer-272 physical scale.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Temporal Motion Completion | [Published results](../leaderboards/hf_space_temporal_condition/temporal_control_results.json) | HumanML3D temporal settings; normalized uTMR FID |
| Sequential Text-to-Motion | [Published results](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard) | BABEL semantic and transition metrics; normalized uTMR FID |

### Canonical Temporal-Completion Snapshot

#### TP2M Prefix · MotionStreamer-272 space

| Setting | n | R@1 | R@2 | R@3 | MotionStreamer FID | MM-Dist | Diversity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-frame prefix | 3,904 | 0.6170 | 0.7800 | 0.8500 | 12.4790 | 16.8490 | 27.1340 |
| 5-frame prefix | 3,904 | 0.6280 | 0.7860 | 0.8530 | 11.2140 | 16.5860 | 27.1440 |
| 9-frame prefix | 3,904 | 0.6330 | 0.7880 | 0.8560 | 11.0770 | 16.4860 | 27.3810 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Default | 3,970 | 0.408 | 0.588 | 0.690 | 0.169 | 3.676 | 9.579 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.6303 | 0.7865 | 0.8498 | 12.2110 | 16.5810 | 27.4637 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.4400 | 0.5967 | 0.6811 | 0.0488 | 35.6739 | 53.8000 | Measured |

### Sequential Text-to-Motion · BABEL

The official MotionStreamer checkpoint is evaluated on all 1,295 processed
BABEL validation episodes with exact manifest lengths and canonical SMPL-22
joints. R-Precision uses action-group multi-positive recall batches of 32.

| Segments | R@1 | R@2 | R@3 | Normalized FID | MM-Dist | Diversity | Normalized Transition FID | AUJ Gap |
| -------: | --: | --: | --: | --: | ------: | --------: | -------------: | ------: |
| 7,285 | 0.2087 | 0.3136 | 0.3955 | 0.1205 | 49.3062 | 56.2576 | 0.1664 | 76.2889 |

See the [Sequential Text-to-Motion · BABEL benchmark](https://huggingface.co/spaces/ZeyuLing/babel-sequential-generation-leaderboard)
for the complete transition diagnostics and cross-method comparison.

### TP2M Results

Protocol: HumanML3D TP2M official-test selected-caption splits scored with the
MotionStreamer-272 evaluator.

| Condition Frames | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ----------------: | ------: | --: | --: | --: | --: | ------: | --------: |
| 1 | 3,904 | 0.617 | 0.780 | 0.850 | 12.479 | 16.849 | 27.134 |
| 5 | 3,904 | 0.628 | 0.786 | 0.853 | 11.214 | 16.586 | 27.144 |
| 9 | 3,904 | 0.633 | 0.788 | 0.856 | 11.077 | 16.486 | 27.381 |

## Motion Representation

MotionStreamer predicts a 272-dimensional global representation at 30 fps:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| root xz velocity | 2 | root planar velocity |
| root heading | 6 | continuous heading rotation |
| joint positions | 66 | 22 joints in xyz |
| joint velocities | 66 | 22 joint velocities |
| local rotations | 132 | 22 local 6D rotations |

The TAE temporal downsample factor is 4 frames per latent token. The pipeline
therefore clamps requested lengths to token-aligned frame counts.

## Citation and License

```bibtex
@inproceedings{xiao2025motionstreamer,
  title={MotionStreamer: Streaming Motion Generation via Diffusion-based Autoregressive Model in Causal Latent Space},
  author={Xiao, Lixing and Lu, Shunlin and Pi, Huaijin and Fan, Ke and Pan, Liang and Zhou, Yueer and Feng, Ziyong and Zhou, Xiaowei and Peng, Sida and Wang, Jingbo},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2025}
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
