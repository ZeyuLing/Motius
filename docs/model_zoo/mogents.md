<h1 align="center">MoGenTS Model Card</h1>

<p align="center">
  <strong>Spatial-temporal joint token modeling for text-to-motion, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2409.17686">Paper</a> ·
  <a href="https://aigc3d.github.io/mogents/">Project Page</a> ·
  <a href="https://github.com/weihaosky/mogents">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MoGenTS-HumanML3D">Motius Checkpoint</a>
</p>

MoGenTS is the text-to-motion baseline from *MoGenTS: Motion Generation based
on Spatial-Temporal Joint Modeling* (Yuan et al., NeurIPS 2024). This Motius
release packages the dual-stream RVQ-VAE, 1D auxiliary token transformer, 2D
spatial-temporal token transformer, residual token refiners, optional length
estimator, CLIP ViT-B/32 text encoder loading, and HumanML3D-263
denormalization behind a consistent inference pipeline.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/cb678b01-8b12-456b-a0db-9bb28d999827" controls></video> | [MP4](https://github.com/user-attachments/assets/cb678b01-8b12-456b-a0db-9bb28d999827) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=mogents) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/cb678b01-8b12-456b-a0db-9bb28d999827" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/6951e102-c01d-43c7-9147-5fe0e8fcf1a4" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/a0d96ab5-c5f3-4dfc-94f9-23f77c14a8b4" controls></video> |

512px / 30fps H.264 video previews rendered from released HumanML3D test outputs.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |

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
| ---- | ----- |
| Method | MoGenTS, spatial-temporal discrete motion tokens |
| Tasks | Text-to-Motion |
| Venue | NeurIPS 2024 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | CLIP ViT-B/32, frozen |
| Tokenizer | Dual-stream RVQ-VAE, 1D auxiliary tokens plus 2D joint-token grid |
| Checkpoint | [`ZeyuLing/Motius-MoGenTS-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-MoGenTS-HumanML3D) |
| Pipeline | `motius.pipelines.mogents.MoGenTSPipeline` |

The checkpoint artifact contains `vq.safetensors`, `mask_aux.safetensors`,
`mask_ts.safetensors`, `res_aux.safetensors`, `res_ts.safetensors`,
`length_est.safetensors`, `clip.safetensors`, `mogents_config.json`,
`Mean.npy`, and `Std.npy`.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.mogents.MoGenTSPipeline` |
| Bundle | `motius.models.mogents.MoGenTSBundle` |
| Runtime | `motius.models.mogents.network` |

The runtime is independent from the original checkout for inference. Raw
upstream checkpoint conversion remains outside this public release surface.

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MoGenTS-HumanML3D",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward then turns around"],
    [120],
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale. If `lengths` is omitted, the packaged
length estimator samples a token length from the prompt embedding.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Default | 3,970 | 0.522 | 0.713 | 0.806 | 0.081 | 2.929 | 9.406 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.5910 | 0.7523 | 0.8138 | 109.8191 | 18.6038 | 25.3317 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.4623 | 0.6238 | 0.7138 | 0.0864 | 36.7141 | 56.5133 | Measured |

## Motion Representation

MoGenTS generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

The model tokenizes motion into a 1D auxiliary stream and a 2D joint-token map.
The 2D stream preserves spatial-temporal structure before decoding back to the
standard 263-dim HumanML3D representation.

## Citation and License

```bibtex
@inproceedings{yuan2024mogents,
  title={MoGenTS: Motion Generation based on Spatial-Temporal Joint Modeling},
  author={Yuan, Weihao and Shen, Weichao and He, Yisheng and Dong, Yuan and Gu, Xiaodong and Dong, Zilong and Bo, Liefeng and Huang, Qixing},
  booktitle={Advances in Neural Information Processing Systems},
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
