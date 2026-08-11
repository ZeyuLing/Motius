<h1 align="center">MoMask Model Card</h1>

<p align="center">
  <strong>Generative masked modeling for text-to-motion, packaged as a Motius pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2312.00063">Paper</a> ·
  <a href="https://ericguo5513.github.io/momask/">Project Page</a> ·
  <a href="https://github.com/EricGuo5513/momask-codes">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MoMask-HumanML3D">Motius Checkpoint</a>
</p>

MoMask is the text-to-motion baseline from *MoMask: Generative Masked Modeling
of 3D Human Motions* (Guo et al., CVPR 2024). This Motius release packages the
RVQ-VAE tokenizer, masked token generator, residual token refiner, optional
length estimator, CLIP ViT-B/32 text encoder loading, and HumanML3D-263
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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/a71b6c90-d1c5-40f6-8c00-6ac94f63ce23" controls></video> | [MP4](https://github.com/user-attachments/assets/a71b6c90-d1c5-40f6-8c00-6ac94f63ce23) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=momask) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/a71b6c90-d1c5-40f6-8c00-6ac94f63ce23" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/55b7a805-e423-4dcb-b78f-1cb19e875e7b" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/b7cae881-9511-4f11-965e-4a905e401686" controls></video> |

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
| Method | MoMask, masked discrete motion token generation |
| Tasks | Text-to-Motion |
| Venue | CVPR 2024 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | CLIP ViT-B/32, frozen |
| Tokenizer | RVQ-VAE, 6 residual quantizers, 512-code codebook |
| Checkpoint | [`ZeyuLing/Motius-MoMask-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-MoMask-HumanML3D) |
| Pipeline | `motius.pipelines.momask.MoMaskPipeline` |

The checkpoint artifact contains `vq.safetensors`, `t2m_trans.safetensors`,
`res_trans.safetensors`, `length_est.safetensors`, `clip.safetensors`,
`momask_config.json`, `Mean.npy`, and `Std.npy`.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.momask.MoMaskPipeline` |
| Bundle | `motius.models.momask.MoMaskBundle` |
| Runtime | `motius.models.momask.network` |

The runtime is independent from the original checkout for inference. Raw
upstream checkpoint conversion remains outside this public release surface.

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MoMask-HumanML3D",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward then sits down"],
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
| HumanML3D Official | Default | 3,970 | 0.516 | 0.709 | 0.804 | 0.097 | 2.990 | 9.460 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.6404 | 0.7974 | 0.8609 | 21.0729 | 18.1216 | 25.9789 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.5665 | 0.7540 | 0.8356 | 0.0782 | 33.3118 | 56.6110 | Measured |

## Motion Representation

MoMask generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

The RVQ-VAE downsamples motion by a factor of four frames. A 196-frame motion
maps to 49 token positions, each represented by 6 residual quantizers.

## Citation and License

```bibtex
@inproceedings{guo2024momask,
  title={MoMask: Generative Masked Modeling of 3D Human Motions},
  author={Guo, Chuan and Mu, Yuxuan and Javed, Muhammad Gohar and Wang, Sen and Cheng, Li},
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
