<h1 align="center">T2M-GPT Model Card</h1>

<p align="center">
  <strong>Discrete text-to-motion generation, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2301.06052">Paper</a> ·
  <a href="https://mael-zys.github.io/T2M-GPT/">Project Page</a> ·
  <a href="https://github.com/Mael-zys/T2M-GPT">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-T2M-GPT-HumanML3D">Motius Checkpoint</a>
</p>

T2M-GPT is the text-to-motion baseline from *T2M-GPT: Generating Human Motion
from Textual Descriptions with Discrete Representations* (Zhang et al., CVPR
2023). This Motius release packages the VQ-VAE motion tokenizer, the
cross-conditional GPT sampler, CLIP text encoder loading, and HumanML3D-263
denormalization behind a consistent pipeline.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/8af79649-8c9f-48cc-a4d5-b05e30a098ce" controls></video> | [MP4](https://github.com/user-attachments/assets/8af79649-8c9f-48cc-a4d5-b05e30a098ce) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=t2mgpt) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/8af79649-8c9f-48cc-a4d5-b05e30a098ce" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/9b6c07ff-bea1-4388-8f0d-90b2e57f43cf" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/d37033ff-08d7-45c2-ab22-06042e339c3b" controls></video> |

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
| Method | T2M-GPT, autoregressive discrete motion tokens |
| Tasks | Text-to-Motion |
| Venue | CVPR 2023 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | CLIP ViT-B/32, frozen |
| Tokenizer | HumanVQVAE, 512-code codebook |
| Checkpoint | [`ZeyuLing/Motius-T2M-GPT-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-T2M-GPT-HumanML3D) |
| Pipeline | `motius.pipelines.t2mgpt.T2MGPTPipeline` |

The checkpoint artifact contains `vq.safetensors`, `gpt.safetensors`,
`clip.safetensors`, `t2mgpt_config.json`, `Mean.npy`, and `Std.npy`.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.t2mgpt.T2MGPTPipeline` |
| Bundle | `motius.models.t2mgpt.T2MGPTBundle` |
| Network | `motius.models.t2mgpt.network` |

The inference path intentionally keeps the GPT in training mode during token
sampling, matching the released T2M-GPT sampling distribution.

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-T2M-GPT-HumanML3D",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward then sits down"],
    [120],
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale. If `lengths` is omitted, the GPT
decides sequence length through its EOS token.

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
| HumanML3D Official | Default | 3,944 | 0.490 | 0.678 | 0.776 | 0.225 | 3.145 | 9.624 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.5516 | 0.7056 | 0.7788 | 25.4913 | 19.0912 | 25.5949 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.4869 | 0.6520 | 0.7359 | 0.1148 | 36.3177 | 55.5376 | Measured |

## Motion Representation

T2M-GPT generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

The VQ-VAE downsamples motion tokens by a factor of four frames. A 196-frame
motion therefore maps to at most 49 discrete tokens from a 512-entry codebook.

## Citation and License

```bibtex
@inproceedings{zhang2023t2m,
  title={T2M-GPT: Generating Human Motion from Textual Descriptions with Discrete Representations},
  author={Zhang, Jianrong and Zhang, Yangsong and Cun, Xiaodong and Huang, Yong and Zhang, Yong and Zhao, Hongwei and Lu, Hongtao and Shen, Xi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2023}
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
