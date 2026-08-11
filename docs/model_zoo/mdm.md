<h1 align="center">MDM Model Card</h1>

<p align="center">
  <strong>Human Motion Diffusion Model, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2209.14916">Paper</a> ·
  <a href="https://openreview.net/forum?id=SJ1kSyO2jwu">OpenReview</a> ·
  <a href="https://guytevet.github.io/mdm-page/">Project Page</a> ·
  <a href="https://github.com/GuyTevet/motion-diffusion-model">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MDM-HumanML3D">Motius Checkpoint</a>
</p>

MDM is the text-to-motion baseline from *Human Motion Diffusion Model* (Tevet
et al., ICLR 2023). This Motius release provides an inference pipeline, a
checkpoint-loading bundle, and the Gaussian diffusion sampler needed to run the
HumanML3D checkpoint with the same public task interface used by other Motius
methods.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/d073e5ff-bcf5-4a2c-b3b2-e701f198ed79" controls></video> | [MP4](https://github.com/user-attachments/assets/d073e5ff-bcf5-4a2c-b3b2-e701f198ed79) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=mdm) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/d073e5ff-bcf5-4a2c-b3b2-e701f198ed79" controls></video> |
| the person swings a golf club. | <video src="https://github.com/user-attachments/assets/d6db6f62-37ad-447c-bf6c-47d6c78c255f" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/3785cf60-15f8-4f9f-b88f-2803a50285ee" controls></video> |

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
| Method | MDM, classifier-free diffusion for human motion |
| Tasks | Text-to-Motion |
| Venue | ICLR 2023 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | CLIP ViT-B/32, frozen |
| Default guidance scale | `2.5` |
| Checkpoint | [`ZeyuLing/Motius-MDM-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-MDM-HumanML3D) |
| Pipeline | `motius.pipelines.mdm.MDMPipeline` |

The checkpoint artifact contains `model.safetensors`, `mdm_config.json`,
`Mean.npy`, and `Std.npy`. The mean and standard-deviation files are the
HumanML3D training normalization statistics and are part of the checkpoint
contract.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.mdm.MDMPipeline` |
| Bundle | `motius.models.mdm.MDMBundle` |
| Network | `motius.models.mdm.network.MDM` |
| Diffusion sampler | `motius.models.mdm.network.diffusion` |
| Collation helper | `motius.models.mdm.network.collate` |

The network and Gaussian diffusion sampler are vendored for inference parity
with the released MDM checkpoint. Training-only geometry losses are represented
by explicit stubs, so unsupported training paths fail clearly.

## Quick Start

Install the Motius package and OpenAI CLIP:

```bash
python -m pip install -e ".[dev]"
python -m pip install git+https://github.com/openai/CLIP.git
```

Run text-to-motion inference:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MDM-HumanML3D",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward then sits down"],
    [120],
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 263)` and is
denormalized to HumanML3D physical scale.

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
| HumanML3D Official | Default | 3,970 | 0.462 | 0.660 | 0.763 | 0.400 | 3.248 | 9.966 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.5208 | 0.6937 | 0.7701 | 35.5169 | 19.4246 | 25.3383 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.4501 | 0.6290 | 0.7262 | 0.1438 | 37.5544 | 56.3969 | Measured |

## Motion Representation

MDM generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

For qualitative inspection and cross-representation evaluation, generated
HumanML3D-263 motions are retargeted to the repository SMPL motion format and
rendered as SMPL mesh videos.

## Citation and License

```bibtex
@inproceedings{
tevet2023human,
title={Human Motion Diffusion Model},
author={Guy Tevet and Sigal Raab and Brian Gordon and Yoni Shafir and Daniel Cohen-or and Amit Haim Bermano},
booktitle={The Eleventh International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=SJ1kSyO2jwu}
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
