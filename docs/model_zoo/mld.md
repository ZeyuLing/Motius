<h1 align="center">MLD Model Card</h1>

<p align="center">
  <strong>Motion Latent Diffusion, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2212.04048">Paper</a> ·
  <a href="https://chenfengye.github.io/motion-latent-diffusion/">Project Page</a> ·
  <a href="https://github.com/ChenFengYe/motion-latent-diffusion">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MLD-HumanML3D">Motius Checkpoint</a>
</p>

MLD is the text-to-motion baseline from *Executing Your Commands via Motion
Diffusion in Latent Space* (Chen et al., CVPR 2023). This Motius release
provides a native inference pipeline with the MLD motion VAE, latent diffusion
denoiser, DDIM scheduler, and frozen SentenceT5 text wrapper.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/3ed06140-49fc-4826-9d78-25d178e752ee" controls></video> | [MP4](https://github.com/user-attachments/assets/3ed06140-49fc-4826-9d78-25d178e752ee) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=mld) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/3ed06140-49fc-4826-9d78-25d178e752ee" controls></video> |
| in a fighting stance, person punches downward with their right hand. | <video src="https://github.com/user-attachments/assets/d81f4fbf-72af-4017-adbb-771c6ea8e355" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/385e885a-b125-4986-8a03-87c51685da40" controls></video> |

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
| Method | MLD, latent diffusion for human motion |
| Tasks | Text-to-Motion |
| Venue | CVPR 2023 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | SentenceT5-Large, frozen |
| Default sampler | DDIM, 50 inference steps |
| Checkpoint | [`ZeyuLing/Motius-MLD-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-MLD-HumanML3D) |
| Pipeline | `motius.pipelines.mld.MLDPipeline` |

The checkpoint artifact contains `vae.safetensors`, `denoiser.safetensors`,
`mld_config.json`, `Mean.npy`, `Std.npy`, and the frozen
`sentence-t5-large` encoder under `text_encoder/`. Loading the pipeline does
not resolve or download a second model.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.mld.MLDPipeline` |
| Bundle | `motius.models.mld.MLDBundle` |
| Shared MLD/LCM runtime | `motius.models.motionlcm.network` |

The runtime is independent from the original checkout for inference. Raw
upstream checkpoint conversion remains outside this public release surface.

## Quick Start

Install the Motius package and the runtime dependencies used by the MLD stack:

```bash
python -m pip install -e ".[dev]"
```

Run text-to-motion inference:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MLD-HumanML3D",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward then sits down"],
    [120],
    num_inference_steps=50,
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
| HumanML3D Official | Default | 4,042 | 0.518 | 0.716 | 0.816 | 0.297 | 2.950 | 9.628 | Measured |
| MotionStreamer Evaluator | Default | — | — | — | — | — | — | — | Not measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.5169 | 0.6850 | 0.7701 | 0.1399 | 36.3447 | 57.3461 | Measured |

## Motion Representation

MLD generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

MLD samples in latent space and decodes directly back to HumanML3D-263.
Conversion to SMPL or MotionStreamer-272 is only needed for
cross-representation evaluation.

## Citation and License

```bibtex
@inproceedings{chen2023executing,
  title={Executing Your Commands via Motion Diffusion in Latent Space},
  author={Chen, Xin and Jiang, Biao and Liu, Wen and Huang, Zilong and Fu, Bin and Chen, Tao and Yu, Gang},
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
