<h1 align="center">MotionLCM Model Card</h1>

<p align="center">
  <strong>Latent consistency motion generation, packaged as a Motius Text-to-Motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2404.19759">Paper</a> ·
  <a href="https://dai-wenxun.github.io/MotionLCM-page/">Project Page</a> ·
  <a href="https://github.com/Dai-Wenxun/MotionLCM">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-MotionLCM-HumanML3D">Motius Checkpoint</a>
</p>

MotionLCM is the real-time controllable motion generation method from
*MotionLCM: Real-time Controllable Motion Generation via Latent Consistency
Model* (Dai et al., ECCV 2024). This Motius release exposes the text-to-motion
path: SentenceT5 text features, latent consistency sampling in the MLD latent
space, MLD VAE decoding, and HumanML3D-263 denormalization.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/ad25ec78-db68-4241-a444-cde7d9520c73" controls></video> | [MP4](https://github.com/user-attachments/assets/ad25ec78-db68-4241-a444-cde7d9520c73) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=motionlcm) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/ad25ec78-db68-4241-a444-cde7d9520c73" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/6bdc8832-19fb-49c2-8417-0ee321d6867a" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/08d4c798-38e1-412e-a7d2-6c971bb24a7e" controls></video> |

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
| Method | MotionLCM, latent consistency model for human motion |
| Tasks | Text-to-Motion |
| Venue | ECCV 2024 |
| Motion representation | HumanML3D-263, 20 fps |
| Text encoder | SentenceT5-Large, frozen |
| Default sampler | LCM, 1 inference step |
| Checkpoint | [`ZeyuLing/Motius-MotionLCM-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-MotionLCM-HumanML3D) |
| Pipeline | `motius.pipelines.motionlcm.MotionLCMPipeline` |

The checkpoint artifact contains `vae.safetensors`, `denoiser.safetensors`,
`motionlcm_config.json`, `Mean.npy`, `Std.npy`, and the frozen
`sentence-t5-large` encoder under `text_encoder/`. Loading the pipeline does
not resolve or download a second model.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.motionlcm.MotionLCMPipeline` |
| Bundle | `motius.models.motionlcm.MotionLCMBundle` |
| Shared MLD/LCM runtime | `motius.models.motionlcm.network` |

The released public surface covers the text-to-motion inference path. Raw
upstream checkpoint conversion and controllable MotionLCM variants remain
outside this scoped release.

## Quick Start

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-MotionLCM-HumanML3D",
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward then sits down"],
    [120],
    num_inference_steps=1,
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
| HumanML3D Official | Default | 4,042 | 0.509 | 0.708 | 0.811 | 0.340 | 2.969 | 9.641 | Measured |
| MotionStreamer Evaluator | Default | — | — | — | — | — | — | — | Not measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.5156 | 0.6915 | 0.7736 | 0.1537 | 36.5871 | 56.9460 | Measured |

## Motion Representation

MotionLCM generates HumanML3D-263 features at 20 fps. Per frame:

| Slice | Dim | Meaning |
| ----- | --- | ------- |
| `root_rot_vel` | 1 | root angular velocity |
| `root_lin_vel` | 2 | root linear velocity in the horizontal plane |
| `root_y` | 1 | root height |
| `ric_data` | 63 | local joint positions |
| `rot_data` | 126 | local joint rotations in continuous 6D format |
| `local_vel` | 66 | local joint velocities |
| `foot_contact` | 4 | binary foot-contact labels |

MotionLCM samples in the MLD latent space and decodes directly back to
HumanML3D-263. Conversion to SMPL or MotionStreamer-272 is only needed for
cross-representation evaluation.

## Citation and License

```bibtex
@inproceedings{dai2024motionlcm,
  title={MotionLCM: Real-time Controllable Motion Generation via Latent Consistency Model},
  author={Dai, Wenxun and Chen, Ling-Hao and Wang, Jingbo and Liu, Jinpeng and Dai, Bo and Tang, Yansong},
  booktitle={European Conference on Computer Vision},
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
