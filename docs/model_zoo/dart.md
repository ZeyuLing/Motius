<h1 align="center">DART Model Card</h1>

<p align="center">
  <strong>DartControl packaged as a Motius autoregressive text-to-motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.05260">Paper</a> ·
  <a href="https://zkf1997.github.io/DART/">Project Page</a> ·
  <a href="https://github.com/zkf1997/DART">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-dart-humanml3d">Motius Checkpoint</a>
</p>

DART, short for DartControl, is a diffusion-based autoregressive motion model
for real-time text-driven motion control. This Motius package exposes the
HumanML3D DART276 rollout path through a `ModelBundle` and task-facing pipeline.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/bf09ae84-0e85-4880-84ca-34d2ee7be079" controls></video> | [MP4](https://github.com/user-attachments/assets/bf09ae84-0e85-4880-84ca-34d2ee7be079) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=dart) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/bf09ae84-0e85-4880-84ca-34d2ee7be079" controls></video> |
| a person jumps with legs open while clapping with hands over head simultaneously. | <video src="https://github.com/user-attachments/assets/989f57a8-3ca6-4c04-91f5-2df7ce75c6e1" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/d3b97464-0b44-4224-af3e-99b4c97aa4f8" controls></video> |

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
| Training motion | 20 fps (DART/HumanML3D motion) |
| Public preview | 30 fps, duration-preserving 20→30 fps resampling |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Method | DART / DartControl |
| Tasks | Text-to-Motion |
| Venue | ICLR 2025 |
| Motion representation | DART276, 20 fps |
| Backbone | Motion primitive VAE + latent diffusion denoiser |
| Default guidance scale | `5.0` |
| Checkpoint | [`ZeyuLing/motius-dart-humanml3d`](https://huggingface.co/ZeyuLing/motius-dart-humanml3d) |
| Pipeline | `motius.pipelines.dart.DARTPipeline` |

The checkpoint artifact contains the DART denoiser, MVAE, runtime configuration, seed motion, and DART276 normalization/text-embedding assets. It intentionally does not include license-controlled SMPL-H or SMPL-X body model files; install those locally under `checkpoints/body_models` or set `MOTIUS_BODY_MODEL_DIR` before full rollout or SMPL export.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.dart.DARTPipeline` |
| Bundle | `motius.models.dart.DARTBundle` |
| Sampling and primitive rollout | `motius.pipelines.dart` |
| Rotation/primitive processing | `motius.motion.representation.dart` |
| Primitive training | `motius.trainers.dart.DARTTrainer` |
| Prepared data recipe | `configs/dart/prepared_motion.py` |

The default rollout follows the original FlowMDM evaluation path: SMPL-H seed
features and SMPL-X reconstruction. `MOTIUS_BODY_MODEL_DIR` must contain the
licensed `smplh/SMPLH_MALE.pkl`, `smplh/SMPLH_FEMALE.pkl` and corresponding
`smplx/SMPLX_MALE.npz`, `smplx/SMPLX_FEMALE.npz` files. These assets are installed
separately and are never bundled into Motius or its checkpoints.

Default public translation preserves the original SMPL translation; it does not
add a per-frame floor correction. Export metadata preserves explicit alternative
translation policies. Rotation/geometry parity uses PyTorch3D 0.7.9 arithmetic.

Independent original-source checks cover four complete DDPM/DDIM rollouts and
both genders' full body losses/gradients. AdamW, clipping, learning-rate decay,
EMA and explicit VAE latent calibration match the original code. Each native
training stage also passes two-GPU updates and public CLI resume versus an
uninterrupted run, with exact neural, EMA and optimizer states. A trained,
calibrated EMA pair survives offline automatic export and inference exactly.
See the [rollout](../architecture/dart-rollout-source.json),
[body objectives](../architecture/dart-body-training-source.json) and
[optimizer](../architecture/dart-optimizer-source.json) reports. These checks
use explicit prepared primitives; they do not certify the original full-corpus
multi-primitive curriculum, AMP or benchmark convergence.

## Quick Start

Install Motius:

```bash
python -m pip install -e ".[dev]"
```

Run the SMPL-sequence export adapter after installing the licensed body-model assets locally:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-dart-humanml3d",
    device="cuda",
)

native_sequences = pipe.infer_text_to_motion(
    ["a person walks forward, then turns left"],
    [196],
    seed=0,
)

smpl_sequences = pipe.infer_t2m_smpl(
    ["a person walks forward, then turns left"],
    [196],
    seed=0,
)
```

`infer_t2m_smpl` is an adapter for rendering/evaluation conversion. DART remains native `DART276`; SMPL and `motion_135` tensors are export adapters, not separate checkpoint representations.

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
| HumanML3D Official | Default | 3,970 | 0.401 | 0.592 | 0.700 | 1.846 | 3.709 | 9.867 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.5476 | 0.7245 | 0.7937 | 127.8302 | 18.5312 | 26.2611 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.4249 | 0.6064 | 0.7019 | 0.2029 | 38.7637 | 56.9485 | Measured |

## Motion Representation

DART uses a 276-dimensional motion-primitive representation (`DART276`) in the
native primitive pipeline. Helper adapters may export SMPL-style tensors for rendering
or evaluator conversion, but those adapters are not the model's native motion
representation and are not a published checkpoint variant.

The underlying runtime uses motion primitives and an autoregressive rollout
loop. The public checkpoint documents this tensor contract in `model_index.json`.

## Citation and License

```bibtex
@inproceedings{Zhao:DartControl:2025,
  title={{DartControl}: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control},
  author={Zhao, Kaifeng and Li, Gen and Tang, Siyu},
  booktitle={The Thirteenth International Conference on Learning Representations},
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
