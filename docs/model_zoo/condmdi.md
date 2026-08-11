<h1 align="center">CondMDI Model Card</h1>

<p align="center">
  <strong>Text-guided motion synthesis with flexible frame and joint controls.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2405.11126">Paper</a> ·
  <a href="https://setarehc.github.io/CondMDI/">Project Page</a> ·
  <a href="https://github.com/setarehc/diffusion-motion-inbetweening">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-condmdi-humanml3d">Motius Checkpoint</a>
</p>

CondMDI is the unified diffusion model from *Flexible Motion In-betweening
with Diffusion Models* (Cohan et al., SIGGRAPH 2024). It accepts text together
with arbitrary observed frames or joint subsets. The Motius release packages
the official randomly sampled frames-and-joints checkpoint behind one pipeline
for text-to-motion, keyframe in-betweening, trajectory control, and partial-body
control.

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
| Text-to-Motion | hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/40124d7f-8d77-48ff-8fc3-072c56eb0eb5" controls></video> | [MP4](https://github.com/user-attachments/assets/40124d7f-8d77-48ff-8fc3-072c56eb0eb5) · [All cases](https://zeyuling-t2m-humanml3d-leaderboard.static.hf.space/cases/index.html?method=condmdi) |
| Temporal Motion Completion | person walking at a average pace forward, swaying arms and torso with a sense of swagger<br><sub>Prediction: first 20%; condition frames 0-59 of 300</sub> | <video src="https://github.com/user-attachments/assets/5b89ea1a-77ae-4d83-8be4-4c08b3726137" controls></video> | [MP4](https://github.com/user-attachments/assets/5b89ea1a-77ae-4d83-8be4-4c08b3726137) · [All cases](https://zeyuling-temporal-condition-leaderboard.static.hf.space/cases/pre20/index.html?method=condmdi&case=004822) |
| Kinematic Motion Control | Text plus spatial motion constraints | <video src="https://github.com/user-attachments/assets/6ca7d278-9f4d-4c14-a50f-9b148e683bfe" controls></video> | [MP4](https://github.com/user-attachments/assets/6ca7d278-9f4d-4c14-a50f-9b148e683bfe) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input | SMPL Preview |
| ---------- | ------------ |
| the person swings a golf club. | <video src="https://github.com/user-attachments/assets/cfd76ccc-cc41-477f-ae7c-4a46dacd6698" controls></video> |
| hands in fighting position while the left foot kicks aggressively up and over. | <video src="https://github.com/user-attachments/assets/40124d7f-8d77-48ff-8fc3-072c56eb0eb5" controls></video> |
| the person who does arms straight out and then it’s doing something with their right hand in front of their face. | <video src="https://github.com/user-attachments/assets/c70c7c82-d6a7-49cd-8186-8adfdce63cf9" controls></video> |

512px / 30fps H.264 video previews rendered from released HumanML3D test outputs.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Motion | `infer_text_to_motion` | [Benchmark and examples](../leaderboards/README.md#text-to-motion) |
| Temporal Motion Completion | `infer_temporal_motion_completion` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/temporal-condition-leaderboard) |
| Kinematic Motion Control | `infer_kinematic_motion_control` | [Benchmark and examples](../leaderboards/kinematic_motion_control.md) |

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
| Method | Conditional Motion Diffusion In-betweening (CondMDI) |
| Tasks | Text-to-Motion, Temporal Motion Completion, Kinematic Motion Control |
| Venue | SIGGRAPH 2024 |
| Training data | HumanML3D |
| Native representation | HumanML3D-263 with absolute root rotation and translation, 20 fps |
| Public I/O representation | Standard HumanML3D-263, physical scale, 20 fps |
| Text encoder | OpenAI CLIP ViT-B/32, frozen |
| Default sampler | DDIM, 100 steps, classifier-free guidance 2.5 |
| Checkpoint | [`ZeyuLing/motius-condmdi-humanml3d`](https://huggingface.co/ZeyuLing/motius-condmdi-humanml3d) |
| Pipeline | `motius.pipelines.condmdi.CondMDIPipeline` |

[Kinematic Motion Control benchmark protocol](../leaderboards/kinematic_motion_control.md)

The Hugging Face artifact contains the frozen OpenAI CLIP ViT-B/32 tensors,
SafeTensors diffusion weights, the exact network and diffusion configuration,
and the official absolute-root normalization statistics. No upstream source
checkout, dataset directory, or second model download is needed at runtime.

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.condmdi.CondMDIPipeline` |
| Bundle | `motius.models.condmdi.CondMDIBundle` |
| UNet and diffusion runtime | `motius.models.condmdi.network` |
| HumanML3D selected-caption runner | `tools/eval_condmdi_humanml3d.py` |
| Official checkpoint exporter | `tools/export_condmdi_hf.py` |

The vendored method runtime retains the upstream MIT license in
`motius/models/condmdi/LICENSE`.

## Quick Start

Install the method-specific dependencies:

```bash
pip install -e ".[condmdi]"
```

Text-to-motion generation:

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-condmdi-humanml3d",
    bundle_kwargs={"respacing": "ddim100"},
    device="cuda",
)

motions = pipe.infer_text_to_motion(
    ["a person walks forward and waves with the right hand"],
    [120],
    seed=42,
)
```

First-and-last-frame in-betweening uses a standard HML263 reference motion:

```python
controlled = pipe.infer_kinematic_motion_control(
    ["a person turns around and walks away"],
    [reference_hml263],
    control_mode="first_last",
    transition_length=10,
    seed=42,
)
```

Other built-in control modes include `start`, `sparse`, `prefix`, `suffix`,
`middle`, `trajectory`, `lower_body`, `pelvis_feet`, `pelvis_vr`, and `joints`.
For arbitrary controls, pass an `(B, 263, 1, T)` Boolean `observation_mask` or
provide `keyframe_indices`. All returned arrays have shape `(T, 263)` in the
standard, denormalized HumanML3D representation.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Motion | [Published results](../leaderboards/hf_space_t2m_humanml3d/t2m_results.json) | HumanML3D semantic, physical, and paper rows |
| Temporal Motion Completion | [Published results](../leaderboards/hf_space_temporal_condition/temporal_control_results.json) | HumanML3D temporal settings; normalized uTMR FID |
| Kinematic Motion Control | [Published results](../leaderboards/kinematic_motion_control.md) | Native-skeleton protocol; rows remain pending |

### Canonical Temporal-Completion Snapshot

#### Temporal Control · Motius normalized space

| Setting | n | R@1 | R@2 | R@3 | Motius FID (normalized) | MM-Dist | Diversity | Constraint error (cm) | Foot skating |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| temporal_start_1f | 4,012 | 0.4918 | 0.6613 | 0.7506 | 0.0579 | 34.1245 | 55.3821 | 1.4378 | 0.1600 |
| temporal_pre20 | 4,012 | 0.3744 | 0.5177 | 0.5998 | 0.2333 | 40.8546 | 51.5968 | 1.6664 | 0.0730 |
| temporal_pre20_uncond | 4,012 | 0.3335 | 0.4706 | 0.5530 | 0.3015 | 43.2236 | 50.4369 | 1.6664 | 0.0580 |
| temporal_both_1f | 4,012 | 0.5614 | 0.7343 | 0.8143 | 0.0301 | 31.6146 | 54.5248 | 2.2058 | 0.1830 |
| temporal_mid80 | 4,012 | 0.4661 | 0.6219 | 0.7064 | 0.1455 | 36.8364 | 52.6705 | 2.4970 | 0.1240 |
| temporal_mid80_uncond | 4,012 | 0.3978 | 0.5472 | 0.6325 | 0.2113 | 39.7552 | 51.6817 | 2.4968 | 0.1080 |
| temporal_adaptive_keyframes | 4,012 | 0.6838 | 0.8432 | 0.9034 | 0.0023 | 27.8305 | 53.7278 | 2.7806 | 0.2417 |
| temporal_adaptive_keyframes_uncond | 4,012 | 0.6748 | 0.8359 | 0.8986 | 0.0024 | 28.1853 | 53.6503 | 2.7816 | 0.2330 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


### Text-to-Motion

Protocol: all 4,042 motions are generated from the HumanML3D selected-caption
test manifest. The official evaluator consumes 3,970 valid HumanML3D clips;
the MotionStreamer retrieval evaluator consumes 4,032 complete batch entries;
the Motius evaluator pairs 4,034 SMPL-22 motions. Results use one deterministic
generation per caption and one metric repeat. For FID and MM-Dist, lower is
better.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| --------- | ------: | --: | --: | --: | --: | ------: | --------: |
| HumanML3D Official | 3,970 | 0.449 | 0.642 | 0.749 | 0.294 | 3.218 | 9.795 |
| MotionStreamer Evaluator | 4,032 | 0.4526 | 0.6111 | 0.7016 | 121.8374 | 19.9702 | 25.4639 |
| Motius Joint-Position Evaluator | 4,034 | 0.4296 | 0.6039 | 0.7024 | 0.1919 | 39.1271 | 55.7954 |

The Motius row reports L2-normalized uTMR FID. MotionStreamer and Motius
evaluation first convert every output through the same SMPL-22 skeleton bridge.

Physical diagnostics use all 4,042 converted SMPL motions. Lower is better for
all metrics; PoseQ is the MBench NRDF pose-quality score.

| Slide | Float | Jitter | Dynamic | Penetration | PoseQ |
| ----: | ----: | -----: | ------: | ----------: | ----: |
| 4.222 | 18.689 | 6.937 | 21.509 | 0.000 | 1.830 |

### Motion Control

Control results use 4,012 HumanML3D test motions. `Start 1f` observes the first
frame, `Both 1f` observes the first and last frames, `Prefix 20` observes the
first 20 frames, and `Middle 80` observes a centered 80-frame interval.

| Setting | Evaluator | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity |
| ------- | --------- | --: | --: | --: | --: | ------: | --------: |
| Start 1f | MotionStreamer | 0.529 | 0.688 | 0.766 | 64.106 | 18.672 | 26.462 |
| Start 1f | Motius Joint-Position | 0.4918 | 0.6613 | 0.7506 | 0.0579 | 34.1245 | 55.3821 |
| Both 1f | MotionStreamer | 0.568 | 0.730 | 0.801 | 54.043 | 18.186 | 26.787 |
| Both 1f | Motius Joint-Position | 0.5614 | 0.7343 | 0.8143 | 0.0301 | 31.6146 | 54.5248 |
| Prefix 20 | MotionStreamer | 0.402 | 0.536 | 0.596 | 166.292 | 21.075 | 24.323 |
| Prefix 20 | Motius Joint-Position | 0.3744 | 0.5177 | 0.5998 | 0.2333 | 40.8546 | 51.5968 |
| Middle 80 | MotionStreamer | 0.484 | 0.628 | 0.707 | 123.567 | 19.812 | 25.010 |
| Middle 80 | Motius Joint-Position | 0.4661 | 0.6219 | 0.7064 | 0.1455 | 36.8364 | 52.6705 |

The Motius rows above are the canonical normalized-FID rows from the Temporal
Motion Completion Leaderboard. MotionStreamer rows remain in the independent
MotionStreamer evaluator space.

The following reconstruction and physical diagnostics are computed on the same
4,012 cases after conversion to the shared SMPL-22 skeleton. MPJPE and P-MPJPE
are in meters; lower is better for every column.

| Setting | Full MPJPE | Generated-region MPJPE | P-MPJPE | Jitter | Foot skating |
| ------- | ----------: | ---------------------: | -------: | -----: | -----------: |
| Start 1f | 0.1339 | 0.1345 | 0.0126 | 46.206 | 0.1601 |
| Both 1f | 0.1134 | 0.1144 | 0.0206 | 49.102 | 0.1829 |
| Prefix 20 | 0.1007 | 0.1235 | 0.0105 | 25.850 | 0.0726 |
| Middle 80 | 0.0945 | 0.1138 | 0.0189 | 34.526 | 0.1240 |

### Reproduction Check

The migrated network was checked against the official implementation using the
same checkpoint, text embedding, input tensor, and diffusion timestep. A single
UNet forward pass differs by at most `1.41e-5` (`6.45e-7` mean absolute error).
For a complete 100-step fp16 sample, accumulated mean absolute error is
`8.62e-4` (`1.59e-2` maximum).

## Motion Representation

The official CondMDI model changes the four root channels of HumanML3D-263
from root-relative velocities to absolute yaw and horizontal translation. All
remaining joint, rotation, velocity, and contact channels keep their original
HumanML3D layout.

Motius performs this conversion inside the pipeline:

1. Standard HML263 input is integrated into the official absolute-root form.
2. The official normalization statistics are applied before diffusion.
3. The generated root trajectory is converted back to standard relative
   HML263 before it is returned.

This keeps public CondMDI outputs compatible with the representation toolkit,
SMPL renderer, and all three T2M evaluators. The conversion round-trip matches
the official formulation to floating-point precision for every recoverable
frame; as with standard HML263, the final forward root delta is not encoded.

## Citation and License

```bibtex
@inproceedings{cohan2024flexible,
  title={Flexible Motion In-betweening with Diffusion Models},
  author={Cohan, Setareh and Tevet, Guy and Reda, Daniele and Peng, Xue Bin and van de Panne, Michiel},
  booktitle={ACM SIGGRAPH 2024 Conference Proceedings},
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
