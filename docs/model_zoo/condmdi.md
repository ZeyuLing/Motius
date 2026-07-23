<h1 align="center">CondMDI Model Card</h1>

<p align="center">
  <strong>Text-guided motion synthesis with flexible frame and joint controls.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2405.11126">Paper</a> |
  <a href="https://setarehc.github.io/CondMDI/">Project Page</a> |
  <a href="https://github.com/setarehc/diffusion-motion-inbetweening">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-condmdi-humanml3d">Motius Checkpoint</a>
</p>

CondMDI is the unified diffusion model from *Flexible Motion In-betweening
with Diffusion Models* (Cohan et al., SIGGRAPH 2024). It accepts text together
with arbitrary observed frames or joint subsets. The Motius release packages
the official randomly sampled frames-and-joints checkpoint behind one pipeline
for text-to-motion, keyframe in-betweening, trajectory control, and partial-body
control.

## Preview

| HumanML3D Sample | Input Text | SMPL Preview |
| ---------------- | ---------- | ------------ |
| `014457` | the person swings a golf club. | ![CondMDI HumanML3D 014457 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/condmdi/condmdi_humanml3d_014457_smpl_mesh_512_30fps.gif) |
| `001840` | hands in fighting position while the left foot kicks aggressively up and over. | ![CondMDI HumanML3D 001840 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/condmdi/condmdi_humanml3d_001840_smpl_mesh_512_30fps.gif) |
| `006944` | the person who does arms straight out and then it's doing something with their right hand in front of their face. | ![CondMDI HumanML3D 006944 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/condmdi/condmdi_humanml3d_006944_smpl_mesh_512_30fps.gif) |

512px / 30fps GIF previews rendered from released HumanML3D test outputs.

## Release Snapshot

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

The Hugging Face artifact is self-contained apart from the frozen OpenAI CLIP
text encoder. It contains SafeTensors weights, the exact network and diffusion
configuration, and the official absolute-root normalization statistics. No
upstream source checkout or dataset directory is needed at runtime.

For offline inference, set `MOTIUS_CLIP_PATH` to a local OpenAI CLIP ViT-B/32
checkpoint. `MOTIUS_CLIP_CACHE` can instead redirect the normal CLIP download
cache.

## Usage

Install the method-specific dependencies:

```bash
pip install -e ".[condmdi]"
```

Text-to-motion generation:

```python
from motius.pipelines.condmdi import CondMDIPipeline

pipe = CondMDIPipeline.from_pretrained(
    "ZeyuLing/motius-condmdi-humanml3d",
    bundle_kwargs={"respacing": "ddim100"},
    device="cuda",
)

motions = pipe.infer_t2m(
    ["a person walks forward and waves with the right hand"],
    [120],
    seed=42,
)
```

First-and-last-frame in-betweening uses a standard HML263 reference motion:

```python
controlled = pipe.infer_control(
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
| MotionStreamer Evaluator | 4,032 | 0.453 | 0.611 | 0.702 | 121.837 | 19.970 | 25.464 |
| Motius Joint-Position Evaluator | 4,034 | 0.430 | 0.604 | 0.702 | 0.1919 | 39.127 | 55.795 |

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
| Start 1f | Motius Joint-Position | 0.492 | 0.661 | 0.751 | 107.142 | 34.124 | 55.393 |
| Both 1f | MotionStreamer | 0.568 | 0.730 | 0.801 | 54.043 | 18.186 | 26.787 |
| Both 1f | Motius Joint-Position | 0.561 | 0.734 | 0.814 | 56.623 | 31.615 | 54.927 |
| Prefix 20 | MotionStreamer | 0.402 | 0.536 | 0.596 | 166.292 | 21.075 | 24.323 |
| Prefix 20 | Motius Joint-Position | 0.374 | 0.518 | 0.600 | 428.528 | 40.855 | 51.799 |
| Middle 80 | MotionStreamer | 0.484 | 0.628 | 0.707 | 123.567 | 19.812 | 25.010 |
| Middle 80 | Motius Joint-Position | 0.466 | 0.622 | 0.706 | 269.269 | 36.836 | 52.746 |

The following reconstruction and physical diagnostics are computed on the same
4,012 cases after conversion to the shared SMPL-22 skeleton. MPJPE and P-MPJPE
are in meters; lower is better for every column.

| Setting | Full MPJPE | Generated-region MPJPE | P-MPJPE | Jitter | Foot skating |
| ------- | ----------: | ---------------------: | -------: | -----: | -----------: |
| Start 1f | 0.1339 | 0.1345 | 0.0126 | 46.206 | 0.1601 |
| Both 1f | 0.1134 | 0.1144 | 0.0206 | 49.102 | 0.1829 |
| Prefix 20 | 0.1007 | 0.1235 | 0.0105 | 25.850 | 0.0726 |
| Middle 80 | 0.0945 | 0.1138 | 0.0189 | 34.526 | 0.1240 |

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

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.condmdi.CondMDIPipeline` |
| Bundle | `motius.models.condmdi.CondMDIBundle` |
| UNet and diffusion runtime | `motius.models.condmdi.network` |
| HumanML3D selected-caption runner | `tools/eval_condmdi_humanml3d.py` |
| Official checkpoint exporter | `tools/export_condmdi_hf.py` |

The vendored method runtime retains the upstream MIT license in
`motius/models/condmdi/LICENSE`.

## Reproduction Check

The migrated network was checked against the official implementation using the
same checkpoint, text embedding, input tensor, and diffusion timestep. A single
UNet forward pass differs by at most `1.41e-5` (`6.45e-7` mean absolute error).
For a complete 100-step fp16 sample, accumulated mean absolute error is
`8.62e-4` (`1.59e-2` maximum).

## Citation

```bibtex
@inproceedings{cohan2024flexible,
  title={Flexible Motion In-betweening with Diffusion Models},
  author={Cohan, Setareh and Tevet, Guy and Reda, Daniele and Peng, Xue Bin and van de Panne, Michiel},
  booktitle={ACM SIGGRAPH 2024 Conference Proceedings},
  year={2024}
}
```
