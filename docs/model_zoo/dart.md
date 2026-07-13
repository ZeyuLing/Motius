<h1 align="center">DART Model Card</h1>

<p align="center">
  <strong>DartControl packaged as a Motius autoregressive text-to-motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.05260">Paper</a> |
  <a href="https://zkf1997.github.io/DART/">Project Page</a> |
  <a href="https://github.com/zkf1997/DART">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-dart-humanml3d">Motius Checkpoint</a>
</p>

DART, short for DartControl, is a diffusion-based autoregressive motion model
for real-time text-driven motion control. This Motius package exposes the
HumanML3D DART276 rollout path through a `ModelBundle` and task-facing pipeline.

## Preview

| HumanML3D Sample | Input Text | SMPL Preview |
| ---------------- | ---------- | ------------ |
| `001840` | someone executes a roundhouse kick with their left foot. | ![DART HumanML3D 001840 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/dart/dart_humanml3d_001840_smpl_mesh_512_30fps.gif) |
| `004545` | a person jumping while raising both hands and moving apart legs. | ![DART HumanML3D 004545 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/dart/dart_humanml3d_004545_smpl_mesh_512_30fps.gif) |
| `006944` | a person moves their right hand left, right, up, and down. | ![DART HumanML3D 006944 SMPL demo](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/dart/dart_humanml3d_006944_smpl_mesh_512_30fps.gif) |

512px / 30fps GIF previews rendered from released HumanML3D test outputs.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | DART / DartControl |
| Tasks | T2M, Motion Control |
| Venue | ICLR 2025 |
| Motion representation | DART276, 20 fps |
| Backbone | Motion primitive VAE + latent diffusion denoiser |
| Default guidance scale | `5.0` |
| Checkpoint | [`ZeyuLing/motius-dart-humanml3d`](https://huggingface.co/ZeyuLing/motius-dart-humanml3d) |
| Pipeline | `motius.pipelines.dart.DARTPipeline` |

The checkpoint artifact contains the DART denoiser, MVAE, runtime configuration, seed motion, and DART276 normalization/text-embedding assets. It intentionally does not include license-controlled SMPL-H or SMPL-X body model files; install those locally under `checkpoints/smpl_models` or set `MOTIUS_BODY_MODEL_DIR` before full rollout or SMPL export.

## Usage

Install Motius:

```bash
python -m pip install -e ".[dev]"
```

Run the SMPL-sequence export adapter after installing the licensed body-model assets locally:

```python
from motius.pipelines.dart import DARTPipeline

pipe = DARTPipeline.from_pretrained(
    "ZeyuLing/motius-dart-humanml3d",
    device="cuda",
)

smpl_sequences = pipe.infer_t2m_smpl(
    ["a person walks forward, then turns left"],
    [196],
    seed=0,
)
```

`infer_t2m_smpl` is an adapter for rendering/evaluation conversion. DART remains native `DART276`; SMPL and `motion_135` tensors are export adapters, not separate checkpoint representations.

## Evaluation Results

Protocol: HumanML3D Official uses the selected-caption HumanML3D test protocol. MotionStreamer Evaluator and Motius Joint-Position Evaluator are computed after converting outputs through the shared SMPL-22 evaluation bridge. For FID and MM-Dist, lower is better.

| Evaluator | Variant | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | Default | 3,970 | 0.401 | 0.592 | 0.700 | 1.846 | 3.709 | 9.867 | Measured |
| MotionStreamer Evaluator | Default | 4,042 | 0.548 | 0.725 | 0.794 | 127.830 | 18.531 | 26.261 | Measured |
| Motius Joint-Position Evaluator | Default | 4,034 | 0.425 | 0.606 | 0.702 | 371.131 | 38.764 | 56.949 | Measured |


## Motion Representation

DART uses a 276-dimensional motion-primitive representation (`DART276`) in the
vendored runtime. Helper adapters may export SMPL-style tensors for rendering
or evaluator conversion, but those adapters are not the model's native motion
representation and are not a published checkpoint variant.

The underlying runtime uses motion primitives and an autoregressive rollout
loop. The public checkpoint documents this tensor contract in `model_index.json`.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius.pipelines.dart.DARTPipeline` |
| Bundle | `motius.models.dart.DARTBundle` |
| Runtime | `motius.models.dart.network` |
| PyTorch3D shim | `motius.models.dart.network.pytorch3d.transforms` |

## Citation

```bibtex
@inproceedings{Zhao:DartControl:2025,
  title={{DartControl}: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control},
  author={Zhao, Kaifeng and Li, Gen and Tang, Siyu},
  booktitle={The Thirteenth International Conference on Learning Representations},
  year={2025}
}
```
