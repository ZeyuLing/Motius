<h1 align="center">DART Model Card</h1>

<p align="center">
  <strong>DartControl packaged as a Motius autoregressive text-to-motion pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.05260">Paper</a> |
  <a href="https://zkf1997.github.io/DART/">Project Page</a> |
  <a href="https://github.com/zkf1997/DART">Original GitHub</a>
</p>

DART, short for DartControl, is a diffusion-based autoregressive motion model
for real-time text-driven motion control. This Motius package exposes the
HumanML3D DART276 rollout path through a `ModelBundle` and task-facing pipeline.

Validated demos will be added after the DART artifact is converted into a
self-contained public checkpoint and its renders are checked.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | DART / DartControl |
| Task | Autoregressive Text-to-Motion and motion control |
| Venue | ICLR 2025 |
| Motion representation | DART276, 20 fps |
| Backbone | Motion primitive VAE + latent diffusion denoiser |
| Default guidance scale | `5.0` |
| Checkpoint | Not released yet; no public Hugging Face artifact is available |
| Pipeline | `motius.pipelines.dart.DARTPipeline` |

The runtime expects a self-contained artifact with denoiser weights, MVAE
weights, seed motion data, text embeddings, and SMPL support files. The current
public code is ready for that artifact layout, but the checkpoint itself has
not been converted, verified, or uploaded.

## Usage

Install Motius:

```bash
python -m pip install -e ".[dev]"
```

Run the SMPL-sequence export adapter after placing a verified DART artifact at
`checkpoints/dart/motius_hml3d`:

```python
from motius.pipelines.dart import DARTPipeline

pipe = DARTPipeline.from_pretrained(
    "checkpoints/dart/motius_hml3d",
    device="cuda",
)

smpl_sequences = pipe.infer_t2m_smpl(
    ["a person walks forward, then turns left"],
    [196],
    seed=0,
)
```

`infer_t2m_smpl` is an adapter for rendering/evaluation conversion. The public
checkpoint artifact must define and validate the DART276 export contract before
this card is marked complete.

## Evaluation Results

Public leaderboard metrics are pending until the self-contained DART checkpoint,
demo render, and evaluation conversion are published.

| Evaluator | Samples | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Status |
| --------- | ------: | --: | --: | --: | --: | ------: | --------: | ------ |
| HumanML3D Official | - | - | - | - | - | - | - | Pending |
| MotionStreamer Evaluator | - | - | - | - | - | - | - | Pending |
| Motius Joint-Position Evaluator | - | - | - | - | - | - | - | Pending |

## Motion Representation

DART uses a 276-dimensional motion-primitive representation (`DART276`) in the
vendored runtime. Helper adapters may export SMPL-style tensors for rendering
or evaluator conversion, but those adapters are not the model's native motion
representation and are not a published checkpoint variant.

The underlying runtime uses motion primitives and an autoregressive rollout
loop. The final public card will document the exact tensor contract after the
self-contained DART checkpoint is converted and verified.

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
