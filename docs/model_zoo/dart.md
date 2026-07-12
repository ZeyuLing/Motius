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
HumanML3D SMPL-H rollout path through a `ModelBundle` and task-facing pipeline.

Validated MP4 previews will be added after the DART artifact is cleaned into a
self-contained public checkpoint and its SMPL renders are checked.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Method | DART / DartControl |
| Task | Autoregressive Text-to-Motion and motion control |
| Venue | ICLR 2025 |
| Motion representation | SMPL-H `motion_135` output bridge |
| Backbone | Motion primitive VAE + latent diffusion denoiser |
| Default guidance scale | `5.0` |
| Checkpoint | Pending HF artifact; local artifact layout is `checkpoints/dart/motius_hml3d` |
| Pipeline | `motius.pipelines.dart.DARTPipeline` |

The runtime expects a self-contained artifact with denoiser weights, MVAE
weights, seed motion data, text embeddings, and SMPL support files. The current
public code is ready for that artifact layout, but the checkpoint itself still
needs cleanup before upload.

## Usage

Install Motius:

```bash
python -m pip install -e ".[dev]"
```

Run autoregressive text-to-motion after placing the DART artifact at
`checkpoints/dart/motius_hml3d`:

```python
from motius.pipelines.dart import DARTPipeline

pipe = DARTPipeline.from_pretrained(
    "checkpoints/dart/motius_hml3d",
    device="cuda",
)

motions = pipe.infer_t2m_motion135(
    ["a person walks forward, then turns left"],
    [196],
    seed=0,
)
```

`motions` is a list of NumPy arrays. Each array has shape `(T, 135)` and stores
global translation plus SMPL-H rotations in row-layout 6D form.

## Evaluation Results

Public leaderboard metrics are pending until the self-contained DART checkpoint
and SMPL render/evaluation conversion are published.

## Motion Representation

The pipeline returns the shared SMPL-H `motion_135` bridge:

| Slice | Dim | Meaning |
| ----- | ---: | ------- |
| `transl` | 3 | global root translation |
| `rot6d` | 132 | 22 SMPL-H joints in 6D rotation format |

The underlying DART runtime uses motion primitives and an autoregressive rollout
loop; `DARTPipeline.infer_t2m_smpl(...)` can also return the intermediate SMPL
sequence dictionaries.

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
