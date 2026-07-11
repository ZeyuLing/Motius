<p align="center">
  <img src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/brand/motius-logo-readme.png" width="620" alt="Motius logo">
</p>

<h1 align="center">Motius</h1>

<p align="center">
  <strong>A modular training, evaluation, and inference framework for human motion generation.</strong>
</p>

<p align="center">
  <a href="#model-zoo">Model Zoo</a> |
  <a href="docs/getting_started.md">Getting Started</a> |
  <a href="docs/architecture.md">Architecture</a> |
  <a href="docs/development.md">Development Guide</a>
</p>

Motius packages motion-generation methods as consistent model bundles,
trainers, pipelines, evaluators, and visualization utilities. The public repo
is being opened method by method: the reusable core is available now, and each
released method will ship with a model card, checkpoint path, evaluation
results, and qualitative SMPL renders.

## Model Zoo

<table>
  <thead>
    <tr>
      <th align="left" width="24%">Method</th>
      <th align="left" width="44%">SMPL Preview</th>
      <th align="left" width="32%">Release</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>
        <strong>MDM</strong><br>
        Human Motion Diffusion Model<br>
        <sub>Text-to-Motion, HumanML3D-263, 20 fps</sub>
      </td>
      <td>
        <video src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/mdm/mdm_humanml3d_000708_smpl_mesh.mp4" controls muted loop width="100%"></video>
      </td>
      <td>
        <a href="docs/model_zoo/mdm.md"><strong>Model Card</strong></a><br>
        <a href="https://huggingface.co/ZeyuLing/hftrainer-mdm-humanml3d">Checkpoint</a><br>
        <a href="https://arxiv.org/abs/2209.14916">Paper</a> |
        <a href="https://github.com/GuyTevet/motion-diffusion-model">Original GitHub</a>
      </td>
    </tr>
  </tbody>
</table>

## What Is Included

| Area | Purpose |
| ---- | ------- |
| `motius.registry` | Central registries for models, bundles, trainers, pipelines, datasets, hooks, evaluators, and visualizers. |
| `motius.models` | `ModelBundle` abstraction and model utility functions. |
| `motius.trainers` | Reusable trainer base classes for method-specific training logic. |
| `motius.pipelines` | Pipeline base classes for inference and task-facing APIs. |
| `motius.runner` | Accelerate-based distributed training runner and train loops. |
| `motius.datasets` | Dataset bases and reusable transform primitives. |
| `motius.hooks` | Checkpoint, EMA, logging, and learning-rate scheduler hooks. |
| `motius.evaluation` | Evaluator base interfaces. |
| `motius.visualization` | File and TensorBoard visualization bases. |
| `configs/_base_` | Minimal runtime config templates. |
| `tools/` | Command-line training entry points. |

## Quick Start

```bash
python -m pip install -e ".[dev]"
```

Run a lightweight import and registration check:

```bash
python - <<'PY'
import motius

motius.register_all_modules()
print("Motius core import OK")
PY
```

Run the current smoke tests:

```bash
pytest -q
```

## Documentation

The detailed architecture, extension points, and package conventions live in
the formal documentation:

- [Architecture](docs/architecture.md)
- [Getting Started](docs/getting_started.md)
- [Development Guide](docs/development.md)

## Release Status

Motius is an early public release. APIs may still change while research-specific
method code is separated from reusable framework code. New methods will be
added through scoped Model Zoo entries and model cards.
