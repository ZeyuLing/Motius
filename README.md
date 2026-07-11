<p align="center">
  <img src="https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/brand/motius-logo-readme.png" width="640" alt="Motius logo">
</p>

# Motius

Motius is a modular framework for training, evaluating, and serving human
motion generation models. It provides the shared runtime layer used across
motion model research: registries, model bundles, trainers, pipelines, hooks,
distributed runners, dataset transforms, evaluators, and visualization bases.

This repository is being opened incrementally. The current public release
contains the reusable core framework. Method implementations, model cards,
checkpoints, datasets, and benchmark reports will be reviewed and added in
separate commits.

## Framework At A Glance

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

## Model Zoo

| Method | Task | Model Card |
| ------ | ---- | ---------- |
| MDM | Text-to-Motion | [MDM](docs/model_zoo/mdm.md) |

The detailed architecture, extension points, and package conventions live in
the formal documentation:

- [Architecture](docs/architecture.md)
- [Getting Started](docs/getting_started.md)
- [Development Guide](docs/development.md)

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

## Development Status

This is an early public core drop. APIs may still change while we separate
research-specific method code from reusable framework code.
