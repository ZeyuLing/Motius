# Motius Architecture

Motius is organized around small framework contracts. A method package should
only implement the pieces it owns, then register them with the shared runtime.

## Repository Layout

```text
Motius/
├── motius/
│   ├── registry.py
│   ├── models/
│   ├── trainers/
│   ├── pipelines/
│   ├── runner/
│   ├── datasets/
│   ├── hooks/
│   ├── evaluation/
│   ├── visualization/
│   └── utils/
├── configs/
│   └── _base_/
├── tools/
├── tests/
└── docs/
```

## Core Concepts

### Registry

`motius.registry` defines the shared registries used by the framework:

| Registry | Expected content |
| -------- | ---------------- |
| `HF_MODELS` / `MODELS` | Model classes and Hugging Face compatible builders. |
| `MODEL_BUNDLES` | `ModelBundle` subclasses. |
| `TRAINERS` | Trainer classes. |
| `PIPELINES` | Inference or task-facing pipeline classes. |
| `DATASETS` | Dataset classes. |
| `TRANSFORMS` | MMEngine-compatible transforms. |
| `HOOKS` | Runner hooks. |
| `EVALUATORS` | Evaluation interfaces and metric runners. |
| `VISUALIZERS` | Visualization backends. |

### Model Bundle

`motius.models.ModelBundle` is the model ownership boundary. It groups the
modules, buffers, checkpoint metadata, and export logic that belong to one
trainable method. Trainers and pipelines should talk to bundles instead of
directly reaching into unrelated module internals.

### Trainer

`motius.trainers.BaseTrainer` defines the method-specific training step
contract. A trainer receives model bundles and batches from the runner, computes
losses, and returns structured outputs that hooks and loggers can consume.

### Pipeline

`motius.pipelines.BasePipeline` is the inference boundary. Method pipelines
should live under `motius/pipelines/{method_name}/` once opened, and should
expose stable task APIs rather than training-only internals.

### Runner

`motius.runner.AccelerateRunner` owns distributed setup, dataloader building,
loop execution, hooks, checkpoint IO, and train/eval orchestration. It is the
default runtime behind `tools/train.py`.

### Hooks

Hooks are small lifecycle extensions registered through `HOOKS`. The core
release includes checkpointing, EMA, logging, and learning-rate scheduling.

## Method Package Convention

New method code should be grouped by method name:

```text
motius/
├── models/{method_name}/
├── trainers/{method_name}/
├── pipelines/{method_name}/
└── evaluation/{method_name}/
```

Shared helpers may live in clearly named utility modules, but method-specific
implementations should not be hidden under generic wrapper directories.

## Outputs

Runtime outputs must be written under `outputs/` or a configured work directory
inside `outputs/`. Public code and docs should not assume that generated
checkpoints, logs, visualizations, or evaluation tables are written to the
repository root.

## Minimal Component Flow

```text
config
  -> registry builders
  -> ModelBundle
  -> BaseTrainer
  -> AccelerateRunner
  -> hooks, checkpoints, logs
  -> BasePipeline or evaluator
```

This flow keeps method code replaceable while preserving one common runtime.
