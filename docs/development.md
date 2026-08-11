# Development Guide

## Public Core Policy

Motius is being opened in stages. Core framework changes should be small,
reviewable, and separated from method releases, datasets, checkpoints, and
benchmark artifacts.

## Naming

The public Python package is `motius`. New code, docs, configs, tests, and
examples should use the Motius name consistently.

## Directory Rules

Use method-name directories for method implementations:

```text
motius/models/{method_name}/
motius/trainers/{method_name}/
motius/pipelines/{method_name}/
motius/evaluation/{method_name}/
```

Avoid adding generic wrapper layers that only group methods by broad domain.

## Generated Files

Generated runtime artifacts belong under `outputs/`. Do not write checkpoints,
logs, temporary evaluation files, or visualizations to the repository root.

## Mirror Workflow

Maintainers should push every public repository change to both configured
remotes: the GitHub repository and the internal mirror. Keep the same branch
name and commit SHA on both remotes whenever possible.

## Pre-Push Checks

Before pushing, run:

```bash
python -m compileall -q motius tools
pytest -q
```

Also run the repository naming audit used by maintainers before release.
