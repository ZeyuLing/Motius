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

## Preserve Published Demos

Published demo videos and animated previews are part of the public release.
Preserve them when reorganizing documentation: keep existing media files,
working video links, and visible animated previews in the main README.
Moving a demo to a model card must retain a discoverable homepage preview or
gallery entry. Do not replace an animated preview with a still image as part
of a layout cleanup.

When a corrected render replaces an inaccurate demo, document the reason and
link the replacement. Keep valid older demos accessible; remove media only
for a concrete correctness, licensing, or maintainer-requested reason.
Character demos must link their source models, creators, and license records
from the public documentation. Keep downloadable character assets and
published demo media separate.

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
