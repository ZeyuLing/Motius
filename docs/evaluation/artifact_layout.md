# Evaluation Artifact Layout

Every benchmark run uses one canonical root:

```text
outputs/evaluation/<task_id>/<benchmark_id>/<protocol_id>/
├── protocol/
│   ├── protocol.json
│   ├── manifest.jsonl
│   └── references/<representation>/<sample_id>.*
├── runs/<method_id>/<run_id>/
│   ├── run.json
│   ├── predictions/<representation>/<sample_id>.*
│   ├── metrics/
│   │   ├── summary.json
│   │   └── per_sample.jsonl
│   ├── visualization/manifest.json
│   └── logs/
└── leaderboard/results.json
```

The three IDs above the run directory identify the scientific contract:

| ID | Meaning |
| -- | ------- |
| `task_id` | Canonical task from `docs/tasks/taxonomy.json` |
| `benchmark_id` | Dataset or benchmark setting from the same registry |
| `protocol_id` | Immutable split, condition-selection, representation, and evaluator version |
| `method_id` | Stable method slug, independent of checkpoint |
| `run_id` | Checkpoint, seed, or release-specific run slug |

`protocol/manifest.jsonl` fixes sample order and selected conditions. A run may
contain multiple prediction representations, but `run.json` must identify the
one consumed by each metric file. A leaderboard row must point to a run under
the same protocol root; files from different protocol IDs are never ranked
together.

## CLI

Initialize and validate directories through the shared helper:

```bash
python tools/evaluation_artifacts.py \
  --task text_to_motion \
  --benchmark text_to_motion_unitree_g1 \
  --protocol unitree-g1-paper-eval-1024-v1 \
  init-protocol \
  --meta evaluator=ZeyuLing/motius-evaluator-g1-38d-tmr

python tools/evaluation_artifacts.py \
  --task text_to_motion \
  --benchmark text_to_motion_unitree_g1 \
  --protocol unitree-g1-paper-eval-1024-v1 \
  init-run \
  --method hymotion-g1 \
  --run iter-20000-seed-20260707

python tools/evaluation_artifacts.py \
  --task text_to_motion \
  --benchmark text_to_motion_unitree_g1 \
  --protocol unitree-g1-paper-eval-1024-v1 \
  validate --require-manifest
```

The canonical root for every registered benchmark is machine-readable in
`docs/tasks/taxonomy.json`. Legacy experiment folders remain read-only
provenance; new generation, metric, and visualization jobs must write through
this layout.
