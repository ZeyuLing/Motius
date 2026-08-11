# Text-to-Multi-Person Motion · InterHuman

<p align="center">
  <a href="README.md">📊 Benchmark Hub</a> ·
  <a href="../evaluator_zoo/interclip.md">📐 InterCLIP Evaluator</a> ·
  <a href="../tasks/README.md">🧭 Task Registry</a>
</p>

This leaderboard evaluates caption-conditioned generation of two synchronized
people in one shared world frame.

## Fixed Protocol

| Field | Contract |
| ----- | -------- |
| Task | Text-to-Multi-Person Motion |
| Dataset | InterHuman official test split |
| Actor layout | `(B, T, 2, 262)` paired InterHuman-262 |
| Evaluator | [InterCLIP](../evaluator_zoo/interclip.md) |
| Retrieval protocol | Batch 96 · 20 official repeats |
| Metrics | R@1 · R@2 · R@3 · normalized FID · MM-Dist · Diversity |
| Included methods | [InterGen](../model_zoo/intergen.md) · [InterMask](../model_zoo/intermask.md) |

Both actors must retain the dataset's shared coordinate frame. Per-actor root
alignment, actor swapping after inference, or independent single-person
evaluation changes the task and is not allowed.

## Leaderboard

| Method | Samples | R@1 ↑ | R@2 ↑ | R@3 ↑ | Normalized FID ↓ | MM-Dist ↓ | Diversity |
| ------ | ------: | ----: | ----: | ----: | ----: | --------: | --------: |
| GT | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| InterGen | Pending | Pending | Pending | Pending | Pending | Pending | Pending |
| InterMask | Pending | Pending | Pending | Pending | Pending | Pending | Pending |

Scores remain pending until complete official-test prediction packs and the
persisted evaluator report are available.
