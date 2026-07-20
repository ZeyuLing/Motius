# Coverage Audit: 2024-2026 LLM-based evaluation metrics for open-ended text generation and caption quality

## Source Ledger Summary

| Source family | Runs | Raw hits | Unique hits | Core papers | Status |
|---|---:|---:|---:|---:|---|
| arXiv open-ended judge benchmarks | 2 | 8 | 8 | 6 | complete |
| ACL/CVF caption metrics | 1 | 8 | 6 | 3 | complete |
| ECCV/NeurIPS motion captioning | 1 | 4 | 2 | 2 | complete |

## Snowballing

| Anchor | Backward new core | Forward new core | Passes | Notes |
|---|---:|---:|---:|---|
| MT-Bench | 1 | 3 | 1 | Led to AlpacaEval LC, Arena-Hard and JudgeBench |
| CLAIR | 1 | 3 | 1 | Led to FLEUR and modality-grounded caption evaluation |

## Blind Spots

- No established motion-caption judge benchmark currently provides human labels.
- A video render can hide joint-level errors and introduces renderer bias.
- Proprietary judges can drift; an open judge must be version-pinned and calibrated.

## Stopping Decision

Ready for the requested protocol decision. Coverage is focused rather than an
exhaustive survey of every general-purpose judge model.
