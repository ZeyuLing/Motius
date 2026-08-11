# Motion Editing · MotionFix Instructions

[Open the interactive leaderboard](https://huggingface.co/spaces/ZeyuLing/instruction-editing-leaderboard).

MotionFix instruction editing evaluates a supplied source motion and a free-form edit instruction against the paired target motion. All methods are converted to the same nominal-30fps HML263 representation and scored by MotionLab's released text_mot_match evaluator.

The public ranking uses fixed Batch-32 target-motion R@3. Source input and protocol-excluded runs remain visible as unranked diagnostics.

| Rank | Method | Native input | Coverage | FID ↓ | R@1₃₂ ↑ | R@2₃₂ ↑ | R@3₃₂ ↑ | AvgR ↓ | M2M-Dist ↓ | Diversity → | Status |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1* | MotionCanvas | source motion + instruction | 1013/1013 | 0.0327 | 80.1579 | 92.7937 | 95.9526 | 1.4274 | 0.7901 | 8.2819 | contaminated |
| 3 | SimMotionEdit | source motion + instruction | 1013/1013 | 0.4999 | 40.4738 | 59.5262 | 69.7927 | 4.0276 | 2.4369 | 7.8199 | complete |
| 4 | MotionFix (TMED) | source motion + instruction | 1013/1013 | 0.5282 | 32.4778 | 48.7660 | 59.8223 | 5.3593 | 3.1974 | 7.8870 | complete |
| 2 | MotionLab | source motion + instruction | 1013/1013 | 0.1339 | 46.1007 | 63.0800 | 73.1491 | 3.7532 | 2.4112 | 8.1804 | complete |
| — | Source input (no edit) | source motion only | 1013/1013 | 0.1066 | 46.3968 | 65.2517 | 75.0247 | 3.5054 | 2.2817 | 8.1981 | complete |

## Protocol

- Dataset: MotionFix official test, 1,013 source/instruction/target triplets.
- Evaluator: MotionLab released text_mot_match motion encoder.
- Ranking metric: fixed Batch-32 target-motion R@3.
- Candidate batches follow target-length ascending order, matching the released dataset.
- FID, M2M-Dist, and Diversity use the same native motion embeddings.
- MotionCanvas is currently unranked: its training annotation contains 1,000/1,013 test pairs with identical source and target motion content.

## Canonical Artifacts

`outputs/evaluation/motion_edit/instruction_editing/motionfix_test/smplh_joints66/{method}/metrics/leaderboard.json`

Last updated: 2026-07-22.
