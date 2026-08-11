# Motion Editing · Style and Content

[Open the interactive leaderboard](https://huggingface.co/spaces/ZeyuLing/motion-edit-leaderboard).

Style and Content editing use the same ten-metric protocol. Every ranked method receives the exact source motion and edit instruction. MotionCLR is retained as an unranked text-generated style diagnostic.

## Style Editing

Change style while preserving the source action. 63 held-out cases.

| Rank | Method | Coverage | S-Acc ↑ | A-Acc ↑ | FID ↓ | MM-Dist ↓ | T2M R@3 ↑ | M2M R@3 ↑ | M2M-Dist ↓ | Diversity → | Foot ↓ | Jitter ↓ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REF | Source input | 63/63 | 23.97 | 97.30 | 0.1205 | 18.55 | 66.72 | 42.86 | 0.4791 | 42.46 | 2.69 | 22.35 |
| 5 | TMED | 63/63 | 23.17 | 46.03 | 1.0784 | 54.74 | 22.97 | 14.29 | 1.2755 | 56.58 | 9.93 | 42.97 |
| 4 | SimMotionEdit | 63/63 | 16.59 | 71.11 | 0.9794 | 49.46 | 37.34 | 19.05 | 1.1696 | 55.68 | 11.19 | 11.11 |
| 2 | MotionLab | 63/63 | 26.67 | 94.52 | 0.2096 | 24.75 | 58.44 | 38.10 | 0.6141 | 47.97 | 2.79 | 7.54 |
| 3 | MotionReFit | 63/63 | 26.75 | 95.32 | 0.1233 | 18.78 | 64.69 | 36.51 | 0.4818 | 41.70 | 3.87 | 20.05 |
| 1 | MotionCanvas | 63/63 | 40.79 | 92.54 | 0.0657 | 18.82 | 65.00 | 68.25 | 0.3781 | 39.90 | 3.22 | 14.52 |
| DIAG | MotionCLR | 63/63 | 15.24 | 79.92 | 0.7161 | 43.12 | 47.50 | 23.81 | 1.0470 | 55.50 | 4.27 | 10.30 |

## Content Editing

Change action while preserving the source style. 67 held-out cases.

| Rank | Method | Coverage | S-Acc ↑ | A-Acc ↑ | FID ↓ | MM-Dist ↓ | T2M R@3 ↑ | M2M R@3 ↑ | M2M-Dist ↓ | Diversity → | Foot ↓ | Jitter ↓ |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| REF | Source input | 67/67 | 52.01 | 19.25 | 0.4850 | 40.07 | 8.44 | 8.96 | 0.9507 | 27.03 | 3.76 | 23.67 |
| 3 | TMED | 67/67 | 25.90 | 29.48 | 1.1329 | 56.31 | 14.14 | 4.48 | 1.3319 | 55.23 | 12.03 | 79.53 |
| 5 | SimMotionEdit | 67/67 | 39.03 | 18.36 | 1.2423 | 58.57 | 8.05 | 2.99 | 1.3688 | 54.54 | 8.15 | 10.41 |
| 3 | MotionLab | 67/67 | 49.10 | 18.51 | 0.6337 | 46.93 | 11.02 | 4.48 | 1.1208 | 42.81 | 2.99 | 7.56 |
| 2 | MotionReFit | 67/67 | 36.12 | 32.61 | 0.4978 | 41.23 | 15.55 | 10.45 | 0.9986 | 40.83 | 3.72 | 30.76 |
| 1 | MotionCanvas | 67/67 | 48.21 | 96.42 | 0.0523 | 18.75 | 64.77 | 68.66 | 0.3796 | 40.83 | 2.65 | 11.28 |
