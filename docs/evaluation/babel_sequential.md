# BABEL Sequential Generation Evaluation

The BABEL Sequential Generation leaderboard evaluates one continuous motion
conditioned on an ordered list of action captions. The protocol uses every
eligible episode in the processed official BABEL validation split. It contains
1,295 episodes, 7,285 captioned action intervals, and 5,990 transitions.

## Protocol

Explicit BABEL transition labels are removed and neighboring actions are cut at
the transition midpoint. Adjacent short actions are greedily merged until each
conditioned interval contains at least 30 frames. Every resulting caption is
rewritten by the supplied precomputed LLM rewrite cache. Raw `proc_label` text
is not used as the evaluation caption.

All reference and generated motions are converted to 30 fps, neutral zero-beta
SMPL-22 joints66. Each captioned interval is independently canonicalized to
first-pelvis XZ origin, first-frame body facing +Z, and floor height zero before
semantic embedding. Transition windows use the same canonicalization.

| Group | Metrics | Reference |
| ----- | ------- | --------- |
| Semantic subsequences | R@1/2/3, MM-Dist, FID, Diversity | The 7,285 paired GT intervals from the same 1,295 episodes |
| 30-frame transitions | Transition FID, Diversity, Peak Jerk, AUJ gap | The 5,990 paired GT transition windows at the same boundaries |

Retrieval and embedding metrics use
[`ZeyuLing/motius-evaluator-universal-smplh-joints66`](https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66).
Diversity and absolute Peak Jerk are diagnostic statistics, not ranked quality
objectives. GT/reference rows do not participate in best/second-best styling.

BABEL captions are not unique. After case-folding and punctuation/whitespace
normalization, the 7,285 intervals contain 2,924 caption groups; 5,149
intervals belong to one of 788 repeated groups. The most frequent caption,
`A person stands.`, occurs 676 times. R-Precision therefore uses
caption-group-aware multi-positive retrieval: every candidate with the same
normalized caption is a valid positive within its 32-sample recall batch.
No interval is removed. MM-Dist uses the nearest positive, while FID and
Diversity continue to use all 7,285 intervals.

## Measured Baseline

| Method | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Transition FID | Transition Diversity | Peak Jerk | AUJ Gap |
| ------ | --: | --: | --: | --: | ------: | --------: | -------------: | -------------------: | --------: | ------: |
| BABEL GT | 0.3619 | 0.5131 | 0.5936 | 0.0000 | 45.2346 | 57.4816 | 0.0000 | 54.5830 | 56.34 | 0.0000 |
| FlowMDM | 0.2706 | 0.3922 | 0.4708 | 160.3988 | 47.4209 | 56.5743 | 205.8370 | 54.7209 | 335.67 | 34.4040 |

This is a single deterministic seed-42 generation and one retrieval repeat.
R-Precision uses 32-sample recall batches, covering 7,264 of the 7,285 paired
segments, and accepts every same-caption candidate as a positive. Distribution
metrics use the full set. `--chunk-size 32` controls the recall candidate set;
`--batch-size 32` controls only evaluator encoding throughput in this run.

Open the [Three.js sequence audit](../../assets/evaluation/babel_sequential_demo/index.html)
to compare BABEL GT and FlowMDM frame by frame. Every subclip has a fixed color,
and the synchronized caption list exposes its exact half-open frame interval.
Each row also reports the nearest three texts for the GT and generated motion,
plus the exact text-to-motion positive rank in the seed-0, 32-candidate batch
used by the leaderboard.

## Data Layout

Download BABEL motions and annotations under its research license, then place
the files as follows:

```text
data/babel/processed/
├── manifests/val.jsonl
├── ms272/val/{episode_id}.npz
└── babel_shortmerge_caption_rewrites.json

checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl
```

The JSONL must be produced from the official BABEL validation annotations. The
MS272 files may contain either the full source motion or the already clipped
episode span. The rewrite cache is keyed by the merged source-label sequence.

## Reproduce FlowMDM

```bash
python tools/build_babel_sequential_manifest.py \
  --processed-manifest data/babel/processed/manifests/val.jsonl \
  --motion272-dir data/babel/processed/ms272/val \
  --rewrite-cache data/babel/processed/babel_shortmerge_caption_rewrites.json \
  --smpl-model checkpoints/body_models/smpl/SMPL_NEUTRAL.pkl \
  --output-root outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2

python tools/generate_babel_sequential.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/manifest.json \
  --model ZeyuLing/motius-flowmdm-babel \
  --output-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/flowmdm_seed42 \
  --device cuda --seed 42

python tools/eval_babel_sequential.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/manifest.json \
  --predictions-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/flowmdm_seed42/joints66 \
  --method FlowMDM \
  --output outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/flowmdm_seed42/metrics.json \
  --device cuda --batch-size 32 --chunk-size 32 --n-repeats 1

python tools/export_babel_retrieval_audit.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/manifest.json \
  --predictions-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/flowmdm_seed42/joints66 \
  --output outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/flowmdm_seed42/retrieval_audit.json \
  --device cuda --batch-size 128 --chunk-size 32 --top-k 3 --seed 0

python tools/build_babel_sequential_viewer.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/manifest.json \
  --predictions-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/flowmdm_seed42/joints66 \
  --retrieval-audit outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_multipositive_v2/flowmdm_seed42/retrieval_audit.json \
  --output-dir outputs/visualization/babel_sequential_audit
```

Both generation and evaluation accept deterministic sharding for cluster runs.
Generated artifacts and metrics must remain under `outputs/`.

## Submission Contract

For another sequential method, write one `joints66/{case_id}.npy` file per
manifest case. Each array must have shape `(T, 66)`, 30 fps, and cover every
half-open segment interval listed in the manifest. Evaluation never imports
the method's original repository at runtime.
