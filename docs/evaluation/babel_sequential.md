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

## Measured Baseline

| Method | R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Transition FID | Transition Diversity | Peak Jerk | AUJ Gap |
| ------ | --: | --: | --: | --: | ------: | --------: | -------------: | -------------------: | --------: | ------: |
| BABEL GT | 0.2939 | 0.4330 | 0.5193 | 0.0000 | 46.5581 | 57.4818 | 0.0000 | 54.5835 | 56.34 | 0.0000 |
| FlowMDM | 0.1032 | 0.1839 | 0.2496 | 2479.8395 | 57.3074 | 32.9413 | 2629.5964 | 31.1648 | 463.92 | 55.8724 |

This is a single deterministic seed-42 generation and one retrieval repeat.
R-Precision uses 32-sample recall batches, covering 7,264 of the 7,285 paired
segments. Distribution metrics use the full set.

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
  --output-root outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1

python tools/generate_babel_sequential.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/manifest.json \
  --model ZeyuLing/motius-flowmdm-babel \
  --output-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42 \
  --device cuda --seed 42

python tools/eval_babel_sequential.py \
  --manifest outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/manifest.json \
  --predictions-dir outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42/joints66 \
  --method FlowMDM \
  --output outputs/evaluation/babel_sequential/official_val_shortmerge30_llm_v1/flowmdm_seed42/metrics.json \
  --device cuda --batch-size 32 --chunk-size 32 --n-repeats 1
```

Both generation and evaluation accept deterministic sharding for cluster runs.
Generated artifacts and metrics must remain under `outputs/`.

## Submission Contract

For another sequential method, write one `joints66/{case_id}.npy` file per
manifest case. Each array must have shape `(T, 66)`, 30 fps, and cover every
half-open segment interval listed in the manifest. Evaluation never imports
the method's original repository at runtime.
