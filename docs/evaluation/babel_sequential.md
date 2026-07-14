# BABEL Sequential Generation Evaluation

The BABEL Sequential Generation leaderboard evaluates one continuous motion
conditioned on an ordered list of action captions. It uses FlowMDM's public
BABEL validation composition file: 64 compositions with 32 prompts each.

## Protocol

All submitted motions are converted to 30 fps canonical SMPL-22 joints66.
Each captioned interval is independently canonicalized to first-pelvis XZ
origin, first-frame body facing +Z, and floor height zero before semantic
embedding. Transition windows use the same canonicalization.

| Group | Metrics | Reference |
| ----- | ------- | --------- |
| Semantic subsequences | R@1/2/3, MM-Dist, FID, Diversity | Deterministic 2,048-action sample from BABEL val frame annotations |
| 30-frame transitions | Transition FID, Diversity, Peak Jerk, AUJ gap | Deterministic 2,048-window sample from BABEL val motions |

Retrieval and embedding metrics use
[`ZeyuLing/motius-evaluator-universal-smplh-joints66`](https://huggingface.co/ZeyuLing/motius-evaluator-universal-smplh-joints66).
Diversity and absolute Peak Jerk are diagnostic statistics, not ranked quality
objectives. GT/reference rows do not participate in best/second-best styling.

## Data Layout

Download BABEL motions and annotations under its research license, then place
the files as follows:

```text
data/babel/
├── babel-smplh-30fps-male/val.pth.tar
├── babel-teach/val.json
└── flowmdm_eval_protocol/dataset/babel_val_set.json
```

The composition JSON comes from the
[FlowMDM evaluation protocol](https://github.com/BarqueroGerman/FlowMDM).

## Reproduce FlowMDM

```bash
python tools/build_babel_sequential_manifest.py \
  --output-root outputs/evaluation/babel_sequential/protocol_v2

python tools/generate_babel_sequential.py \
  --manifest outputs/evaluation/babel_sequential/protocol_v2/manifest.json \
  --model ZeyuLing/motius-flowmdm-babel \
  --output-dir outputs/evaluation/babel_sequential/flowmdm_official_seed42 \
  --device cuda --seed 42

python tools/eval_babel_sequential.py \
  --manifest outputs/evaluation/babel_sequential/protocol_v2/manifest.json \
  --predictions-dir outputs/evaluation/babel_sequential/flowmdm_official_seed42/joints66 \
  --method FlowMDM \
  --output outputs/evaluation/babel_sequential/flowmdm_official_seed42/metrics_motius_joint_evaluator.json \
  --device cuda --batch-size 512 --chunk-size 32 --n-repeats 1
```

Both generation and evaluation accept deterministic sharding for cluster runs.
Generated artifacts and metrics must remain under `outputs/`.

## Submission Contract

For another sequential method, write one `joints66/{case_id}.npy` file per
manifest case. Each array must have shape `(T, 66)`, 30 fps, and cover every
half-open segment interval listed in the manifest. Evaluation never imports
the method's original repository at runtime.
