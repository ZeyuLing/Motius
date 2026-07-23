<h1 align="center">InterMask Model Card</h1>

<p align="center">
  <strong>Collaborative masked token generation for two-person interaction.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.10010">Paper</a> |
  <a href="https://gohar-malik.github.io/intermask/">Project Page</a> |
  <a href="https://github.com/gohar-malik/InterMask">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-intermask-interhuman">Motius Checkpoint</a>
</p>

InterMask represents each person's motion as a 2D discrete token map and fills
both masked token grids collaboratively with spatial, temporal, and cross-person
attention. This Motius release packages the official InterHuman model.

## Preview

| Input Text | Paired SMPL Preview |
| ---------- | ------------------- |
| two people hug each other and then step back | ![InterMask hug](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/intermask/intermask_interhuman_hug_smpl_pair_512_30fps.gif) |
| one person gently pushes the other person backward | ![InterMask push](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/intermask/intermask_interhuman_gentle_push_smpl_pair_512_30fps.gif) |

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Tasks | Two-Person Text-to-Motion |
| Dataset | InterHuman |
| Representation | `(B, T, 2, 262)` InterHuman-262, 30 fps |
| Tokenizer | Shared 2D RVQ-VAE |
| Generator | Collaborative Inter-M masked Transformer |
| Text encoder | CLIP ViT-L/14@336px text tower, frozen |
| Checkpoint | [`ZeyuLing/motius-intermask-interhuman`](https://huggingface.co/ZeyuLing/motius-intermask-interhuman) |
| Pipeline | `motius.pipelines.intermask.InterMaskPipeline` |
| License | MIT |

## Usage

```bash
pip install -e ".[intermask]"
```

```python
from motius.pipelines.intermask import InterMaskPipeline

pipe = InterMaskPipeline.from_pretrained(
    "ZeyuLing/motius-intermask-interhuman",
    bundle_kwargs={"device": "cuda"},
)
motion = pipe(
    "two people hug each other and then step back",
    motion_len=120,  # multiple of four, 16..300
    seed=42,
)  # (1, 120, 2, 262)
```

`cond_scale`, `time_steps`, `topk_filter_thres`, and `temperature` are exposed
as optional pipeline arguments.

## Evaluation

These are **official InterMask paper results** on InterHuman, not a Motius
rerun. The protocol uses InterCLIP with 20 repetitions, except multimodality
which uses 5.

| R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Multimodality |
| ---: | ---: | ---: | --: | ------: | --------: | ------------: |
| 0.449 | 0.599 | 0.683 | 5.154 | 3.790 | 7.944 | 1.737 |

Use `tools/eval_interhuman262.py` and the public InterCLIP artifact for a local
reproduction. Input packs contain `m1`, `m2`, `lens`, and `texts` arrays.

## Motion Representation

The public InterHuman artifact returns native paired InterHuman-262 after
de-normalization. Motius does not independently canonicalize the two people.
Exact joint positions are available through:

```python
from motius.motion import convert_motion

joints = convert_motion(motion[0], "interhuman262", "joints")
# (T, 2, 22, 3), still in one shared interaction frame
```

## Verification

The original VQ and Transformer training archives were converted to two
SafeTensors files with all optimizer/scheduler state removed. The artifact
loads with zero missing or unexpected keys. A deterministic 60-frame sample is
exactly equal before and after conversion (`max_abs=0`, `mean_abs=0`).

The vendored method runtime retains the upstream MIT license in
`motius/models/intermask/LICENSE`.
