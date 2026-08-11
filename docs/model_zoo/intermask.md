<h1 align="center">InterMask Model Card</h1>

<p align="center">
  <strong>Collaborative masked token generation for two-person interaction.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2410.10010">Paper</a> ·
  <a href="https://gohar-malik.github.io/intermask/">Project Page</a> ·
  <a href="https://github.com/gohar-malik/InterMask">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-intermask-interhuman">Motius Checkpoint</a>
</p>

InterMask represents each person's motion as a 2D discrete token map and fills
both masked token grids collaboratively with spatial, temporal, and cross-person
attention. This Motius release packages the official InterHuman model.

<!-- MOTIUS_MODEL_CARD_NAV:START -->
<p align="center">
  <a href="#visual-results">Visual Results</a> ·
  <a href="#model-overview">Overview</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#evaluation-results">Evaluation</a> ·
  <a href="#motion-representation">Motion Representation</a>
</p>
<!-- MOTIUS_MODEL_CARD_NAV:END -->

## Visual Results

<!-- MOTIUS_TASK_DEMOS:START -->

### Task Demos

| Task | Input / condition | Rendered output | More |
| --- | --- | --- | --- |
| Text-to-Multi-Person Motion | Native task input | <video src="https://github.com/user-attachments/assets/e939fb25-4b84-417d-b7e5-2339422f16bf" controls></video> | [MP4](https://github.com/user-attachments/assets/e939fb25-4b84-417d-b7e5-2339422f16bf) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input Text | Native InterHuman-262 Preview |
| ---------- | ------------------- |
| two people hug each other and then step back | <video src="https://github.com/user-attachments/assets/f8087bb6-e25d-4eff-8bed-ccba0ecb441f" controls></video> |
| one person gently pushes the other person backward | <video src="https://github.com/user-attachments/assets/e939fb25-4b84-417d-b7e5-2339422f16bf" controls></video> |

Both figures are rendered directly from the model's paired InterHuman-262
joints. The public preview does not pass through an SMPL fitting stage.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Text-to-Multi-Person Motion | `infer_text_to_multi_person_motion` | [Benchmark and examples](../leaderboards/text_to_multi_person_interhuman.md) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 30 fps (InterHuman-262) |
| Public preview | 30 fps native |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Tasks | Text-to-Multi-Person Motion |
| Dataset | InterHuman |
| Representation | `(B, T, 2, 262)` InterHuman-262, 30 fps |
| Tokenizer | Shared 2D RVQ-VAE |
| Generator | Collaborative Inter-M masked Transformer |
| Text encoder | CLIP ViT-L/14@336px text tower, frozen |
| Checkpoint | [`ZeyuLing/motius-intermask-interhuman`](https://huggingface.co/ZeyuLing/motius-intermask-interhuman) |
| Pipeline | `motius.pipelines.intermask.InterMaskPipeline` |
| License | MIT |

[Text-to-Multi-Person Motion benchmark](../leaderboards/text_to_multi_person_interhuman.md)

## Quick Start

```bash
pip install -e ".[intermask]"
```

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-intermask-interhuman",
    bundle_kwargs={"device": "cuda"},
)
motion = pipe.infer_text_to_multi_person_motion(
    "two people hug each other and then step back",
    motion_len=120,  # multiple of four, 16..300
    seed=42,
)  # (1, 120, 2, 262)
```

`cond_scale`, `time_steps`, `topk_filter_thres`, and `temperature` are exposed
as optional pipeline arguments.

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Multi-Person Motion | [Published results](../leaderboards/text_to_multi_person_interhuman.md) | InterHuman protocol; rows remain pending |

<!-- MOTIUS_CANONICAL_METRICS:END -->


These are **official InterMask paper results** on InterHuman, not a Motius
rerun. The protocol uses InterCLIP with 20 repetitions, except multimodality
which uses 5.

| R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Multimodality |
| ---: | ---: | ---: | --: | ------: | --------: | ------------: |
| 0.449 | 0.599 | 0.683 | 5.154 | 3.790 | 7.944 | 1.737 |

Use `tools/eval_interhuman262.py` and the public InterCLIP artifact for a local
reproduction. Input packs contain `m1`, `m2`, `lens`, and `texts` arrays.

### Verification

The original VQ and Transformer training archives were converted to two
SafeTensors files with all optimizer/scheduler state removed. The artifact
loads with zero missing or unexpected keys. A deterministic 60-frame sample is
exactly equal before and after conversion (`max_abs=0`, `mean_abs=0`).

The vendored method runtime retains the upstream MIT license in
`motius/models/intermask/LICENSE`.

## Motion Representation

The public InterHuman artifact returns native paired InterHuman-262 after
de-normalization. Motius does not independently canonicalize the two people.
Exact joint positions are available through:

```python
from motius.motion import convert_motion

joints = convert_motion(motion[0], "interhuman262", "joints")
# (T, 2, 22, 3), still in one shared interaction frame
```

## Citation and License

- Paper: [InterMask: 3D Human Interaction Generation via Collaborative
  Masked Modeling](https://arxiv.org/abs/2410.10010)
- Original source:
  [gohar-malik/InterMask](https://github.com/gohar-malik/InterMask)
- License: MIT, preserved in
  [`motius/models/intermask/LICENSE`](../../motius/models/intermask/LICENSE).

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
