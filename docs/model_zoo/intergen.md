<h1 align="center">InterGen Model Card</h1>

<p align="center">
  <strong>Diffusion-based text generation for two-person interactions.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2304.05684">Paper</a> ·
  <a href="https://tr3e.github.io/intergen-page/">Project Page</a> ·
  <a href="https://github.com/tr3e/InterGen">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/motius-intergen-interhuman">Motius Checkpoint</a>
</p>

InterGen uses cooperative diffusion denoisers with shared weights and mutual
attention to generate two synchronized people from one interaction caption.
Motius packages the released InterHuman checkpoint behind a self-contained
SafeTensors pipeline without importing an upstream checkout.

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
| Text-to-Multi-Person Motion | Native task input | <video src="https://github.com/user-attachments/assets/09d7b6a5-b203-4be0-a0ca-3258a480165f" controls></video> | [MP4](https://github.com/user-attachments/assets/09d7b6a5-b203-4be0-a0ca-3258a480165f) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


| Input Text | Native InterHuman-262 Preview |
| ---------- | ------------------- |
| two people shake hands and then step apart | <video src="https://github.com/user-attachments/assets/09d7b6a5-b203-4be0-a0ca-3258a480165f" controls></video> |
| one person helps another person stand up | <video src="https://github.com/user-attachments/assets/3be9194a-a9db-4d39-9668-76d485863dfc" controls></video> |

The blue and coral figures are rendered directly from the paired
InterHuman-262 joints. The public preview deliberately avoids an IK-derived
SMPL fit, so conversion artifacts cannot be mistaken for model behavior.

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
| Text encoder | CLIP ViT-L/14@336px text tower, frozen |
| Default sampler | DDIM, 50 steps |
| Checkpoint | [`ZeyuLing/motius-intergen-interhuman`](https://huggingface.co/ZeyuLing/motius-intergen-interhuman) |
| Pipeline | `motius.pipelines.intergen.InterGenPipeline` |
| License | CC BY-NC-SA 4.0, following the official repository |

[Text-to-Multi-Person Motion benchmark](../leaderboards/text_to_multi_person_interhuman.md)

## Quick Start

```bash
pip install -e ".[intergen]"
```

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/motius-intergen-interhuman",
    bundle_kwargs={"device": "cuda"},
)
motion = pipe.infer_text_to_multi_person_motion(
    "two people shake hands and then step apart",
    motion_len=120,
    seed=42,
)  # (1, 120, 2, 262)
```

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Text-to-Multi-Person Motion | [Published results](../leaderboards/text_to_multi_person_interhuman.md) | InterHuman protocol; rows remain pending |

<!-- MOTIUS_CANONICAL_METRICS:END -->


The following values are **reported by the InterGen paper**, not a new Motius
rerun. The official protocol uses InterCLIP on the complete InterHuman test set,
20 repetitions for all metrics except multimodality (5 repetitions).

| R@1 | R@2 | R@3 | FID | MM-Dist | Diversity | Multimodality |
| ---: | ---: | ---: | --: | ------: | --------: | ------------: |
| 0.371 | 0.515 | 0.624 | 5.918 | 5.108 | 7.387 | 2.141 |

Run the packaged evaluator on native outputs:

```bash
python tools/eval_interhuman262.py \
  --evaluator ZeyuLing/motius-evaluator-interhuman-interclip \
  --gt data/interhuman/test_native262.npz \
  --pred InterGen=outputs/intergen_test_native262.npz \
  --output outputs/evaluation/intergen/interclip.json
```

### Verification

The Motius network loads the released checkpoint with zero missing or
unexpected keys. Under the same prompt, seed, length, and DDIM50 schedule, the
legacy artifact and the SafeTensors artifact are exactly equal (`max_abs=0`,
`mean_abs=0`). The Hub runtime contains no external source checkout, upstream
package, or absolute workspace import.

The InterHuman dataset is not redistributed. The official repository states
that the code and materials use CC BY-NC-SA 4.0 and separately prohibits
dataset redistribution.

## Motion Representation

Each person stores 66 global joint positions, 66 global joint displacements,
126 local-rotation channels for the 21 non-root joints, and four foot contacts.
Both people share person 1's canonical frame, preserving their relative yaw and
translation. See the [InterHuman-262 reference](../motion/representations.md#interhuman-262).

## Citation and License

- Paper: [InterGen: Diffusion-based Multi-human Motion Generation under
  Complex Interactions](https://arxiv.org/abs/2304.05684)
- Original source: [tr3e/InterGen](https://github.com/tr3e/InterGen)
- Release terms: CC BY-NC-SA 4.0; packaged notices are recorded in
  [`motius/models/intergen/NOTICE`](../../motius/models/intergen/NOTICE).

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
