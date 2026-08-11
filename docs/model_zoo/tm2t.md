<h1 align="center">TM2T Model Card</h1>

<p align="center">
  <strong>Tokenized reciprocal motion-language translation, reproduced as a standalone Motius M2T pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2207.01696">Paper</a> ·
  <a href="https://ericguo5513.github.io/TM2T/">Project Page</a> ·
  <a href="https://github.com/EricGuo5513/TM2T">Original GitHub</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-TM2T-HumanML3D">Motius Checkpoint</a>
</p>

TM2T is the ECCV 2022 reciprocal text-to-motion and motion-to-text method. The
Motius release contains the HumanML3D VQ tokenizer, motion-to-text Transformer,
vocabulary, statistics, and inference runtime. It does not import an original
repository checkout.

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
| Motion-to-Text | SMPL motion input | <video src="https://github.com/user-attachments/assets/0699f1cd-361e-4fe4-b74f-beb92b38b333" controls></video> | [MP4](https://github.com/user-attachments/assets/0699f1cd-361e-4fe4-b74f-beb92b38b333) · [All cases](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?method=tm2t) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


The
[M2T all-case explorer](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html)
shows every evaluated HumanML3D clip, human reference, and TM2T prediction in
one comparison surface. The representative cases below remain linked directly
from the evaluation table.

## Model Overview

<!-- MOTIUS_MODEL_CARD_TASKS:START -->

### Task APIs

| Task | Pipeline API | Evaluation and examples |
| --- | --- | --- |
| Motion-to-Text | `infer_motion_to_text` | [Benchmark and examples](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard) |

<!-- MOTIUS_MODEL_CARD_TASKS:END -->

<!-- MOTIUS_FRAME_RATE_CONTRACT:START -->

### Frame-Rate Contract

| Clock | Rate |
| --- | --- |
| Training motion | 20 fps motion input (HumanML3D) |
| Public preview | 30 fps, duration-preserving 20→30 fps resampling |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Tasks | Motion-to-Text |
| Motion representation | HumanML3D-263, 20 fps |
| Motion tokenizer | 1,024-code VQ tokenizer |
| Caption model | 4-layer encoder / 4-layer decoder Transformer |
| Decoding | Beam search, beam size 2 |
| Checkpoint | [`ZeyuLing/Motius-TM2T-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-TM2T-HumanML3D) |
| Pipeline | `motius.pipelines.tm2t.TM2TPipeline` |

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius/pipelines/tm2t/pipeline.py` |
| Bundle | `motius/models/tm2t/bundle.py` |
| Runtime | `motius/models/tm2t/network.py` |
| License | `motius/models/tm2t/LICENSE` |

## Quick Start

```python
import numpy as np
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-TM2T-HumanML3D",
    bundle_kwargs={"device": "cuda"},
)
motion = np.load("sample.npy")  # denormalized HumanML3D-263
caption = pipe.infer_motion_to_text([motion], lengths=[len(motion)])[0]
```

## Evaluation Results

<!-- MOTIUS_CANONICAL_METRICS:START -->

> **Canonical metrics.** Public results are tied to the sources below. Motius/uTMR FID always means per-sample L2-normalized embedding-space FID; `—` means the normalized value has not been recomputed. Historical raw-space FID is never substituted.

| Task | Canonical result source | Protocol |
| --- | --- | --- |
| Motion-to-Text | [Published results](../leaderboards/hf_space_m2t_humanml3d/m2t_results.json) | HumanML3D M2T metrics |

### Canonical Motion-to-Text Snapshot

| Method | n | BLEU-1 | BLEU-4 | ROUGE-L | CIDEr | BERT raw | BERT rescaled | R@1 | R@2 | R@3 | Matching |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TM2T | 4,400 | 0.6105 | 0.2213 | 0.4911 | 0.7253 | 0.8940 | 0.3721 | 0.5180 | 0.7178 | 0.8079 | 2.9584 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


Full 4,400-sample evaluation follows the shared [HumanML3D M2T protocol](../tasks/m2t.md).
Results are published only after the complete prediction set and metric artifact
pass the population and sample-ID checks.

| Samples | BLEU-4 | ROUGE-L | CIDEr | BERT raw | BERT rescaled | R@1 | R@2 | R@3 | Matching |
| ------: | -----: | ------: | ----: | -------: | --------------: | --: | --: | --: | -------: |
| 4,400 | 0.2213 | 0.4911 | 0.7253 | 0.8940 | 0.3721 | 0.5180 | 0.7178 | 0.8079 | 2.9584 |

`BERT raw` is the unscaled RoBERTa-large cosine score. `BERT rescaled` applies
the official English layer-17 baseline (`0.83122575`) and is the TM2T-paper
scale. Browse every evaluated prediction in the [M2T case explorer](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard#case-explorer).

### M2T Demo Cases

| Human reference | TM2T prediction | Motion |
| --------------- | --------------- | ------ |
| a man kicks something or someone with his left leg. | a person kick something with their left foot | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000000%230) |
| person jogs around to the left and right | a person jog in place then jog to the right then jog to the left | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000019%230) |
| a person jumping while raising both hands and moving apart legs. | a person is do jump jack | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=004545%230) |

## Motion Representation

TM2T normalizes HumanML3D-263 features with its released training statistics.
The VQ encoder removes four contact dimensions, maps each clip to discrete
motion tokens, and the reciprocal Transformer translates those tokens to text.

## Citation and License

```bibtex
@inproceedings{guo2022tm2t,
  title={TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts},
  author={Guo, Chuan and Zuo, Xinxin and Wang, Sen and Cheng, Li},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
