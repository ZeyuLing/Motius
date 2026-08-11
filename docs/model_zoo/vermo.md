<h1 align="center">VerMo Model Card</h1>

<p align="center">
  <strong>A Motius-native autoregressive motion-language baseline aligned on HumanML3D captions.</strong>
</p>

<p align="center">
  <a href="https://github.com/ZeyuLing/Motius/tree/main/motius/models/vermo">Motius Implementation</a> ·
  <a href="https://huggingface.co/ZeyuLing/Motius-VerMo-HumanML3D">Motius Checkpoint</a>
</p>

VerMo is a Motius-native research baseline rather than a reproduction of an
external paper. The released M2T checkpoint uses a Llama-3.2-1B-Instruct
language backbone, a 16K motion tokenizer, and an explicit SMPL-22 motion
processor. No external paper or original repository is claimed for this row.

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
| Motion-to-Text | SMPL motion input | <video src="https://github.com/user-attachments/assets/0699f1cd-361e-4fe4-b74f-beb92b38b333" controls></video> | [MP4](https://github.com/user-attachments/assets/0699f1cd-361e-4fe4-b74f-beb92b38b333) · [All cases](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?method=vermo) |

Every public `infer_*` API is represented by a GitHub-native H.264 video player. **All cases** opens the optional interactive comparison.

<!-- MOTIUS_TASK_DEMOS:END -->


The
[M2T all-case explorer](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html)
shows every evaluated HumanML3D clip, selected human reference, and VerMo
prediction with synchronized SMPL Mesh playback.

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
| Training motion | 20 fps motion input (VerMo-138) |
| Public preview | 30 fps, duration-preserving 20→30 fps resampling |

Training FPS is the checkpoint's native temporal clock. Preview FPS only controls media playback; any conversion listed above preserves duration.

<!-- MOTIUS_FRAME_RATE_CONTRACT:END -->


| Item | Value |
| ---- | ----- |
| Tasks | Motion-to-Text |
| Evaluation input | HumanML3D-263, converted through the Motius SMPL-22 bridge |
| Native motion representation | VerMo-138, 20 fps |
| Motion tokenizer | 16K VQ motion tokenizer |
| Language backbone | Llama-3.2-1B-Instruct |
| Checkpoint | [`ZeyuLing/Motius-VerMo-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-VerMo-HumanML3D) |
| Pipeline | `motius.pipelines.vermo.VermoPipeline` |

### Implementation Notes

| Component | Path |
| --------- | ---- |
| Pipeline | `motius/pipelines/vermo/pipeline.py` |
| Bundle | `motius/models/vermo/bundle.py` |
| Processor | `motius/models/vermo/processor.py` |
| Motion tasks | `motius/models/vermo/task_utils/` |

## Quick Start

```python
import numpy as np
from motius import Pipeline

pipe = Pipeline.from_pretrained(
    "ZeyuLing/Motius-VerMo-HumanML3D",
    bundle_kwargs={"device": "cuda"},
    smpl_model_dir="checkpoints/body_models/smpl",
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
| VerMo | 4,400 | 0.4501 | 0.0579 | 0.3703 | 0.0948 | 0.8878 | 0.3350 | 0.5055 | 0.7021 | 0.7972 | 2.9419 |

<!-- MOTIUS_CANONICAL_METRICS:END -->


| Protocol | Samples | BLEU-4 | ROUGE-L | CIDEr | BERT raw | BERT rescaled | R@1 | R@2 | R@3 | Matching |
| -------- | ------: | -----: | ------: | ----: | -------: | --------------: | --: | --: | --: | -------: |
| [HumanML3D M2T](../tasks/m2t.md) | 4,400 | 0.0579 | 0.3703 | 0.0948 | 0.8878 | 0.3350 | 0.5055 | 0.7021 | 0.7972 | 2.9419 |

The [M2T case explorer](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard#case-explorer)
contains VerMo's prediction for every one of the 4,400 evaluated clips.

### M2T Demo Cases

| Human reference | VerMo prediction | Motion |
| --------------- | ---------------- | ------ |
| a man kicks something or someone with his left leg. | a person kicks with their left leg. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000000%230) |
| person jogs around to the left and right | a person jogs in a half a circle | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000019%230) |
| a person jumping while raising both hands and moving apart legs. | a person does jumping jacks. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=004545%230) |

## Motion Representation

VerMo-138 stores absolute root translation (3), frame-to-frame root
translation (3), and 22 local joint rotations in column-major 6D form (132).
HumanML3D inputs are recovered to SMPL-22 joints, solved to `motion135` with
position IK, then repacked explicitly from row-major to column-major 6D.

## Citation and License

VerMo is a Motius-native baseline. No external paper or original repository is
claimed. The implementation and checkpoint follow the Motius repository
license; third-party Llama and body-model assets retain their own terms.

<!-- MOTIUS_MODEL_CARD_FOOTER:START -->
---

<p align="center">
  <a href="README.md">Model Zoo</a> ·
  <a href="../tasks/README.md">Task Registry</a> ·
  <a href="../leaderboards/README.md">Benchmark Hub</a> ·
  <a href="../motion/README.md">Motion Toolkit</a>
</p>
<!-- MOTIUS_MODEL_CARD_FOOTER:END -->
