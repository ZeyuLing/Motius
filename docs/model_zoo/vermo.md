<h1 align="center">VerMo Model Card</h1>

<p align="center">
  <strong>A Motius-native autoregressive motion-language baseline aligned on HumanML3D captions.</strong>
</p>

<p align="center">
  <a href="https://github.com/ZeyuLing/Motius/tree/main/motius/models/vermo">Motius Implementation</a> |
  <a href="https://huggingface.co/ZeyuLing/Motius-VerMo-HumanML3D">Motius Checkpoint</a>
</p>

VerMo is a Motius-native research baseline rather than a reproduction of an
external paper. The released M2T checkpoint uses a Llama-3.2-1B-Instruct
language backbone, a 16K motion tokenizer, and an explicit SMPL-22 motion
processor. No external paper or original repository is claimed for this row.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Released task | M2T |
| Evaluation input | HumanML3D-263, converted through the Motius SMPL-22 bridge |
| Native motion representation | VerMo-138, 20 fps |
| Motion tokenizer | 16K VQ motion tokenizer |
| Language backbone | Llama-3.2-1B-Instruct |
| Checkpoint | [`ZeyuLing/Motius-VerMo-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-VerMo-HumanML3D) |
| Pipeline | `motius.pipelines.vermo.VermoPipeline` |

## Usage

```python
import numpy as np
from motius.pipelines.vermo import VermoPipeline

pipe = VermoPipeline.from_pretrained(
    "ZeyuLing/Motius-VerMo-HumanML3D",
    bundle_kwargs={"device": "cuda"},
    smpl_model_dir="checkpoints/body_models/smpl",
)
motion = np.load("sample.npy")  # denormalized HumanML3D-263
caption = pipe.infer_m2t([motion], lengths=[len(motion)])[0]
```

## M2T Evaluation

| Protocol | Samples | BLEU-4 | ROUGE-L | CIDEr | BERT raw | BERT rescaled | R@1 | R@2 | R@3 | Matching |
| -------- | ------: | -----: | ------: | ----: | -------: | --------------: | --: | --: | --: | -------: |
| [HumanML3D M2T](../tasks/m2t.md) | 4,400 | 0.0579 | 0.3703 | 0.0948 | 0.8878 | 0.3350 | 0.5055 | 0.7021 | 0.7972 | 2.9419 |

The [M2T case explorer](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard#case-explorer)
contains VerMo's prediction for every one of the 4,400 evaluated clips.

### M2T Demo Cases

| Sample | Human reference | VerMo prediction | Motion |
| ------ | --------------- | ---------------- | ------ |
| `000000` | a man kicks something or someone with his left leg. | a person kicks with their left leg. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000000%230) |
| `000019` | person jogs around to the left and right | a person jogs in a half a circle | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000019%230) |
| `004545` | a person jumping while raising both hands and moving apart legs. | a person does jumping jacks. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=004545%230) |

## Motion Representation

VerMo-138 stores absolute root translation (3), frame-to-frame root
translation (3), and 22 local joint rotations in column-major 6D form (132).
HumanML3D inputs are recovered to SMPL-22 joints, solved to `motion135` with
position IK, then repacked explicitly from row-major to column-major 6D.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius/pipelines/vermo/pipeline.py` |
| Bundle | `motius/models/vermo/bundle.py` |
| Processor | `motius/models/vermo/processor.py` |
| Motion tasks | `motius/models/vermo/task_utils/` |
