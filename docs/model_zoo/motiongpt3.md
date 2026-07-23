<h1 align="center">MotionGPT3 Model Card</h1>

<p align="center">
  <strong>A continuous-latent bimodal motion-language model, packaged for HumanML3D motion captioning.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2506.24086">Paper</a> |
  <a href="https://motiongpt3.github.io/">Project Page</a> |
  <a href="https://github.com/OpenMotionLab/MotionGPT3">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/Motius-MotionGPT3-HumanML3D">Motius Checkpoint</a>
</p>

MotionGPT3 separates text and motion processing into modality-specific branches
with shared attention. Unlike tokenized motion-language models, it represents
motion in a continuous VAE latent space. The Motius artifact packages the final
official multi-task checkpoint and all model/tokenizer configuration required by
`Pipeline.from_pretrained`.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Tasks | Motion-to-Text |
| Input representation | HumanML3D-263, 20 fps |
| Motion latent | Continuous temporal VAE latent |
| Language model | GPT-2-family bimodal Transformer |
| Checkpoint provenance | Official final MotionGPT3 checkpoint |
| Checkpoint | [`ZeyuLing/Motius-MotionGPT3-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-MotionGPT3-HumanML3D) |
| Pipeline | `motius.pipelines.motiongpt3.MotionGPT3Pipeline` |

## Usage

```python
import numpy as np
from motius.pipelines.motiongpt3 import MotionGPT3Pipeline

pipe = MotionGPT3Pipeline.from_pretrained(
    "ZeyuLing/Motius-MotionGPT3-HumanML3D",
    bundle_kwargs={"device": "cuda"},
)
motion = np.load("sample.npy")  # denormalized HumanML3D-263
caption = pipe.infer_m2t([motion], lengths=[len(motion)])[0]
```

## M2T Evaluation

| Protocol | Samples | BLEU-4 | ROUGE-L | CIDEr | BERT raw | BERT rescaled | R@1 | R@2 | R@3 | Matching |
| -------- | ------: | -----: | ------: | ----: | -------: | --------------: | --: | --: | --: | -------: |
| [HumanML3D M2T](../tasks/m2t.md) | 4,400 | 0.0651 | 0.3672 | 0.1063 | 0.8820 | 0.3007 | 0.5333 | 0.7178 | 0.8043 | 2.8533 |

The [M2T case explorer](https://huggingface.co/spaces/ZeyuLing/m2t-humanml3d-leaderboard#case-explorer)
contains MotionGPT3's prediction for every one of the 4,400 evaluated clips.

### M2T Demo Cases

| Sample | Human reference | MotionGPT3 prediction | Motion |
| ------ | --------------- | --------------------- | ------ |
| `000000` | a man kicks something or someone with his left leg. | a person kicks with left leg. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000000%230) |
| `000019` | person jogs around to the left and right | a person runs to the right, then back to where they started. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000019%230) |
| `004545` | a person jumping while raising both hands and moving apart legs. | a person performs jumping jacks. | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=004545%230) |

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius/pipelines/motiongpt3/pipeline.py` |
| Bundle | `motius/models/motiongpt3/bundle.py` |
| Bimodal GPT runtime | `motius/models/motiongpt3/mot_example_gpt2_sepattn.py` |
| Generation runtime | `motius/models/motiongpt3/mot_example_gpt2_sepattn_gen.py` |

## Citation

```bibtex
@misc{zhu2025motiongpt3,
  title={MotionGPT3: Human Motion as a Second Modality},
  author={Zhu, Bingfan and Jiang, Biao and Wang, Sunyi and Tang, Shixiang and Chen, Tao and Luo, Linjie and Zheng, Youyi and Chen, Xin},
  year={2025},
  eprint={2506.24086},
  archivePrefix={arXiv}
}
```
