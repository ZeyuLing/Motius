<h1 align="center">TM2T Model Card</h1>

<p align="center">
  <strong>Tokenized reciprocal motion-language translation, reproduced as a standalone Motius M2T pipeline.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2207.01696">Paper</a> |
  <a href="https://ericguo5513.github.io/TM2T/">Project Page</a> |
  <a href="https://github.com/EricGuo5513/TM2T">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/Motius-TM2T-HumanML3D">Motius Checkpoint</a>
</p>

TM2T is the ECCV 2022 reciprocal text-to-motion and motion-to-text method. The
Motius release contains the HumanML3D VQ tokenizer, motion-to-text Transformer,
vocabulary, statistics, and inference runtime. It does not import an original
repository checkout.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Tasks | Motion-to-Text |
| Motion representation | HumanML3D-263, 20 fps |
| Motion tokenizer | 1,024-code VQ tokenizer |
| Caption model | 4-layer encoder / 4-layer decoder Transformer |
| Decoding | Beam search, beam size 2 |
| Checkpoint | [`ZeyuLing/Motius-TM2T-HumanML3D`](https://huggingface.co/ZeyuLing/Motius-TM2T-HumanML3D) |
| Pipeline | `motius.pipelines.tm2t.TM2TPipeline` |

## Usage

```python
import numpy as np
from motius.pipelines.tm2t import TM2TPipeline

pipe = TM2TPipeline.from_pretrained(
    "ZeyuLing/Motius-TM2T-HumanML3D",
    bundle_kwargs={"device": "cuda"},
)
motion = np.load("sample.npy")  # denormalized HumanML3D-263
caption = pipe.infer_m2t([motion], lengths=[len(motion)])[0]
```

## M2T Evaluation

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

| Sample | Human reference | TM2T prediction | Motion |
| ------ | --------------- | --------------- | ------ |
| `000000` | a man kicks something or someone with his left leg. | a person kick something with their left foot | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000000%230) |
| `000019` | person jogs around to the left and right | a person jog in place then jog to the right then jog to the left | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=000019%230) |
| `004545` | a person jumping while raising both hands and moving apart legs. | a person is do jump jack | [Play](https://zeyuling-m2t-humanml3d-leaderboard.static.hf.space/cases/index.html?case=004545%230) |

## Motion Representation

TM2T normalizes HumanML3D-263 features with its released training statistics.
The VQ encoder removes four contact dimensions, maps each clip to discrete
motion tokens, and the reciprocal Transformer translates those tokens to text.

## Motius Components

| Component | Path |
| --------- | ---- |
| Pipeline | `motius/pipelines/tm2t/pipeline.py` |
| Bundle | `motius/models/tm2t/bundle.py` |
| Runtime | `motius/models/tm2t/network.py` |
| License | `motius/models/tm2t/LICENSE` |

## Citation

```bibtex
@inproceedings{guo2022tm2t,
  title={TM2T: Stochastic and Tokenized Modeling for the Reciprocal Generation of 3D Human Motions and Texts},
  author={Guo, Chuan and Zuo, Xinxin and Wang, Sen and Cheng, Li},
  booktitle={European Conference on Computer Vision},
  year={2022}
}
```
