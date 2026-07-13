<h1 align="center">InterCLIP Evaluator Card</h1>

<p align="center">
  <strong>Official caption-to-interaction evaluator for InterHuman-262.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2304.05684">InterGen Paper</a> |
  <a href="https://github.com/tr3e/InterGen">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-evaluator-interhuman-interclip">Motius Checkpoint</a>
</p>

InterCLIP jointly embeds one caption and two synchronized InterHuman-262 tracks.
It is the official evaluator used by InterGen and by InterMask on InterHuman.

## Release Snapshot

| Item | Value |
| ---- | ----- |
| Motion input | `(B, T, 2, 262)` paired InterHuman-262 |
| Motion encoder | 8-layer Transformer, width 1024 |
| Text encoder | CLIP token embedding + 8-layer Transformer, width 768 |
| Embedding | 512D, L2 normalized, learned latent scale |
| Metrics | R@1/R@2/R@3, FID, MM-Dist, Diversity |
| Official retrieval batch | 96 |
| Official repeats | 20 |
| Checkpoint | [`ZeyuLing/motius-evaluator-interhuman-interclip`](https://huggingface.co/ZeyuLing/motius-evaluator-interhuman-interclip) |

## Input Pack

```python
import numpy as np

np.savez(
    "pred.npz",
    m1=motion_person_1,  # (N, T, 262)
    m2=motion_person_2,  # (N, T, 262)
    lens=lengths,        # (N,)
    texts=captions,      # (N,)
)
```

## Python API

```python
from motius.evaluation.evaluators import InterHuman262Evaluator

evaluator = InterHuman262Evaluator.from_pretrained(
    "ZeyuLing/motius-evaluator-interhuman-interclip",
    device="cuda",
)
metrics = evaluator.evaluate_npz(
    "data/interhuman/test_native262.npz",
    {
        "InterGen": "outputs/intergen_test_native262.npz",
        "InterMask": "outputs/intermask_test_native262.npz",
    },
)
```

## Command Line

```bash
python tools/eval_interhuman262.py \
  --evaluator ZeyuLing/motius-evaluator-interhuman-interclip \
  --gt data/interhuman/test_native262.npz \
  --pred InterGen=outputs/intergen_test_native262.npz \
  --pred InterMask=outputs/intermask_test_native262.npz \
  --output outputs/evaluation/interhuman/interclip.json
```

The CLI defaults match the official batch-96, 20-repeat retrieval protocol.
Use lower values only for smoke tests, and report the changed protocol.

## Verification

The official Lightning checkpoint was reduced to 583 MB of inference-only
SafeTensors weights. Text and motion embeddings are exactly equal to the legacy
loader (`max_abs=0`, `mean_abs=0`). A copied motion pack gives FID below
`3e-9`, validating the embedding and FID path numerically.

InterCLIP follows the InterGen repository's CC BY-NC-SA 4.0 terms. The
InterHuman dataset is not included in the artifact.
