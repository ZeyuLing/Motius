<h1 align="center">InterGen Model Card</h1>

<p align="center">
  <strong>Diffusion-based text generation for two-person interactions.</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2304.05684">Paper</a> |
  <a href="https://tr3e.github.io/intergen-page/">Project Page</a> |
  <a href="https://github.com/tr3e/InterGen">Original GitHub</a> |
  <a href="https://huggingface.co/ZeyuLing/motius-intergen-interhuman">Motius Checkpoint</a>
</p>

InterGen uses cooperative diffusion denoisers with shared weights and mutual
attention to generate two synchronized people from one interaction caption.
Motius packages the released InterHuman checkpoint behind a self-contained
SafeTensors pipeline without importing an upstream checkout.

## Preview

| Input Text | Paired SMPL Preview |
| ---------- | ------------------- |
| two people shake hands and then step apart | ![InterGen handshake](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/intergen/intergen_interhuman_handshake_smpl_pair_512_30fps.gif) |
| one person helps another person stand up | ![InterGen helping interaction](https://raw.githubusercontent.com/ZeyuLing/Motius/main/assets/model_zoo/intergen/intergen_interhuman_help_stand_smpl_pair_512_30fps.gif) |

The blue and coral meshes are fitted jointly framed SMPL bodies. The native
model output remains paired InterHuman-262 rather than SMPL parameters.

## Release Snapshot

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

## Usage

```bash
pip install -e ".[intergen]"
```

```python
from motius.pipelines.intergen import InterGenPipeline

pipe = InterGenPipeline.from_pretrained(
    "ZeyuLing/motius-intergen-interhuman",
    bundle_kwargs={"device": "cuda"},
)
motion = pipe(
    "two people shake hands and then step apart",
    motion_len=120,
    seed=42,
)  # (1, 120, 2, 262)
```

## Evaluation

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

## Motion Representation

Each person stores 66 global joint positions, 66 global joint displacements,
126 local-rotation channels for the 21 non-root joints, and four foot contacts.
Both people share person 1's canonical frame, preserving their relative yaw and
translation. See the [InterHuman-262 reference](../motion/representations.md#interhuman-262).

## Verification

The Motius network loads the released checkpoint with zero missing or
unexpected keys. Under the same prompt, seed, length, and DDIM50 schedule, the
legacy artifact and the SafeTensors artifact are exactly equal (`max_abs=0`,
`mean_abs=0`). The Hub runtime contains no external source checkout, upstream
package, or absolute workspace import.

The InterHuman dataset is not redistributed. The official repository states
that the code and materials use CC BY-NC-SA 4.0 and separately prohibits
dataset redistribution.
