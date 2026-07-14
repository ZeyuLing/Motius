# MotionCLR Attribution

MotionCLR is licensed under the IDEA License 1.0, Copyright (c) IDEA. All
Rights Reserved. The license is included in `LICENSE` and permits
non-commercial research use under its stated terms.

Authoritative source:

- Repository: `https://github.com/IDEA-Research/MotionCLR`
- Revision: `a6f44a791940682fe335c82f1b436bae05a1cebb`
- Released weights: `EvanTHU/MotionCLR`
- Upstream capability represented here: HumanML3D text-to-motion generation

Adapted files:

- `models/unet.py` -> `motius/models/motionclr/network.py`
- `models/gaussian_diffusion.py` and `config/diffuser_params.yaml` ->
  `motius/pipelines/motionclr/pipeline.py`
- `models/__init__.py`, `options/train_options.py`, and the released `opt.txt`
  -> official architecture defaults in `motius/models/motionclr/bundle.py`

The Motius adaptation preserves official module/state-dict names, uses
package-local imports, and removes visualization/editing globals that do not
own checkpoint parameters. Motius artifacts may package the public OpenAI CLIP
ViT-B/32 weight file under `clip/ViT-B-32.pt`; when present,
`from_pretrained` uses that local file and does not require a second model
download. The upstream `clip` package still constructs the encoder.

Official release checksums:

- `model/latest.tar`: `5852e139bbe45f5ca45b67b72cc54ab02b7da7ae18b42f27ea630a715c5c2b5f`
- `meta/mean.npy`: `0bdb5ba69a3a9e34d71990db15bc535ebc024c8d95ddb5574196f96058faa7d3`
- `meta/std.npy`: `487855309295f986d08e96d65e415fb6b2a94211ac34ce444007e84cba8f33bb`
