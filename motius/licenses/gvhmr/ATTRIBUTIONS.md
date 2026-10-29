# GVHMR Attribution

Motius implements the released network, trainer, preprocessing, motion codec and
postprocessing as native components derived from
[zju3dv/GVHMR](https://github.com/zju3dv/GVHMR), revision
`6ec3ca39336c50492c0fae65fba2fb831fc7d866`.

The neural tree contains only the motion network and required attention layers.
HMR2 feature-token and ViTPose heatmap networks share their identical ViT
backbone under `motius.models.vision.gvhmr`. Inference, tracking, geometry and
rendering use their explicit framework owners. Original training applications,
registries, datasets, demos and unused vision architectures are excluded.
The optional DPVO configuration is pinned to submodule revision
`c0c5a104c9c58663aa9be62c3f125d5b52874f3e`.

See `motius/licenses/gvhmr/LICENSE` for the upstream terms.

GVHMR is copyright 2022–2023 the 3D Vision Group at the State Key Lab of
CAD&CG, Zhejiang University. Its repository license permits educational,
research, and non-profit use, requires attribution and open-source
modifications, and prohibits commercial use without separate permission.
Consult the upstream `LICENSE` before use.

The release checkpoint and preprocessing checkpoints originate from the
Google Drive folder linked by upstream `docs/INSTALL.md`. The Motius
Hugging Face artifact preserves their original bytes and filenames and records
their SHA256 digests. The GVHMR non-commercial terms continue to apply.

Licensed SMPL and SMPL-X model files are never redistributed. Users must place
them under `inputs/checkpoints/body_models/` in the downloaded artifact, or
pass an equivalent local body-model directory to the Motius bundle.
