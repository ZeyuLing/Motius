# GVHMR Attribution

This integration invokes the unmodified official
[zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) runtime at commit
`6ec3ca39336c50492c0fae65fba2fb831fc7d866` through a subprocess boundary.
No GVHMR model implementation or checkpoint is copied into Motius.

GVHMR is copyright 2022–2023 the 3D Vision Group at the State Key Lab of
CAD&CG, Zhejiang University. Its repository license permits educational,
research, and non-profit use, requires attribution and open-source
modifications, and prohibits commercial use without separate permission.
Consult the upstream `LICENSE` before use.

The release checkpoint is distributed by the GVHMR authors through the Google
Drive folder linked from upstream `docs/INSTALL.md`, subject to the associated
licenses. Motius does not redistribute it. The expected filename is
`inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt`; this integration computes
and records the SHA256 of the local file used for each result.

SMPL and SMPL-X model files must be obtained separately from their official
sites and remain subject to their own licenses.
