# GVHMR Attribution

Motius vendors the inference runtime from
[zju3dv/GVHMR](https://github.com/zju3dv/GVHMR) at immutable revision
`6ec3ca39336c50492c0fae65fba2fb831fc7d866`.

The vendored source is distributed for educational, research, and non-profit
use under the upstream terms in `vendor/GVHMR_LICENSE`. Motius changes the
package namespace, artifact path resolution, command entrypoint, optional
rendering boundary, degenerate two-view handling, output contract, and parity
instrumentation. The numerical preprocessing and network calls retain the
upstream order and implementation.

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
