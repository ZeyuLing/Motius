# PromptHMR Attribution

This integration invokes the official
[yufu-wang/PromptHMR](https://github.com/yufu-wang/PromptHMR) repository at
commit `3b566b7dbb28ce506c7ea972c18693f4c705ce8c`. No upstream source code is
vendored into Motius.

PromptHMR code, models, and derivative works are restricted to
**non-commercial scientific research, non-commercial education, and
non-commercial artistic projects** under the upstream PromptHMR license.
Commercial, pornographic, military, surveillance, misleading, libelous, and
defamatory uses are prohibited. Users must review and accept the complete
upstream `LICENSE` before downloading or running the software or checkpoints.
SMPL/SMPL-X body-model files have separate registration and license terms and
are never redistributed by this integration.

The preferred video checkpoint is the combined BEDLAM1+BEDLAM2 model. The
official BEDLAM2 checksum manifest published at
`https://download.is.tue.mpg.de/bedlam2/ml/videos/bedlam2_phmr.sha256` records:

- `phmr_b1.ckpt`: `d06ae5ddc74ef74c252f4ec34e4e3092cd8fc18cba104af5aa978cdd2c669b5a`
- `phmr_b1b2.ckpt`: `2a36132715b5db0ea2acb6f1f92bbf963c9cf0fb1c3aea8d0f73dfede0b9e5e5`
- `phmr_b2.ckpt`: `631433bf4dfd548dc5c6e2df037e11a11ce4a83c37367ee0f31b2f1627aa06d9`

The official default video pipeline also uses SAM 2, Detectron2 Keypoint
R-CNN, ViTPose-H, DeepLabV3, DROID-SLAM, Metric3D, SPEC, GVHMR components,
and an image PromptHMR checkpoint. Their code and weights remain governed by
their respective upstream licenses. Motius records these components as
provenance and hashes the local image/video PromptHMR checkpoints at runtime.

Optional geometry materialization is a separate
`licensed_smplx_replay` step. It requires a user-supplied SMPL-X model file,
checks its declared gender/version and SHA256, and records only publishable
provenance. The model file is never downloaded, copied, embedded, or
represented as an original field from PromptHMR's `results.pkl`.

Please cite the upstream paper:

```bibtex
@article{wang2025prompthmr,
  title={PromptHMR: Promptable Human Mesh Recovery},
  author={Wang, Yufu and Sun, Yu and Patel, Priyanka and Daniilidis, Kostas and Black, Michael J and Kocabas, Muhammed},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```
