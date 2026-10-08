# ProjFlow Attribution

This package contains a Motius-native adaptation of the model architecture and
projection sampler released with:

> Akihisa Watanabe, Qing Yu, Edgar Simo-Serra, and Kent Fujiwara. ProjFlow:
> Projection Sampling with Flow Matching for Zero-Shot Exact Spatial Motion
> Control. CVPR 2026.

- Paper: https://arxiv.org/abs/2602.22742
- Project: https://akihisa-watanabe.github.io/projflow.github.io/
- Official code: https://github.com/Akihisa-Watanabe/ProjFlow
- Reference revision: `9550501a439964a73063505b7a52e574ae11a43c`
- Official weights: https://huggingface.co/Akihisa-Watanabe/ProjFlow

The implementation is packaged under `motius.models.projflow` and does not
import or execute code from an external ProjFlow checkout at runtime. The
upstream repository did not include a standalone license file at the reference
revision; downstream users should review the upstream terms before reuse.

## Joint normalization assets

`assets/joints_mean.npy` and `assets/joints_std.npy` are byte-for-byte copies of
`utils/22x3_mean_std/t2m/22x3_mean.npy` and `22x3_std.npy` at the reference
revision above. They normalize the official HumanML3D joint coordinates.

| File | SHA-256 |
| --- | --- |
| `joints_mean.npy` | `ebfe87efeb6828b33c69f9df88d7a0373b9af7ac546496a4c7701112a748f12f` |
| `joints_std.npy` | `4733059903169bc95993999fde322d8e84fb95f3e6ff1b277c283a0eb877d0c5` |
