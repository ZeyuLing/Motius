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
