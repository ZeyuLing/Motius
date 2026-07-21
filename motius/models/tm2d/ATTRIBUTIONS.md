# TM2D attribution

This package is a Motius-native implementation of the architecture introduced
in **TM2D: Bimodality Driven 3D Dance Generation via Music-Text Integration**
(ICCV 2023).

- Paper: https://openaccess.thecvf.com/content/ICCV2023/html/Gong_TM2D_Bimodality_Driven_3D_Dance_Generation_via_Music-Text_Integration_ICCV_2023_paper.html
- Project: https://garfield-kh.github.io/TM2D/
- Upstream repository: https://github.com/Garfield-kh/TM2D
- Audited upstream revision: `98bef9571419b6459927630d5d96f8450898687e`

The runtime code in this directory does not import an upstream checkout. The
released weights were converted from the authors' `E0190` VQ-VAE and `E0020`
joint Transformer checkpoints. The upstream repository did not contain a
license file at the audited revision; users remain responsible for complying
with the authors' terms for those weights and associated vocabulary data.
