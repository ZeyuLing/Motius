# EDGE Attribution

This package adapts the official MIT-licensed implementation of
[EDGE: Editable Dance Generation From Music](https://github.com/Stanford-TML/EDGE)
at revision `17c3428669ed6733edd9d8c66f7dc62060b8e46d`.

Copyright (c) 2022 Jonathan Tseng, Rodrigo Castellon, and C. Karen Liu.
The original license is included in `LICENSE`.

The native training/optimizer/sampling/pose/FK operations follow the same original repository. The selected rotation conversions are from Meta PyTorch3D v0.7.2, under the full BSD notice in PYTORCH3D_LICENSE. The independent original comparison pins that rotation implementation and controlled FP32/math attention; EDGE's original README does not pin a PyTorch3D version.
