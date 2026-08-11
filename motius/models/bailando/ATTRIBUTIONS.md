# Bailando Attribution

This implementation adapts the network architecture and evaluation protocol
from [lisiyao21/Bailando](https://github.com/lisiyao21/Bailando) at revision
`cc90b98bff81c9709570db413c9610c2562e27ca`.

Bailando is distributed under the S-Lab License 1.0. The vendored license is
included as `LICENSE`. The kinetic and geometric feature implementations used
by the evaluator originate from Meta/Facebook Fairmotion and retain their BSD
license headers.

The AIST++ reference features are derived from the AIST++ Dance Motion Dataset,
Copyright 2021 Google LLC and licensed under CC BY 4.0. The evaluator artifact
contains derived feature vectors and source-sequence audit metadata, not the raw
AIST++ motion files.

Paper: Li Siyao et al., "Bailando: 3D Dance Generation by Actor-Critic GPT
With Choreographic Memory," CVPR 2022.
