"""T2M-GPT networks used by the Motius runtime.

The package contains only the model components needed for inference. Training-
only and evaluator-only upstream utilities are intentionally outside the
runtime package.

Public entry points used by :class:`T2MGPTBundle`:

* :class:`HumanVQVAE` — VQ-VAE (Encoder/Decoder + EMA-reset quantizer).
* :class:`Text2Motion_Transformer` — the cross-conditional GPT.
"""

from .t2m_trans import Text2Motion_Transformer
from .vqvae import HumanVQVAE, VQVAE_251

__all__ = ["HumanVQVAE", "VQVAE_251", "Text2Motion_Transformer"]
