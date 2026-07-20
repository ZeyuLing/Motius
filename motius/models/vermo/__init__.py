from motius.models.vermo.bundle import VermoBundle
from motius.models.vermo.fs_quantizer import FSQuantizer
from motius.models.vermo.llama import VermoLlamaForCausalLM
from motius.models.vermo.processor import VermoProcessor
from motius.models.vermo.pose_processor import VermoSMPL22Processor
from motius.models.vermo.vqvae_2d import VQVAEVermo2DTK
from motius.models.vermo.vqvae_1d import VQVAEVermo1D
from motius.registry import HF_MODELS

try:
    from motius.models.vermo.qwen3 import VermoQwen3ForCausalLM
except (ImportError, ModuleNotFoundError):
    VermoQwen3ForCausalLM = None

# Backward-compatible aliases used in existing VerMo configs/code paths.
VQVAEWanMotion1D = VQVAEVermo1D
VQVAEWanMotion2DTK = VQVAEVermo2DTK

if not HF_MODELS.get('VQVAEWanMotion1D'):
    HF_MODELS.register_module(name='VQVAEWanMotion1D', module=VQVAEWanMotion1D, force=True)
if not HF_MODELS.get('VQVAEWanMotion2DTK'):
    HF_MODELS.register_module(name='VQVAEWanMotion2DTK', module=VQVAEWanMotion2DTK, force=True)

__all__ = [
    'VermoBundle',
    'VermoLlamaForCausalLM',
    'VermoProcessor',
    'VermoSMPL22Processor',
    'FSQuantizer',
    'VQVAEVermo1D',
    'VQVAEVermo2DTK',
    'VQVAEWanMotion1D',
    'VQVAEWanMotion2DTK',
]

if VermoQwen3ForCausalLM is not None:
    __all__.append('VermoQwen3ForCausalLM')
