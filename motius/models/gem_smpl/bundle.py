"""GEM-SMPL's released conditioner and motion denoiser networks."""
from motius.models.gem_bundle import GEMModelBundle
from motius.registry import MODEL_BUNDLES

@MODEL_BUNDLES.register_module()
class GemSmplBundle(GEMModelBundle):
    VARIANT = 'gem_smpl'
    CHECKPOINT = 'gem_smpl.ckpt'
