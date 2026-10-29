"""GEM-X's released conditioner and motion denoiser networks."""
from motius.models.gem_bundle import GEMModelBundle
from motius.registry import MODEL_BUNDLES

@MODEL_BUNDLES.register_module()
class GemXBundle(GEMModelBundle):
    VARIANT = 'gem_x'
    CHECKPOINT = 'gem_soma.ckpt'
