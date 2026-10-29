"""Native GEM-X feature-to-body inference composition."""
from motius.motion.conditioning.gem_x_processor import GemXProcessor
from motius.pipelines.gem_prediction import GEMPrediction
from .inference import GemXInference


class GemXComposition(GemXInference,GEMPrediction,GemXProcessor):
    def __init__(self,bundle,codec=None):
        GemXProcessor.__init__(self,bundle,codec)
        self.initialize_prediction()
