"""Native GEM-SMPL feature-to-body inference composition."""
from motius.motion.conditioning.gem_smpl_processor import GemSmplProcessor
from motius.pipelines.gem_prediction import GEMPrediction
from .inference import GemSmplInference


class GemSmplComposition(GemSmplInference,GEMPrediction,GemSmplProcessor):
    def __init__(self,bundle,codec=None):
        GemSmplProcessor.__init__(self,bundle,codec)
        self.initialize_prediction()
