"""GenTrack neural model bundle."""
from .g1_bundle import PhysFlowG1Bundle
from motius.registry import MODEL_BUNDLES
GenTrackG1Bundle=PhysFlowG1Bundle
MODEL_BUNDLES.register_module(name='GenTrackG1Bundle',module=GenTrackG1Bundle,force=True)
__all__=['PhysFlowG1Bundle','GenTrackG1Bundle']
