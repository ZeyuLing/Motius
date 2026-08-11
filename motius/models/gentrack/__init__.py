"""GenTrack generator post-training models and reward backends.

The ``PhysFlow*`` names are retained as checkpoint/config compatibility names
from the internal research implementation.  New public configs use the
``GenTrack*`` registry aliases below.
"""

from motius.models.gentrack.dataset import PhysFlowPromptDataset
from motius.models.gentrack.flow_grpo import group_relative_advantages
from motius.models.gentrack.g1_bundle import PhysFlowG1Bundle
from motius.registry import DATASETS, MODEL_BUNDLES


MODEL_BUNDLES.register_module(
    name="GenTrackG1Bundle",
    module=PhysFlowG1Bundle,
    force=True,
)
DATASETS.register_module(
    name="GenTrackPromptDataset",
    module=PhysFlowPromptDataset,
    force=True,
)

GenTrackG1Bundle = PhysFlowG1Bundle
GenTrackPromptDataset = PhysFlowPromptDataset

__all__ = [
    "GenTrackG1Bundle",
    "GenTrackPromptDataset",
    "PhysFlowG1Bundle",
    "PhysFlowPromptDataset",
    "group_relative_advantages",
]
