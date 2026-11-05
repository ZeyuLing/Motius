"""VGGT-Omega network construction and strict checkpoint loading."""
from contextlib import nullcontext

import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from .models.vggt_omega import VGGTOmega


@MODEL_BUNDLES.register_module()
class VGGTOmegaBundle(ModelBundle):
    def __init__(self, checkpoint, *, low_cpu_mem_usage=True, **network_config):
        super().__init__()
        with torch.device('meta') if low_cpu_mem_usage else nullcontext():
            self.model = VGGTOmega(**network_config)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if low_cpu_mem_usage:
            targets = self.model.state_dict()
            state = {key: value.to(dtype=targets[key].dtype) if key in targets else value
                     for key, value in state.items()}
        self.model.load_state_dict(state, strict=True, assign=low_cpu_mem_usage)
        # Normalization constants are constructed explicitly as CPU FloatTensors;
        # all remaining buffers (including RoPE periods) come from the checkpoint.
        if any(t.is_meta for t in (*self.model.parameters(), *self.model.buffers())):
            raise RuntimeError('VGGT checkpoint left unmaterialized tensors')
        self._trainable_modules = ["model"]
        self._save_ckpt_modules = ["model"]
        self._module_checkpoint_formats = {"model": "full"}

    def predict(self, images):
        return self.model(images)
