"""YOLOX network construction and strict checkpoint loading."""
from pathlib import Path

import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from .yolo_head import YOLOXHead
from .yolo_pafpn import YOLOPAFPN
from .yolox import YOLOX


@MODEL_BUNDLES.register_module()
class YOLOXBundle(ModelBundle):
    def __init__(self, depth=1.0, width=1.0, num_classes=80, checkpoint=None):
        super().__init__()
        self.model = YOLOX(YOLOPAFPN(depth, width), YOLOXHead(num_classes, width))
        for module in self.model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eps = 1e-3
                module.momentum = 0.03
        self.model.head.initialize_biases(1e-2)
        self._trainable_modules = ["model"]
        self._save_ckpt_modules = ["model"]
        self._module_checkpoint_formats = {"model": "full"}
        if checkpoint is not None:
            state = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state.get("model", state), strict=True)

    def predict(self, images):
        return self.model(images)
