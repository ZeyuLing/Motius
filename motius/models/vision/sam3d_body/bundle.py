"""SAM 3D Body network construction and checkpoint loading."""

from pathlib import Path

import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from motius.utils.sam3d.config import get_config
from motius.utils.sam3d.checkpoint import load_state_dict
from .meta_arch.sam3d_body import SAM3DBody


@MODEL_BUNDLES.register_module()
class SAM3DBodyBundle(ModelBundle):
    def __init__(self, checkpoint, mhr_path, config_path=None, *, low_cpu_mem_usage=True, precision="source"):
        super().__init__()
        checkpoint = Path(checkpoint)
        if config_path:
            path = Path(config_path)
        else:
            path = checkpoint.parent / "model_config.yaml"
            if not path.is_file():
                path = checkpoint.parent.parent / "model_config.yaml"
        self.config = get_config(str(path))
        self.config.defrost()
        self.config.MODEL.MHR_HEAD.MHR_MODEL_PATH = str(mhr_path)
        if precision not in ("source", "float16", "float32"):
            raise ValueError("precision must be source, float16 or float32")
        if precision == "float32":
            self.config.TRAIN.USE_FP16 = False
        elif precision == "float16":
            self.config.TRAIN.USE_FP16 = True
            self.config.TRAIN.FP16_TYPE = "float16"
        self.config.freeze()
        # The released DINO backbone is completely supplied by the checkpoint.
        # Keep MHR and other initialized metadata on CPU, and avoid constructing
        # hundreds of millions of disposable backbone weights.
        meta_backbone = low_cpu_mem_usage and self.config.MODEL.BACKBONE.TYPE.startswith('dinov3_')
        self.model = SAM3DBody(self.config, init_backbone_device='meta' if meta_backbone else None)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        # Match the released loader: face-related keys can be absent from the
        # configured body-only network. The same loader reports incompatibilities.
        state = state.get("state_dict", state)
        self.model.checkpoint_backbone_dtypes = sorted({str(value.dtype) for key, value in state.items()
            if key.startswith("backbone.") and torch.is_tensor(value)})
        if precision == "float16":
            limit = torch.finfo(torch.float16).max
            for key, value in state.items():
                if key.startswith("backbone.") and torch.is_tensor(value) and value.is_floating_point() and value.numel():
                    if not torch.isfinite(value).all() or value.abs().max().float() > limit:
                        raise ValueError(f"SAM backbone checkpoint tensor {key} cannot be represented in FP16; select upstream or FP32 precision")
        if meta_backbone:
            targets = self.model.backbone.state_dict()
            backbone = {key.removeprefix('backbone.'): value for key, value in state.items()
                        if key.startswith('backbone.')}
            # Released SAM omits DINO's mask token; upstream initializes it to
            # zero and unmasked inference includes 0 * mask_token in its CLS.
            if 'encoder.mask_token' in targets and 'encoder.mask_token' not in backbone:
                backbone['encoder.mask_token'] = torch.zeros_like(targets['encoder.mask_token'], device='cpu')
            backbone = {key: value.to(dtype=targets[key].dtype) if key in targets else value
                        for key, value in backbone.items()}
            self.model.backbone.load_state_dict(backbone, strict=True, assign=True)
            if any(t.is_meta for t in (*self.model.backbone.parameters(), *self.model.backbone.buffers())):
                raise RuntimeError('SAM checkpoint left unmaterialized backbone tensors')
        load_state_dict(self.model, state, strict=False)
        self._trainable_modules = ["model"]
        self._save_ckpt_modules = ["model"]
        self._module_checkpoint_formats = {"model": "full"}

    def predict(self, batch, decoder_type="body"):
        self.model._initialize_batch(batch)
        return self.model.forward_step(batch, decoder_type=decoder_type)
