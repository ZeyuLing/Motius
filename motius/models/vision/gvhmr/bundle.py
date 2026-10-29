"""Neural-only vision bundles with explicit released checkpoint projection."""
from pathlib import Path
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from .networks import HMR2FeatureNetwork, ViTPoseNetwork


@MODEL_BUNDLES.register_module()
class GVHMRFeatureBundle(ModelBundle):
    def __init__(self, checkpoint, device="cuda"):
        super().__init__()
        self.model=HMR2FeatureNetwork()
        state=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)["state_dict"]
        needed={k:v for k,v in state.items() if k.startswith(("backbone.","smpl_head.transformer."))}
        # GVHMR uses the feature token and does not evaluate HMR2's SMPL heads,
        # discriminator, body meshes or unused mean-pose/camera buffers.
        self.model.load_state_dict(needed,strict=True)
        # Preserve the released parameter flags: ATen matmul uses requires_grad
        # when choosing folded versus batched GEMM, even inside no_grad. For the
        # HMR2 bias-free context projections, freezing here changes roundoff.
        # The pipeline disables autograd and no vision parameters enter an optimizer.
        self.model.to(device).eval()
        self._save_ckpt_modules=["model"]
        self._frozen_modules=["model"]

    def forward(self,images):
        return self.model(images)


@MODEL_BUNDLES.register_module()
class GVHMRPoseBundle(ModelBundle):
    def __init__(self, checkpoint, device="cuda"):
        super().__init__()
        self.model=ViTPoseNetwork()
        state=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)["state_dict"]
        self.model.load_state_dict(state,strict=True)
        # Preserve the released parameter flags: ATen matmul uses requires_grad
        # when choosing folded versus batched GEMM, even inside no_grad. For the
        # HMR2 bias-free context projections, freezing here changes roundoff.
        # The pipeline disables autograd and no vision parameters enter an optimizer.
        self.model.to(device).eval()
        self._save_ckpt_modules=["model"]
        self._frozen_modules=["model"]

    def forward(self,images):
        return self.model(images)



@MODEL_BUNDLES.register_module()
class GVHMRDetectorBundle(ModelBundle):
    """Load the released serialized YOLO network through explicit checkpoint I/O."""
    def __init__(self,checkpoint):
        super().__init__()
        from ultralytics.cfg import DEFAULT_CFG_DICT
        from ultralytics.nn.tasks import guess_model_task
        state=torch.load(Path(checkpoint),map_location="cpu",weights_only=False)
        network=(state.get("ema") or state["model"]).float()
        args={**DEFAULT_CFG_DICT,**state.get("train_args",{})}
        network.args={key:value for key,value in args.items() if key in DEFAULT_CFG_DICT}
        network.pt_path=str(checkpoint)
        network.task=guess_model_task(network)
        if not hasattr(network,"stride"):network.stride=torch.tensor([32.])
        for module in network.modules():
            if hasattr(module,"inplace"):module.inplace=True
            elif isinstance(module,torch.nn.Upsample) and not hasattr(module,"recompute_scale_factor"):
                module.recompute_scale_factor=None
        self.model=network.eval()
        self._save_ckpt_modules=["model"]
        self._frozen_modules=["model"]

    def forward(self,images):
        return self.model(images)
