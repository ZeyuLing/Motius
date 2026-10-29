"""GVHMR training graph expressed through the shared native neural bundle."""
from pathlib import Path
from types import SimpleNamespace
import json
import torch
import torch.nn.functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.motion.representation.gvhmr.codec import EnDecoder
from motius.motion.representation.gvhmr.camera import compute_bbox_info_bedlam,compute_transl_full_cam
from .objective import randomly_set_null_condition,compute_extra_incam_loss,compute_extra_global_loss


@TRAINERS.register_module()
class GVHMRTrainer(BaseTrainer):
    def __init__(self,bundle,*,body_model_path=None,weights=None,static_velocity_threshold=.15,**kwargs):
        super().__init__(bundle,**kwargs)
        weights={**dict(cr_j3d=500.,transl_c=1.,cr_verts=500.,j2d=1000.,verts2d=1000.,
                        transl_w=1.,static_conf_bce=1.),**(weights or {})}
        self.weights=SimpleNamespace(**weights)
        self.args=SimpleNamespace(static_conf=SimpleNamespace(vel_thr=float(static_velocity_threshold)))
        path=body_model_path or bundle.body_models_root/"smplx/SMPLX_NEUTRAL.npz"
        self.endecoder=EnDecoder(path).to(bundle.device).requires_grad_(False)
        stats=json.loads((Path(__file__).resolve().parents[2]/"configuration/gvhmr/stats.json").read_text())["camera_angular_velocity"]
        self.register_buffer("cam_angvel_mean",torch.tensor(stats["mean"],device=bundle.device),persistent=False)
        self.register_buffer("cam_angvel_std",torch.tensor(stats["std"],device=bundle.device),persistent=False)
        bundle._trainable_modules=["model"]
        bundle._save_ckpt_modules=["model"]

    def train_step(self,batch):
        def on_device(value):
            if isinstance(value,dict):return {k:on_device(v) for k,v in value.items()}
            return value.to(self.bundle.device) if isinstance(value,torch.Tensor) else value
        inputs=on_device(batch)
        conditions=dict(obs=inputs["obs"],
            f_cliffcam=compute_bbox_info_bedlam(inputs["bbx_xys"],inputs["K_fullimg"]),
            f_cam_angvel=(inputs["cam_angvel"]-self.cam_angvel_mean)/self.cam_angvel_std,
            f_imgseq=inputs["f_imgseq"])
        conditions=randomly_set_null_condition(conditions,.1)
        model_output=self.bundle.predict_motion(length=inputs["length"],**conditions)
        decoded=self.endecoder.decode(model_output["pred_x"])
        outputs=dict(model_output=model_output,decode_dict=decoded,
            pred_smpl_params_incam=dict(body_pose=decoded["body_pose"],betas=decoded["betas"],
                global_orient=decoded["global_orient"],transl=compute_transl_full_cam(
                    model_output["pred_cam"],inputs["bbx_xys"],inputs["K_fullimg"])))
        prediction=model_output["pred_x"]
        target=self.endecoder.encode(inputs)
        mask=inputs["mask"]["valid"][:,:,None].expand(-1,-1,prediction.size(2)).clone()
        mask[inputs["mask"]["spv_incam_only"],:,142:]=False
        simple=(F.mse_loss(prediction,target,reduction="none")*mask).mean()
        total=simple
        metrics={"gvhmr_simple":simple.detach()}
        for fn in [compute_extra_incam_loss,compute_extra_global_loss]:
            extra,logs=fn(inputs,outputs,self)
            total=total+extra
            metrics.update({"gvhmr_"+key:value.detach() for key,value in logs.items()})
        return {"loss":total,**metrics}

    @torch.no_grad()
    def val_step(self,batch):
        return {"val_"+key:value.detach() for key,value in self.train_step(batch).items()}
