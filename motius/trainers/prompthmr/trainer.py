"""Published PromptHMR temporal training graph using the shared video network."""
from pathlib import Path
from omegaconf import OmegaConf
import torch
from torch.nn import functional as F
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.motion.representation.prompthmr.training_codec import PromptHMRTrainingCodec
from motius.motion.representation.gvhmr.camera import compute_bbox_info_bedlam,compute_transl_full_cam
from motius.motion.representation.prompthmr.camera import compute_transl_full_cam_prompthmr,compute_transl_full_cam_metric3d
from .objective import randomly_set_null_condition,compute_extra_incam_loss,compute_extra_global_loss


@TRAINERS.register_module()
class PromptHMRVideoTrainer(BaseTrainer):
    """Train the released temporal head on prepared, body-supervised sequences.

    The image model's training code/dataset is not published by PromptHMR.
    This trainer implements the independently published video objective only.
    """
    def __init__(self,bundle,*,body_model_path,weights=None,static_velocity_threshold=.15,**kwargs):
        super().__init__(bundle,**kwargs)
        self.weights=OmegaConf.create({**dict(cr_j3d=500.,transl_c=1.,cr_verts=500.,j2d=1000.,verts2d=1000.,
            transl_w=1.,static_conf_bce=1.),**(weights or {})})
        self.args=OmegaConf.create(dict(camt_type=bundle.video_config.get('camt_type','cliff'),
            static_conf=dict(vel_thr=static_velocity_threshold)))
        self.endecoder=PromptHMRTrainingCodec(bundle.device,body_model_path)
        import json
        stats=json.loads((Path(__file__).resolve().parents[2]/'configuration/prompthmr/stats.json').read_text())['camera_angular_velocity']
        self.register_buffer('cam_angvel_mean',torch.tensor(stats['mean'],device=bundle.device),persistent=False)
        self.register_buffer('cam_angvel_std',torch.tensor(stats['std'],device=bundle.device),persistent=False)
        bundle._trainable_modules=['video']
        bundle._frozen_modules=[name for name in bundle._save_ckpt_modules if name!='video']

    def train_step(self,batch):
        def to_device(value):
            if isinstance(value,dict):return {key:to_device(item) for key,item in value.items()}
            return value.to(self.bundle.device) if isinstance(value,torch.Tensor) else value
        inputs=to_device(batch)
        conditions=dict(obs=inputs['obs'],f_cliffcam=compute_bbox_info_bedlam(inputs['bbx_xys'],inputs['K_fullimg']),
            f_prhmrcam=inputs['f_imgseq'][:,:,1],f_cam_angvel=(inputs['cam_angvel']-self.cam_angvel_mean)/self.cam_angvel_std,
            f_imgseq=inputs['f_imgseq'][:,:,0])
        conditions=randomly_set_null_condition(conditions,.1)
        model=self.bundle.predict_video(length=inputs['length'],cam_int=inputs['K_fullimg'],**conditions)
        decoded=self.endecoder.decode(model['pred_x'])
        decode_cam=dict(cliff=compute_transl_full_cam,prompthmr=compute_transl_full_cam_prompthmr,metric3d=compute_transl_full_cam_metric3d)
        translation=decode_cam[self.args.camt_type](model['pred_cam'],inputs['bbx_xys'],inputs['K_fullimg'])
        outputs=dict(model_output=model,decode_dict=decoded,pred_smpl_params_incam=dict(body_pose=decoded['body_pose'],
            betas=decoded['betas'],global_orient=decoded['global_orient'],transl=translation))
        prediction=model['pred_x'];target=self.endecoder.encode(inputs)
        simple=F.mse_loss(prediction,target,reduction='none')
        mask=inputs['mask']['valid'][:,:,None].expand(-1,-1,prediction.size(2)).clone()
        mask[inputs['mask']['spv_incam_only'],:,142:]=False
        simple=(simple*mask).mean();total=simple
        metrics=dict(simple_loss=simple)
        inputs['nb_epoch']=batch.get('nb_epoch',0)
        for objective in [compute_extra_incam_loss,compute_extra_global_loss]:
            extra,logs=objective(inputs,outputs,self);total=total+extra;metrics.update(logs)
        return dict(loss=total,**{name:value.detach() for name,value in metrics.items()})

    @torch.no_grad()
    def val_step(self,batch):return {'val_'+name:value.detach() for name,value in self.train_step(batch).items()}
