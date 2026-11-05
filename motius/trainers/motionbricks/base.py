"""Native state shared by MotionBricks' three independently trained stages."""
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from motius.trainers.base_trainer import BaseTrainer
from motius.configuration.motionbricks import source_config,build_representation
from motius.motion.conditioning.motionbricks_codec import MotionBricksCodec


class _MotionBricksStageTrainer(BaseTrainer):
    STAGE=None

    def __init__(self,bundle,representation=None,training_options=None,**kwargs):
        super().__init__(bundle,**kwargs)
        if training_options is None:training_options=source_config(bundle.artifact_root,self.STAGE).model.args
        self._args=OmegaConf.create(training_options)
        self.motion_rep=representation if representation is not None else build_representation(bundle.artifact_root,self.STAGE)
        self.motion_rep.to(bundle.device)
        self.global_motion_rep=self.motion_rep.dual_rep.global_motion_rep
        self.local_motion_rep=self.motion_rep.dual_rep.local_motion_rep
        self.codec=MotionBricksCodec(bundle,self.local_motion_rep)
        self.root_net=None
        self.DEFAULT_NUM_JOINTS=self.motion_rep.num_joints
        self._reported={}
        trained=['encoder','decoder','quantizer'] if self.STAGE=='vqvae' else [self.STAGE]
        bundle._trainable_modules=[name for name in trained if next(getattr(bundle,name).parameters(),None) is not None]
        bundle._save_ckpt_modules=['encoder','decoder','quantizer','pose','root']
        bundle._frozen_modules=[name for name in bundle._save_ckpt_modules if name not in trained]
        bundle.train()

    @property
    def args(self):return self._args

    @property
    def network(self):
        network=getattr(self.bundle,self.STAGE)
        return getattr(network,'module',network)

    def _network_forward(self,*args,**kwargs):
        if self.STAGE=='pose':return self.bundle.predict_pose(*args,**kwargs)
        target=kwargs.pop('groundtruth_num_tokens')
        return self.bundle.predict_root(*args,**kwargs,chosen_token=target.reshape(-1)-self._args.min_tokens)

    def _reconstruct_pose(self,*args,**kwargs):
        inputs=self.codec.reconstruction_inputs(*args,**kwargs)
        result=self.bundle.quantize_features(self.bundle.encode_features(inputs.pop('features')))
        indices=result['indices']
        if indices.ndim==2:indices=indices[:,:,None]
        if self.bundle.options.tokenizer.get('calculate_per_head_perplexity',True):
            perplexity=0
            for head in range(self.codec.num_heads):
                values=indices[:,:,head].reshape(-1)
                onehot=torch.zeros(self.codec.codes_per_head,values.numel(),device=values.device)
                onehot.scatter_(0,values.view(1,-1),1)
                count=onehot.sum(-1);prob=count/count.sum()
                perplexity=perplexity+torch.exp(-torch.sum(prob*torch.log(prob+1e-7)))
            result['perplexity']=perplexity/self.codec.num_heads
        else:
            multipliers=self.codec.codes_per_head**torch.arange(self.codec.num_heads,device=indices.device)[None,None,:]
            values=(indices*multipliers).sum(-1).reshape(-1)
            onehot=torch.zeros(int(self.bundle.options.tokenizer.nb_code),values.numel(),device=values.device)
            onehot.scatter_(0,values.view(1,-1),1)
            count=onehot.sum(-1);prob=count/count.sum()
            result['perplexity']=torch.exp(-torch.sum(prob*torch.log(prob+1e-7)))
        result['l_commit']=F.mse_loss(result['encoder_features'],result['code_vectors'])
        result['recon_state']=self.bundle.decode_features(self.bundle.project_quantized(result),**inputs)
        return result

    def log(self,key,value,**kwargs):self._reported[key.removeprefix('loss/train_')]=value

    def train_step(self,batch):
        self._reported={}
        loss=self.training_step(dict(batch),0)
        return dict(self._reported,loss=loss)

    @torch.no_grad()
    def val_step(self,batch):
        if self.STAGE=='vqvae':values=self.train_step(batch)
        else:
            metadata={}
            self.training_step(dict(batch),0,use_outside_training=True,meta_info=metadata)
            values=metadata['losses']
        return {'val_'+key:value for key,value in values.items()}
