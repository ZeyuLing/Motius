"""Shared native GEM data/representation preparation metadata."""
import importlib
from pathlib import Path
import torch


class GEMProcessor:
    def __init__(self,bundle,codec=None):
        self.bundle=bundle
        self.args=bundle.config.pipeline_args
        self.model_cfg=bundle.config.model_config
        self.music_mask_prob=self.model_cfg.get('music_mask_prob',0.)
        self.audio_mask_prob=self.model_cfg.get('audio_mask_prob',0.)
        self.remove_fingers=self.args.get('remove_fingers',False)
        self.condition_source=bundle._unwrap(bundle.conditioner).condition_source
        self.endecoder=codec
        if codec is None:
            from omegaconf import OmegaConf
            options=OmegaConf.to_container(bundle.config.codec,resolve=True);options.pop('_target_',None)
            if bundle.VARIANT=='gem_smpl':root=bundle.body_models_root or bundle.artifact_root/'inputs/checkpoints/body_models'
            else:root=bundle.artifact_root/'inputs/soma_assets'
            self.endecoder=importlib.import_module('motius.motion.representation.'+bundle.VARIANT+'.codec').EnDecoder(body_models_root=root,**options)
        self.endecoder.to(bundle.device)
        self.endecoder.build_obs_indices_dict()
        self.use_text_encoder='text_encoder' in self.model_cfg
        self.max_text_len=self.model_cfg.get('text_encoder',{}).get('max_text_len',16)
        self.text_encoder=bundle.text_encoder
        self.tokenizer=None
        if self.text_encoder is not None:
            from transformers import T5Tokenizer
            self.tokenizer=T5Tokenizer.from_pretrained(bundle.text_encoder_path)
        stats=importlib.import_module('motius.motion.representation.'+bundle.VARIANT+'.stats_compose').cam_angvel['manual']
        self.cam_angvel_mean=torch.tensor(stats['mean'],device=bundle.device)
        self.cam_angvel_std=torch.tensor(stats['std'],device=bundle.device)
        self.normalizer_stats={key:torch.load(bundle.artifact_root/path,map_location='cpu',weights_only=False) for key,path in self.model_cfg.get('norm_attr_stats',{}).items()}

    @property
    def device(self):return self.bundle.device

    @property
    def body_model(self):
        if self.bundle.VARIANT=='gem_smpl':return self.endecoder.smplx_model
        return self.endecoder.get_soma_body_model()
