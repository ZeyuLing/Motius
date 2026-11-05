"""Feature selection and code packing around MotionBricks' neural tokenizer."""
import torch
from .motionbricks import extract_feature_from_motion_rep,convert_sparse_cond_to_dense_cond_if_needed


class MotionBricksCodec:
    def __init__(self,bundle,motion_rep):
        self.bundle=bundle;self.motion_rep=motion_rep
        self.feature_mode=list(bundle.options.tokenizer.feature_mode)
        self.encoder_input_feature_mode,self.decoder_input_feature_mode,self.decoder_target_cond_feature_mode,self.decoder_external_cond_feature_mode=self.feature_mode
        self.num_heads=int(bundle.options.tokenizer.num_heads)
        self.codes_per_head=int(round(int(bundle.options.tokenizer.nb_code)**(1/self.num_heads)))

    def extract_feature(self,x,feature):return extract_feature_from_motion_rep(x,self.motion_rep,feature)

    def encode_into_idx(self,x,fetch_overall_indices=True):
        codes=self.bundle.encode_tokens(self.extract_feature(x,self.encoder_input_feature_mode))
        if not fetch_overall_indices:return codes
        multipliers=self.codes_per_head**torch.arange(self.num_heads,device=codes.device)[None,None,:]
        return (codes*multipliers).sum(-1)

    def reconstruction_inputs(self,x,target_cond,has_target_cond=None,external_cond=None):
        if target_cond is not None:
            target_cond,has_target_cond=convert_sparse_cond_to_dense_cond_if_needed(target_cond,has_target_cond,x.shape[1])
            target_cond=self.extract_feature(target_cond,self.decoder_target_cond_feature_mode)
        return dict(features=self.extract_feature(x,self.encoder_input_feature_mode),target_cond=target_cond,has_target_cond=has_target_cond,external_cond=external_cond)
