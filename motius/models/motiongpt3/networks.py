"""MotionGPT3 neural language, latent projection and diffusion networks."""
import torch
from torch import nn
from .language_model import MoTGPT2LMHeadModel
from .diffusion_head import DiffLoss

class MotionUndHead(nn.Module):

    def __init__(self, input_dim, output_dim, projector_type='linear', depth=1, **kwargs):
        super().__init__()
        if projector_type == 'identity':
            modules = nn.Identity()
        elif projector_type == 'linear':
            modules = nn.Linear(input_dim, output_dim)
        elif projector_type == 'mlp_gelu':
            mlp_depth = depth
            modules = [nn.Linear(input_dim, output_dim)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(output_dim, output_dim))
            modules = nn.Sequential(*modules)
        else:
            raise ValueError(f'Unknown projector type: {projector_type}')
        self.layers = modules

    def forward(self, motion_tokens):
        if len(motion_tokens.shape) == 3:
            bs, latent_dim, latent_channel = motion_tokens.shape
            motion_tokens = motion_tokens.permute(1, 0, 2)
            motion_tokens = motion_tokens.reshape(bs, -1)
        motion_embedding = self.layers(motion_tokens)
        return motion_embedding

class MotionGPT3LanguageNetwork(nn.Module):
    def __init__(self,config,layout,options,diffusion_config,vae_channels,vae_size,base_keys):
        super().__init__()
        self.language_model=MoTGPT2LMHeadModel(config,motion_codebook_size=6,
            mot_factor=options.get('mot_factor',1.),attention_mode=options.get('attention_mode','all'))
        self.language_model.transformer._attn_implementation='sdpa'
        self.language_model.config.mot_lm_dim=6
        self.language_model.config.text_vocab_size=config.vocab_size
        self.language_model.set_modality_info(layout)
        text_trained='finetune' in options['stage']
        for name,parameter in self.language_model.named_parameters():
            parameter.requires_grad_(text_trained if name in base_keys or 'fn.0' in name else True)
        for module in [self.language_model.pre_processors[0],self.language_model.post_processors[0]]:
            module.weight.requires_grad_(True)
        dimension=self.language_model.mot_embed_dim
        self.motion_und_head=MotionUndHead(vae_channels,dimension*options['holder_num_in_input'],projector_type='linear')
        multi_hidden=diffusion_config.get('multi_hidden',False)
        if multi_hidden and options['with_hid_norm']:
            self.diffusion_pos_embed_learned=nn.Parameter(torch.zeros(1,options['motion_holder_repeat'],dimension))
            torch.nn.init.normal_(self.diffusion_pos_embed_learned,std=.02)
            self.norm_layer=nn.LayerNorm(dimension,eps=1e-6)
        d={**diffusion_config,'target_channels':vae_channels,'target_size':vae_size,
           'z_channels':dimension if multi_hidden else dimension*options['motion_holder_repeat']}
        self.diffloss=DiffLoss(**d)
        if options['guidance_scale']>1.:
            mode=options['fake_latent_mode']
            value=torch.zeros(options['motion_holder_repeat'],dimension)
            if mode=='learnable_rand':torch.nn.init.normal_(value,std=.02)
            if mode in ['learnable_zero','learnable_rand']:self.fake_latent=nn.Parameter(value)
            elif mode=='all_zero':self.register_buffer('fake_latent',value,persistent=False)
            else:raise ValueError(mode)

    def forward(self,*args,motion_tokens=None,input_is_motion=None,motion_num_in_input=None,
                output_is_motion=None,return_diffusion_hidden=False,**kwargs):
        if motion_tokens is None:return self.language_model(*args,**kwargs)
        inputs_embeds=self.language_model.get_embeddings_from_ids(kwargs.pop('input_ids'),kwargs['type_ids'])
        for i,tokens in enumerate(motion_tokens):
            if tokens is None:continue
            projected=self.motion_und_head(tokens)
            parts=projected[:motion_num_in_input[i]].split(self.language_model.mot_embed_dim,dim=-1)
            inputs_embeds[1][i,input_is_motion[i],:]=torch.cat(parts,dim=0)
        output=self.language_model(*args,inputs_embeds=inputs_embeds,**kwargs)
        if not return_diffusion_hidden:return output
        hidden=output.hidden_states[-1][1]
        selected=[hidden[i:i+1,mask,:] for i,mask in enumerate(output_is_motion) if mask.sum()>0]
        selected=torch.cat(selected) if selected else None
        if selected is not None and hasattr(self,'norm_layer'):
            selected=self.norm_layer(selected)+self.diffusion_pos_embed_learned
        return output,selected
