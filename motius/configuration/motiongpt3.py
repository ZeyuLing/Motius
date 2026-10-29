"""Read architectural values from released MotionGPT3 artifacts."""
from pathlib import Path
from omegaconf import OmegaConf


def architecture(artifact,config_path=None):
    artifact=Path(artifact)
    cfg=OmegaConf.merge(OmegaConf.load(artifact/'configs/default.yaml'),OmegaConf.load(config_path or artifact/'configs/test.yaml'))
    for folder in ['lm','vae']:
        for path in sorted((artifact/'configs'/folder).glob('*.yaml')):
            OmegaConf.update(cfg,f'{folder}.{path.stem}',OmegaConf.load(path),force_add=True)
    cfg=OmegaConf.merge(cfg,OmegaConf.load(artifact/'configs/assets.yaml'))
    # Only neural subtrees are resolved; application targets are never imported.
    lm=OmegaConf.to_container(cfg.model.params.lm.params,resolve=True)
    vae=OmegaConf.to_container(cfg.model.params.motion_vae.params,resolve=True)
    vae['datatype']='humanml3d'
    defaults=dict(max_length=192,framerate=20.,down_t=4,predict_ratio=.2,inbetween_ratio=.25,
        quota_ratio=.5,noise_density=.15,mean_noise_span_length=3,stage='lm_pretrain')
    options={**defaults,**{k:v for k,v in lm.items() if k not in ['diffhead','model_path','ablation','model_type']}}
    options['multi_hidden']=lm['diffhead']['params'].get('multi_hidden',False)
    options['with_hid_norm']=options['multi_hidden'] and options.get('with_hid_norm',True)
    return options,vae,lm['diffhead']['params']


class TokenLayout:
    """Static neural vocabulary boundaries; contains no text tokenizer."""
    def __init__(self,base_vocab_size):
        self.bos_token_id=self.eos_token_id=self.pad_token_id=base_vocab_size-1
        self.ids={word:base_vocab_size+i for i,word in enumerate([
            '<start_of_motion>','<end_of_motion>','<masked_motion>','<pad_motion>',
            '<motion_latent_holder_output>','<motion_latent_holder>'])}
    def convert_tokens_to_ids(self,tokens):
        return [self.ids[word] for word in tokens]
