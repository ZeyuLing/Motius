"""Native MotionGPT3 neural bundle with strict released checkpoint loading."""
from pathlib import Path
import shutil
import numpy as np
import torch
from transformers import GPT2Config
from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from motius.configuration.motiongpt3 import architecture,TokenLayout
from .networks import MotionGPT3LanguageNetwork
from .vae import MldVae


@MODEL_BUNDLES.register_module()
class MotionGPT3Bundle(ModelBundle):
    def __init__(self,artifact_dir=None,cfg=None,checkpoint=None,mean_path=None,std_path=None,
                 guidance_scale=7.5,runtime_dir=None,device='cuda',**kwargs):
        super().__init__()
        artifact=Path(artifact_dir or 'checkpoints/motiongpt3').expanduser().resolve()
        self.artifact_root=artifact;self.tokenizer_path=artifact/'deps/mot-gpt2'
        self.options,vae_config,self.diffusion_config=architecture(artifact,cfg)
        self.diffusion_options={k:v for k,v in self.diffusion_config.items() if k in ['noise_schedule','use_kl','learn_sigma','sigma_small']}
        self.guidance_scale=float(guidance_scale)
        state=torch.load(checkpoint or artifact/'motiongpt3.ckpt',map_location='cpu',weights_only=True,mmap=True)['state_dict']
        import json
        from motius import configuration
        metadata=json.loads((Path(configuration.__file__).parent/'motiongpt3_text_parameters.json').read_text())
        config=GPT2Config.from_pretrained(self.tokenizer_path,local_files_only=True)
        layout=TokenLayout(config.vocab_size)
        if config.n_layer!=metadata['layers']:raise ValueError('This native text-freezing layout requires the released 12-layer architecture')
        config.vocab_size=state['lm.language_model.transformer.wte.weight'].shape[0]
        self.vae=MldVae(**vae_config)
        self.lm=MotionGPT3LanguageNetwork(config,layout,self.options,self.diffusion_config,self.vae.latent_dim,self.vae.latent_size,set(metadata['parameter_names']))
        self.diffusion_head=self.lm.diffloss
        del self.lm.diffloss
        if hasattr(self.lm,'fake_latent'):
            self.diffusion_head.register_parameter('fake_latent',self.lm.fake_latent)
            del self.lm.fake_latent
        excluded=[k for k in state if not k.startswith(('lm.','vae.'))]
        if any(not k.startswith(('metrics.','_losses.')) for k in excluded):raise RuntimeError('Unknown released checkpoint namespace')
        lm_state={k[3:]:v for k,v in state.items() if k.startswith('lm.') and not k.startswith('lm.diffloss.') and k!='lm.fake_latent'}
        self.lm.load_state_dict(lm_state,strict=True)
        diffusion_state={k[len('lm.diffloss.'):]:v for k,v in state.items() if k.startswith('lm.diffloss.')}
        if 'lm.fake_latent' in state:diffusion_state['fake_latent']=state['lm.fake_latent']
        self.diffusion_head.load_state_dict(diffusion_state,strict=True)
        self.vae.load_state_dict({k[4:]:v for k,v in state.items() if k.startswith('vae.')},strict=True)
        self.register_buffer('mean',torch.from_numpy(np.load(mean_path or artifact/'assets/meta/mean.npy')).float())
        self.register_buffer('std',torch.from_numpy(np.load(std_path or artifact/'assets/meta/std.npy')).float())
        self.load_report={'strict_neural_keys':sum(k.startswith(('lm.','vae.')) for k in state),'excluded_non_neural_keys':excluded}
        self.to_device(device);self.eval()

    @staticmethod
    def unwrap_network(model):
        while hasattr(model,'module'):model=model.module
        return model

    @property
    def device(self):return self.mean.device

    def to_device(self,device):
        return self.to(torch.device(device if torch.cuda.is_available() or str(device)=='cpu' else 'cpu'))

    def predict_language(self,*args,**kwargs):return self.lm(*args,**kwargs)
    def project_motion(self,*args,**kwargs):return self.unwrap_network(self.lm).motion_und_head(*args,**kwargs)
    def predict_diffusion(self,*args,**kwargs):return self.diffusion_head(*args,**kwargs)
    def forward(self,*args,**kwargs):return self.predict_language(*args,**kwargs)

    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path,**kwargs):
        path=Path(pretrained_model_name_or_path)
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path=Path(snapshot_download(str(pretrained_model_name_or_path)))
        return cls(artifact_dir=str(path),**kwargs)

    def save_pretrained(self,save_directory,**kwargs):
        output=Path(save_directory);output.mkdir(exist_ok=True,parents=True)
        for directory in ['configs','deps']:
            shutil.copytree(self.artifact_root/directory,output/directory,dirs_exist_ok=True,ignore=shutil.ignore_patterns('model_state_dict.pth'))
        assets=output/'assets/meta';assets.mkdir(exist_ok=True,parents=True)
        np.save(assets/'mean.npy',self.mean.detach().cpu().numpy());np.save(assets/'std.npy',self.std.detach().cpu().numpy())
        state={name+'.'+key:value.detach().cpu() for name in ['lm','vae'] for key,value in self.unwrap_network(getattr(self,name)).state_dict().items()}
        state.update({('lm.fake_latent' if key=='fake_latent' else 'lm.diffloss.'+key):value.detach().cpu() for key,value in self.unwrap_network(self.diffusion_head).state_dict().items()})
        torch.save({'state_dict':state},output/'motiongpt3.ckpt')
        import json
        (output/'model_index.json').write_text(json.dumps({'_class_name':'MotionGPT3Pipeline','_library_name':'motius','model_type':'motiongpt3','bundle_class':'motius.models.motiongpt3.MotionGPT3Bundle','pipeline_class':'motius.pipelines.motiongpt3.MotionGPT3Pipeline'},indent=2)+'\n')
        return str(output)
