"""Shared checkpoint construction for GEM's two independently wrapped neural roots."""
from pathlib import Path
import importlib,json,shutil
import torch
from motius.configuration.gem import load_gem_config
from motius.models.base_model_bundle import ModelBundle


class GEMModelBundle(ModelBundle):
    VARIANT=None
    CHECKPOINT=None

    def __init__(self,artifact_root,*,device=None,export_device='cuda',config=None,
                 body_models_root=None,manifest=None,text_encoder_path=None):
        super().__init__()
        self.artifact_root=Path(artifact_root).expanduser().resolve()
        self.body_models_root=None if body_models_root is None else Path(body_models_root).expanduser().resolve()
        saved=self.artifact_root/'gem_architecture.json'
        if config is None and saved.exists():config=json.loads(saved.read_text())
        self.config=load_gem_config(self.VARIANT,config)
        from omegaconf import OmegaConf
        self.manifest=dict(manifest or {})
        metadata=self.artifact_root/(self.VARIANT+'_config.json')
        if not self.manifest and metadata.exists():self.manifest=json.loads(metadata.read_text())
        self.checkpoint_path=self.artifact_root/'inputs/pretrained'/self.CHECKPOINT
        state=torch.load(self.checkpoint_path,map_location='cpu',weights_only=False,mmap=True)['state_dict']
        options=OmegaConf.to_container(self.config.denoiser,resolve=True);options.pop('_target_',None)
        conditioner=importlib.import_module('motius.models.'+self.VARIANT+'.conditioning').ConditionEncoder
        denoiser=importlib.import_module('motius.models.'+self.VARIANT+'.denoiser').NetworkEncoderRoPE
        if self.VARIANT=='gem_x':
            options.update(fixed_scale_mean=state['pipeline.denoiser3d.denoiser.scale_mean'],fixed_scale_components=state['pipeline.denoiser3d.denoiser.scale_comps'])
        self.conditioner=conditioner(self.config.pipeline_args,self.config.model_config,self.config.denoiser,33 if self.VARIANT=='gem_x' else 17)
        self.denoiser=denoiser(**options)
        prefix='pipeline.denoiser3d.denoiser.'
        self.conditioner.load_state_dict({k:v for k,v in state.items() if not k.startswith(prefix)},strict=True)
        self.denoiser.load_state_dict({k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)},strict=True)
        self.text_encoder=None
        self.text_encoder_path=Path(text_encoder_path) if text_encoder_path else self.artifact_root/'text_encoder'
        if self.text_encoder_path.is_dir():
            from transformers import T5EncoderModel
            self.text_encoder=T5EncoderModel.from_pretrained(self.text_encoder_path).requires_grad_(False).eval()
        resolved=device or export_device
        if str(resolved)=='cuda' and not torch.cuda.is_available():resolved='cpu'
        self.to(resolved).eval()
        self._save_ckpt_modules=['conditioner','denoiser']

    @property
    def device(self):return next(self.denoiser.parameters()).device

    def encode_conditions(self,batch,existence,visible,first_k_frames=None):
        return self.conditioner(batch,existence,visible,first_k_frames)

    def predict_denoise(self,x,timesteps,y=None,**kwargs):
        return self.denoiser(x,timesteps,y=y,**kwargs)

    def encode_text_tokens(self,input_ids,attention_mask):
        if self.text_encoder is None:raise RuntimeError('No T5 encoder checkpoint is configured')
        return self.text_encoder(input_ids=input_ids,attention_mask=attention_mask).last_hidden_state

    def forward(self,x,timesteps,y=None,**kwargs):
        return self.predict_denoise(x,timesteps,y=y,**kwargs)

    @staticmethod
    def _unwrap(module):return module.module if hasattr(module,'module') else module

    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path,*,revision=None,cache_dir=None,token=None,local_files_only=False,**kwargs):
        path=Path(pretrained_model_name_or_path).expanduser()
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path=Path(snapshot_download(pretrained_model_name_or_path,revision=revision,cache_dir=cache_dir,token=token,local_files_only=local_files_only))
        return cls(artifact_root=path,**kwargs)

    def save_pretrained(self,save_directory,**kwargs):
        from omegaconf import OmegaConf
        target=Path(save_directory);target.mkdir(parents=True,exist_ok=True)
        for relative in self.manifest.get('assets',[]):
            source=self.artifact_root/relative
            destination=target/relative
            if source==self.checkpoint_path:continue
            if not source.is_file():raise FileNotFoundError(source)
            destination.parent.mkdir(parents=True,exist_ok=True)
            if source.resolve()!=destination.resolve():shutil.copy2(source,destination)
        state=dict(self._unwrap(self.conditioner).state_dict())
        state.update({'pipeline.denoiser3d.denoiser.'+k:v for k,v in self._unwrap(self.denoiser).state_dict().items()})
        destination=target/'inputs/pretrained'/self.CHECKPOINT;destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_suffix(destination.suffix+'.tmp')
        torch.save({'state_dict':{k:v.detach().cpu() for k,v in state.items()}},temporary)
        temporary.replace(destination)
        (target/'gem_architecture.json').write_text(json.dumps(OmegaConf.to_container(self.config,resolve=True),indent=2)+'\n')
        from motius.utils.hashing import sha256_file
        metadata=dict(self.manifest,artifact_format='motius-'+self.VARIANT.replace('_','-')+'-v2')
        metadata['sha256']={relative:sha256_file(target/relative) for relative in metadata.get('assets',[])}
        (target/(self.VARIANT+'_config.json')).write_text(json.dumps(metadata,indent=2)+'\n')
        (target/'model_index.json').write_text(json.dumps(dict(model_type=self.VARIANT,bundle_class=type(self).__module__+'.'+type(self).__name__,pipeline_class='motius.pipelines.'+self.VARIANT+'.'+('GemSmplPipeline' if self.VARIANT=='gem_smpl' else 'GemXPipeline')),indent=2)+'\n')
        if self.text_encoder is not None:self._unwrap(self.text_encoder).save_pretrained(target/'text_encoder')
        return str(target)
