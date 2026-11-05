"""PromptHMR image and temporal neural components with strict checkpoint import."""
from pathlib import Path
import json
import torch
from motius.utils.torch_checkpoint import load_torch_checkpoint
from motius.models.base_model_bundle import ModelBundle


from motius.registry import MODEL_BUNDLES

@MODEL_BUNDLES.register_module()
class PromptHMRBundle(ModelBundle):
    def __init__(self, artifact_root=None, *, image_config=None, video_config=None, device='cuda'):
        super().__init__()
        from .components.image_encoder import ImageEncoder
        from .components.camera_embed import CameraEncoder
        from .components.prompt_encoder import PromptEncoder
        from .components.smpl_decoder import SMPLDecoder
        from .components.twoway_transformer import TwoWayTransformer
        from .text import ClipTextEncoder
        from .video import NetworkEncoderRoPE
        self.artifact_root=None if artifact_root is None else Path(artifact_root).expanduser().resolve()
        from omegaconf import OmegaConf
        root=self.artifact_root
        native=None if root is None else root/'prompthmr_neural.json'
        if native is not None and native.is_file():
            config=json.loads(native.read_text());image_config=config['image'];video_config=config['video']
        elif root is not None:
            if image_config is None:image_config=OmegaConf.to_container(OmegaConf.load(root/'config.yaml').MODEL,resolve=True)
            if video_config is None:
                cfg=OmegaConf.load(root/'video/prhmr_release_002.yaml')
                video_config=OmegaConf.to_container(cfg.network,resolve=True)
                video_config['camt_type']=cfg.pipeline.args.get('camt_type','cliff')
        if image_config is None or video_config is None:raise ValueError('Provide both image and video architectures or an artifact root')
        self.image_config=dict(image_config);self.video_config=dict(video_config)
        c=self.image_config;dim=c['EMBED_DIM'];size=c['IMG_SIZE'];grid=size//c['PATCH_SIZE']
        self.image_encoder=ImageEncoder(c['BACKBONE'],dim)
        self.cam_encoder=CameraEncoder(patch_size=c['PATCH_SIZE'],img_size=size)
        self.text_encoder=ClipTextEncoder()
        self.prompt_encoder=PromptEncoder(embed_dim=dim,input_image_size=(size,size),image_embedding_size=(grid,grid),mask_prompt=c.get('MASK_PROMPT',False))
        transformer=TwoWayTransformer(depth=c['DECODER_DEPTH'],embedding_dim=dim,mlp_dim=c['MLP_DIM'],num_heads=8,attention_block=c.get('TRANSFORMER','oneway'))
        self.smpl_decoder=SMPLDecoder(transformer=transformer,transformer_dim=dim,smpl_head_hidden_dim=dim,smpl_head_depth=c['REGRESSOR_DEPTH'],inverse_depth=c.get('INVERSE_DEPTH',False))
        self.video=NetworkEncoderRoPE(**self.video_config)
        self._save_ckpt_modules=['image_encoder','cam_encoder','text_encoder','prompt_encoder','smpl_decoder','video']
        self._trainable_modules=list(self._save_ckpt_modules)
        if root is not None:
            if native.is_file():self.load_state_dict(load_torch_checkpoint(root/'prompthmr_networks.pt',map_location='cpu',weights_only=True,mmap=True),strict=True)
            else:self._load_original(root)
        self.to(device).eval()

    def _load_original(self,root):
        state=load_torch_checkpoint(root/'checkpoint.ckpt',map_location='cpu',weights_only=True,mmap=True)['state_dict']
        for name in self._save_ckpt_modules:
            if name=='video':continue
            prefix='prompt_encoder.clip_encoder.' if name=='text_encoder' else name+'.'
            component={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
            if name=='prompt_encoder':component={key:value for key,value in component.items() if not key.startswith('clip_encoder.')}
            getattr(self,name).load_state_dict(component,strict=True)
        allowed=('image_encoder.','cam_encoder.','prompt_encoder.','smpl_decoder.','smpl.','smplx.','smplx2smpl')
        unexpected=[key for key in state if not key.startswith(allowed)]
        if unexpected:raise ValueError('Unexpected image checkpoint components: '+str(unexpected[:8]))
        checkpoint=root/'video/phmr_b1b2.ckpt'
        if not checkpoint.is_file():checkpoint=root/'video/prhmr_release_002.ckpt'
        state=load_torch_checkpoint(checkpoint,map_location='cpu',weights_only=False,mmap=True)['state_dict']
        prefix='pipeline.denoiser3d.'
        self.video.load_state_dict({key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)},strict=True)
        extra=[key for key in state if not key.startswith((prefix,'pipeline.endecoder.','smplx.'))]
        if extra:raise ValueError('Unexpected temporal checkpoint components: '+str(extra[:8]))

    @property
    def device(self):return next(self.image_encoder.parameters()).device

    def encode_images(self,images,intrinsics):return self.cam_encoder(self.image_encoder(images),intrinsics)
    def encode_text_tokens(self,tokens):return self.text_encoder(tokens)
    def encode_prompts(self,*args,**kwargs):return self.prompt_encoder(*args,**kwargs)
    def decode_image(self,*args,**kwargs):return self.smpl_decoder(*args,**kwargs)
    def predict_video(self,*args,**kwargs):return self.video(*args,**kwargs)

    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path,*,revision=None,cache_dir=None,token=None,local_files_only=False,**kwargs):
        path=Path(pretrained_model_name_or_path).expanduser()
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path=Path(snapshot_download(str(pretrained_model_name_or_path),revision=revision,cache_dir=cache_dir,token=token,local_files_only=local_files_only))
        return cls(path,**kwargs)

    def save_pretrained(self,save_directory,**kwargs):
        root=Path(save_directory);root.mkdir(parents=True,exist_ok=True)
        state={name+'.'+key:value.detach().cpu() for name in self._save_ckpt_modules for key,value in getattr(getattr(self,name),'module',getattr(self,name)).state_dict().items()}
        torch.save(state,root/'prompthmr_networks.pt')
        (root/'prompthmr_neural.json').write_text(json.dumps(dict(image=self.image_config,video=self.video_config),indent=2)+'\n')
        (root/'model_index.json').write_text(json.dumps(dict(model_type='prompthmr',bundle_class='motius.models.prompthmr.PromptHMRBundle',pipeline_class='motius.pipelines.prompthmr.PromptHMRPipeline',artifact_format='motius-prompthmr-native-v2'),indent=2)+'\n')
        return str(root)
