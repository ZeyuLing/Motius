"""MotionBricks neural roots, separate from navigation, representation and losses."""
from pathlib import Path
import json
import torch
from omegaconf import OmegaConf
from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from motius.configuration.motionbricks import checkpoint_root,network_options
from .pose import pose_backbone_network
from .root import root_backbone_network
from types import SimpleNamespace
from .encdec import Encoder
from .encdec_double_cond import DoubleCondDecoder
from .quantizer import QuantizeEMAResetMultiHead


@MODEL_BUNDLES.register_module()
class MotionBricksBundle(ModelBundle):
    def __init__(self,artifact_root=None,*,checkpoint_dir=None,options=None,device='cuda'):
        super().__init__()
        if artifact_root is None:artifact_root=checkpoint_dir
        self.artifact_root=None if artifact_root is None else Path(artifact_root).expanduser().resolve()
        native=None if self.artifact_root is None else self.artifact_root/'motionbricks_neural.json'
        if options is None and self.artifact_root is None:raise ValueError('Provide artifact_root or network options')
        if options is None:options=json.loads(native.read_text()) if native.is_file() else network_options(artifact_root)
        self.options=OmegaConf.create({key:options[key] for key in ['pose','root','tokenizer']})
        cfg=self.options.tokenizer
        self.encoder=Encoder(cfg.encoder_state_dim,cfg.output_emb_width,cfg.down_t,cfg.stride_t,cfg.width,cfg.depth,
            cfg.get('dilation_growth_rate',3),activation=cfg.get('activation','relu'),norm=cfg.get('norm'))
        self.decoder=DoubleCondDecoder(cfg.decoder_state_dim,cfg.output_emb_width,cfg.down_t,cfg.width,cfg.depth,
            cfg.get('dilation_growth_rate',3),activation=cfg.get('activation','relu'),norm=cfg.get('norm'),
            target_cond_dim=cfg.decoder_target_cond_dim,external_cond_dim=cfg.decoder_external_cond_dim,
            cond_fusion_last_layer=cfg.get('cond_fusion_last_layer',False))
        self.quantizer=QuantizeEMAResetMultiHead(cfg.nb_code,cfg.code_dim,
            SimpleNamespace(mu=cfg.get('quantizer_mu',.99),num_heads=cfg.num_heads,kmeans_init=cfg.get('kmeans_init',True)))
        self.pose=pose_backbone_network(self.options.pose)
        self.root=root_backbone_network(self.options.root)
        if self.artifact_root is not None:
            if native.is_file():
                self.load_state_dict(torch.load(self.artifact_root/'motionbricks_networks.pt',map_location='cpu',weights_only=True,mmap=True),strict=True)
            else:
                root=checkpoint_root(artifact_root)
                for name,module,prefix in [('pose',self.pose,'backbone_net.'),('root',self.root,'backbone_net.')]:
                    state=torch.load(root/('motionbricks_'+name)/'version_1/checkpoints/model-step=2000000.ckpt',map_location='cpu',weights_only=False,mmap=True)['state_dict']
                    if any(not key.startswith(prefix) for key in state):raise ValueError('Unexpected learned component in '+name)
                    module.load_state_dict({key[len(prefix):]:value for key,value in state.items()},strict=True)
                state=torch.load(root/'motionbricks_vqvae/version_1/checkpoints/model-step=2000000.ckpt',map_location='cpu',weights_only=False,mmap=True)['state_dict']
                consumed=set()
                for name in ['encoder','decoder','quantizer']:
                    prefix='pose_net.'+name+'.'
                    component={key[len(prefix):]:value for key,value in state.items() if key.startswith(prefix)}
                    consumed.update(prefix+key for key in component)
                    getattr(self,name).load_state_dict(component,strict=True)
                if consumed!=set(state):raise ValueError('Unexpected learned component in vqvae')
        self._save_ckpt_modules=['encoder','decoder','quantizer','pose','root']
        self.to(device).eval()

    @property
    def device(self):return next(self.pose.parameters()).device

    def predict_pose(self,*args,**kwargs):return self.pose(*args,**kwargs)
    def predict_root(self,*args,**kwargs):return self.root(*args,**kwargs)
    def encode_features(self,features):return self.encoder(features.permute(0,2,1))
    def quantize_features(self,latents):return self.quantizer(latents)
    def project_quantized(self,result):return QuantizeEMAResetMultiHead.straight_through(result)

    def decode_features(self,quantized,*,external_cond=None,target_cond=None,has_target_cond=None,token_mask=None):
        return self.decoder(quantized,external_cond,target_cond,has_target_cond,token_mask=token_mask).permute(0,2,1)

    def encode_tokens(self,features):return self.quantize_features(self.encode_features(features))['indices']

    def decode_tokens(self,indices,**kwargs):
        quantized=self.quantizer.dequantize(indices).permute(0,2,1).contiguous()
        return {'recon_state':self.decode_features(quantized,**kwargs)}

    def reconstruct(self,features,**kwargs):
        result=self.quantize_features(self.encode_features(features))
        result['recon_state']=self.decode_features(self.project_quantized(result),**kwargs)
        return result

    @classmethod
    def from_pretrained(cls,pretrained_model_name_or_path,*,revision=None,cache_dir=None,token=None,local_files_only=False,**kwargs):
        path=Path(pretrained_model_name_or_path).expanduser()
        if not path.is_dir():
            from huggingface_hub import snapshot_download
            path=Path(snapshot_download(str(pretrained_model_name_or_path),revision=revision,cache_dir=cache_dir,token=token,local_files_only=local_files_only))
        return cls(path,**kwargs)

    def save_pretrained(self,save_directory,**kwargs):
        destination=Path(save_directory);destination.mkdir(parents=True,exist_ok=True)
        state={prefix+'.'+key:value.detach().cpu() for prefix in ['encoder','decoder','quantizer','pose','root'] for key,value in getattr(getattr(self,prefix),'module',getattr(self,prefix)).state_dict().items()}
        torch.save(state,destination/'motionbricks_networks.pt')
        (destination/'motionbricks_neural.json').write_text(json.dumps(OmegaConf.to_container(self.options,resolve=True),indent=2)+'\n')
        (destination/'model_index.json').write_text(json.dumps(dict(model_type='motionbricks',bundle_class='motius.models.motionbricks.MotionBricksBundle',pipeline_class='motius.pipelines.motionbricks.MotionBricksPipeline',artifact_format='motius-motionbricks-native-v2'),indent=2)+'\n')
        return str(destination)
