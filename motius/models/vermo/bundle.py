"""VerMo language, motion-codec and optional audio neural modules."""
from __future__ import annotations
import copy, inspect, json, os
from pathlib import Path
from typing import Dict, Any
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES, MODELS

@MODEL_BUNDLES.register_module()
class VermoBundle(ModelBundle):
    def __init__(self, processor, lm, mean_init_embeddings=False, device='cuda'):
        super().__init__()
        self.processor_config=copy.deepcopy(processor)
        self.mean_init_embeddings=mean_init_embeddings
        motion=copy.deepcopy(processor['motion_tokenizer'])
        motion.setdefault('trainable',False)
        modules={'motion_tokenizer':motion,'lm':lm}
        self._build_modules(modules)
        self.audio_tokenizer=None
        if processor.get('audio_tokenizer') is not None:
            self.audio_tokenizer=MODELS.build(processor['audio_tokenizer'])
            if processor.get('pretrained_audio_tokenizer'):
                from mmengine.runner import load_checkpoint
                load_checkpoint(self.audio_tokenizer,processor['pretrained_audio_tokenizer'],map_location='cpu')
            self.audio_tokenizer.requires_grad_(False)
        self.to(device if torch.cuda.is_available() or str(device)!='cuda' else 'cpu').eval()

    @property
    def device(self):return next(self.lm.parameters()).device

    def resize_language_embeddings(self,vocab_size):
        options={}
        if 'mean_resizing' in inspect.signature(self.lm.resize_token_embeddings).parameters:
            options['mean_resizing']=self.mean_init_embeddings
        self.lm.resize_token_embeddings(vocab_size,**options)

    def forward_lm(self,inputs):return self.lm(**inputs)

    def forward(self,**inputs):return self.lm(**inputs)

    def save_pretrained(self,save_directory,merge_lora=True,**kwargs):
        target=Path(save_directory);target.mkdir(parents=True,exist_ok=True)
        if merge_lora and self.is_lora_module('lm'):self.merge_lora_weights(['lm'])
        language=self.lm.module if hasattr(self.lm,'module') else self.lm
        language.save_pretrained(target/'lm',**kwargs)
        getattr(self.motion_tokenizer,'module',self.motion_tokenizer).save_pretrained(target/'motion_tokenizer')
        config=dict(processor=self.processor_config,lm=self.get_module_build_cfg('lm'),mean_init_embeddings=self.mean_init_embeddings)
        config['lm']['from_pretrained']['torch_dtype']=str(next(language.parameters()).dtype).removeprefix('torch.')
        from motius.configuration.vermo import export_audio_network, export_processor_assets
        config=copy.deepcopy(config)
        export_audio_network(self,config,target)
        export_processor_assets(config,target)
        return str(target)

    @classmethod
    def _bundle_config_from_pretrained(cls, pretrained_model_name_or_path: str, device: str='cuda', **kwargs) -> Dict[str, Any]:
        candidate = Path(pretrained_model_name_or_path).expanduser()
        if not candidate.is_dir():
            from huggingface_hub import snapshot_download
            candidate = Path(snapshot_download(repo_id=pretrained_model_name_or_path, repo_type='model'))
        root = str(candidate.resolve())
        cfg_path = os.path.join(root, 'bundle_config.json')
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(f'Expected bundle_config.json under {root}. Use VermoBundle.save_pretrained() exports, or construct via from_config().')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            bundle_cfg = json.load(f)
        processor_cfg = bundle_cfg['processor']
        lm_cfg = bundle_cfg['lm']
        text_tok_cfg = processor_cfg.get('pretrained_text_tokenizer', {})
        fp = text_tok_cfg.get('from_pretrained') or {}
        fp['pretrained_model_name_or_path'] = os.path.join(root, 'tokenizer')
        text_tok_cfg['from_pretrained'] = fp
        motion_tok_cfg = processor_cfg.get('motion_tokenizer', {})
        motion_fp = motion_tok_cfg.get('from_pretrained') or {}
        motion_fp['pretrained_model_name_or_path'] = os.path.join(root, 'motion_tokenizer')
        motion_tok_cfg['from_pretrained'] = motion_fp
        if processor_cfg.get('audio_tokenizer',{}):
            audio=processor_cfg['audio_tokenizer']
            if audio.get('pretrained'):
                audio['pretrained']=str(Path(root)/audio['pretrained'])
        for name in ['smpl_pose_processor','multi_person_smpl_pose_processor']:
            if processor_cfg.get(name) is not None:
                processor_cfg[name]['stats_file']=str(Path(root)/processor_cfg[name].get('stats_file','stats/smplh_universal_stats_aug.json'))
        lm_fp = lm_cfg.get('from_pretrained') or {}
        lm_fp['pretrained_model_name_or_path'] = os.path.join(root, 'lm')
        lm_cfg['from_pretrained'] = lm_fp
        return {'processor': processor_cfg, 'lm': lm_cfg, 'mean_init_embeddings': bundle_cfg.get('mean_init_embeddings', False), 'device': device}
