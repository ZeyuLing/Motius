"""VerMo multimodal-token bundle."""

from __future__ import annotations

import json
import os
import inspect
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class VermoBundle(ModelBundle):
    """ModelBundle for VerMo multimodal motion generation."""

    def __init__(
        self,
        processor: dict,
        lm: dict,
        mean_init_embeddings: bool = False,
        device: str = "cuda",
    ):
        super().__init__()
        self.mean_init_embeddings = mean_init_embeddings
        self._build_modules({'processor': processor, 'lm': lm})
        self._resize_token_embeddings()
        resolved_device = device if torch.cuda.is_available() else 'cpu'
        self.to(resolved_device).eval()

    @classmethod
    def _bundle_config_from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        device: str = "cuda",
        **kwargs,
    ) -> Dict[str, Any]:
        candidate = Path(pretrained_model_name_or_path).expanduser()
        if not candidate.is_dir():
            from huggingface_hub import snapshot_download

            candidate = Path(
                snapshot_download(repo_id=pretrained_model_name_or_path, repo_type='model')
            )
        root = str(candidate.resolve())
        cfg_path = os.path.join(root, 'bundle_config.json')
        if not os.path.isfile(cfg_path):
            raise FileNotFoundError(
                f"Expected bundle_config.json under {root}. "
                "Use VermoBundle.save_pretrained() exports, or construct via from_config()."
            )
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
        stats_path = os.path.join(root, 'stats', 'smplh_universal_stats_aug.json')
        processor_cfg['smpl_pose_processor']['stats_file'] = stats_path
        if processor_cfg.get('multi_person_smpl_pose_processor') is not None:
            processor_cfg['multi_person_smpl_pose_processor']['stats_file'] = stats_path
        lm_fp = lm_cfg.get('from_pretrained') or {}
        lm_fp['pretrained_model_name_or_path'] = os.path.join(root, 'lm')
        lm_cfg['from_pretrained'] = lm_fp

        return {
            'processor': processor_cfg,
            'lm': lm_cfg,
            'mean_init_embeddings': bundle_cfg.get('mean_init_embeddings', False),
            'device': device,
        }

    def _resize_token_embeddings(self):
        if hasattr(self.lm, 'resize_token_embeddings'):
            resize_kwargs = {}
            if 'mean_resizing' in inspect.signature(self.lm.resize_token_embeddings).parameters:
                resize_kwargs['mean_resizing'] = self.mean_init_embeddings
            self.lm.resize_token_embeddings(
                self.processor.vocab_size,
                **resize_kwargs,
            )

    def save_pretrained(self, save_directory: str, merge_lora: bool = True, **kwargs):
        os.makedirs(save_directory, exist_ok=True)
        if merge_lora and self.is_lora_module('lm'):
            self.merge_lora_weights(['lm'])

        self.lm.save_pretrained(os.path.join(save_directory, 'lm'), **kwargs)
        self.processor.text_tokenizer.save_pretrained(os.path.join(save_directory, 'tokenizer'))
        self.processor.motion_tokenizer.save_pretrained(
            os.path.join(save_directory, 'motion_tokenizer')
        )
        stats_dir = os.path.join(save_directory, 'stats')
        os.makedirs(stats_dir, exist_ok=True)
        shutil.copy2(
            self.processor.smpl_pose_processor.stats_file,
            os.path.join(stats_dir, 'smplh_universal_stats_aug.json'),
        )
        with open(os.path.join(save_directory, 'bundle_config.json'), 'w', encoding='utf-8') as f:
            json.dump(
                {
                    'processor': self.get_module_build_cfg('processor'),
                    'lm': self.get_module_build_cfg('lm'),
                    'mean_init_embeddings': self.mean_init_embeddings,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

    def process_train(self, inputs: Dict[str, Any]):
        return self.processor.process_train(inputs)

    def forward_lm(self, inputs: Dict[str, Any]):
        return self.lm(**self.process_train(inputs))
