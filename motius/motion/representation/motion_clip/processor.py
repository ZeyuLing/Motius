"""Raw text and motion preparation, separate from neural encoders."""

from typing import Dict, Any
from copy import deepcopy
from pathlib import Path
import torch


def _build_tokenizer(cfg: Dict[str, Any]):
    cfg = deepcopy(cfg)
    cfg_type = cfg.pop("type", "CLIPTokenizer")
    from_pretrained = cfg.pop("from_pretrained", None)
    if cfg_type == "CLIPTokenizer":
        from transformers import CLIPTokenizer as _Cls
    elif cfg_type == "AutoTokenizer":
        from transformers import AutoTokenizer as _Cls
    else:
        from transformers import AutoTokenizer as _Cls
    if from_pretrained is None:
        raise ValueError("Tokenizer config must include 'from_pretrained'.")
    return _Cls.from_pretrained(**from_pretrained)


class MotionCLIPProcessor:
    def __init__(self, bundle):
        self.bundle = bundle
        config = deepcopy(bundle.tokenizer_config)
        root = getattr(bundle, "artifact_root", None)
        if root is not None and (Path(root) / "tokenizer").is_dir():
            config["from_pretrained"] = {
                "pretrained_model_name_or_path": str(Path(root) / "tokenizer"),
                "local_files_only": True,
            }
        self.tokenizer = _build_tokenizer(config)
        self.smpl_pose_processor = None
        if bundle.smpl_pose_processor_config is not None:
            from motius.registry import HF_MODELS

            self.smpl_pose_processor = HF_MODELS.build(
                bundle.smpl_pose_processor_config
            )
            self.smpl_pose_processor.eval().requires_grad_(False)

    def tokenize(self, texts, max_length=None):
        if max_length is None:
            max_length = (
                self.bundle.motionclip_model.config.text_config.max_position_embeddings
            )
        if isinstance(texts, str):
            texts = [texts]
        return self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

    def normalize_motion(self, motion):
        if self.smpl_pose_processor is None:
            return motion
        self.smpl_pose_processor.to(motion.device).eval()
        with torch.no_grad():
            return self.smpl_pose_processor.normalize(motion)

    def prepare_batch(self, motion, captions, num_frames):
        device = motion.device
        motion = self.normalize_motion(motion)
        max_len = max(int(nf) for nf in num_frames)
        motion = motion[:, :max_len].contiguous()
        attention = torch.zeros(
            motion.shape[0], max_len, device=device, dtype=motion.dtype
        )
        for index, count in enumerate(num_frames):
            attention[index, : int(count)] = 1.0
        encoded = self.tokenize(captions)
        return dict(
            input_ids=encoded["input_ids"].to(device),
            motion_values=motion,
            attention_mask=encoded["attention_mask"].to(device),
            motion_attention_mask=attention,
        )

    def save_pretrained(self, directory):
        self.tokenizer.save_pretrained(str(Path(directory) / "tokenizer"))
