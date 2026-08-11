# coding=utf-8
"""MotionCLIPPipeline: inference / retrieval / embedding extraction.

Used as the inference-side companion to :class:`MotionCLIPBundle`.
Exposes:
  * ``encode_text(texts) -> (B, projection_dim)``
  * ``encode_motion(motion, num_frames=None) -> (B, projection_dim)``
  * ``score(texts, motion, num_frames=None) -> (B,)`` cosine similarity per pair
  * ``retrieve_motion_from_text(query_texts, motion_db, top_k)``
  * ``retrieve_text_from_motion(query_motions, text_db, top_k)``
"""

from __future__ import annotations

from typing import Dict, List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class MotionCLIPPipeline(BasePipeline):
    """MotionCLIP evaluator pipeline for text/motion embeddings and retrieval."""

    BUNDLE_CLS = "motius.models.motion_clip.MotionCLIPBundle"

    def __init__(self, bundle, device: Optional[str] = None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)

    def to(self, device):
        self.bundle.to(device)
        return self

    @property
    def device(self) -> torch.device:
        return next(self.bundle.motionclip_model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.bundle.motionclip_model.parameters()).dtype

    # ------------------------------------------------------------------
    # Embedding APIs
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
    ) -> Tensor:
        if isinstance(texts, str):
            texts = [texts]
        enc = self.bundle.tokenize(texts)
        input_ids = enc['input_ids'].to(self.device)
        attn = enc['attention_mask'].to(self.device)
        feats = self.bundle.encode_text(input_ids, attn)
        if normalize:
            feats = F.normalize(feats, dim=-1)
        return feats

    @torch.no_grad()
    def encode_motion(
        self,
        motion: Tensor,
        num_frames: Optional[List[int]] = None,
        already_normalized: bool = False,
        normalize_output: bool = True,
    ) -> Tensor:
        """Encode motion to MotionCLIP embedding space.

        Args:
            motion: (B, T, D) raw or normalized motion tensor.
            num_frames: list of valid frame counts (defaults to T for each sample).
            already_normalized: skip SMPLPoseProcessor.normalize if True.
            normalize_output: L2-normalize the projected embedding (default True
                for retrieval). Set False to compute FID on the raw projection.
        """
        if motion.dim() == 2:
            motion = motion.unsqueeze(0)
        motion = motion.to(self.device, dtype=torch.float32)

        if not already_normalized and self.bundle.smpl_pose_processor is not None:
            motion = self.bundle.smpl_pose_processor.normalize(motion)

        B, T, D = motion.shape
        if num_frames is None:
            num_frames = [T] * B

        max_len = max(int(nf) for nf in num_frames)
        motion = motion[:, :max_len].contiguous()
        attn = torch.zeros(B, max_len, device=self.device, dtype=motion.dtype)
        for i, nf in enumerate(num_frames):
            attn[i, : int(nf)] = 1.0

        feats = self.bundle.encode_motion(motion, attn)
        if normalize_output:
            feats = F.normalize(feats, dim=-1)
        return feats

    # ------------------------------------------------------------------
    # Convenience: pairwise score
    # ------------------------------------------------------------------

    @torch.no_grad()
    def score(
        self,
        texts: Union[str, List[str]],
        motion: Tensor,
        num_frames: Optional[List[int]] = None,
    ) -> Tensor:
        text_emb = self.encode_text(texts, normalize=True)
        motion_emb = self.encode_motion(
            motion, num_frames=num_frames, normalize_output=True,
        )
        if text_emb.shape[0] != motion_emb.shape[0]:
            raise ValueError(
                f"text batch ({text_emb.shape[0]}) and motion batch "
                f"({motion_emb.shape[0]}) sizes must match for pairwise score."
            )
        return (text_emb * motion_emb).sum(dim=-1)

    @torch.no_grad()
    def retrieve_motion_from_text(
        self,
        query_texts: Union[str, List[str]],
        motion_db: Tensor,
        motion_num_frames: Optional[List[int]] = None,
        top_k: int = 5,
    ) -> Dict[str, Tensor]:
        text_emb = self.encode_text(query_texts, normalize=True)
        motion_emb = self.encode_motion(
            motion_db, num_frames=motion_num_frames, normalize_output=True,
        )
        sim = torch.matmul(text_emb, motion_emb.T)
        topv, topi = sim.topk(min(top_k, motion_emb.shape[0]), dim=-1)
        return {
            'similarity': sim,
            'top_k_indices': topi,
            'top_k_scores': topv,
        }

    @torch.no_grad()
    def retrieve_text_from_motion(
        self,
        query_motions: Tensor,
        text_db: List[str],
        motion_num_frames: Optional[List[int]] = None,
        top_k: int = 5,
    ) -> Dict[str, Tensor]:
        motion_emb = self.encode_motion(
            query_motions, num_frames=motion_num_frames, normalize_output=True,
        )
        text_emb = self.encode_text(text_db, normalize=True)
        sim = torch.matmul(motion_emb, text_emb.T)
        topv, topi = sim.topk(min(top_k, len(text_db)), dim=-1)
        return {
            'similarity': sim,
            'top_k_indices': topi,
            'top_k_scores': topv,
            'top_k_texts': [
                [text_db[int(j)] for j in idx] for idx in topi
            ],
        }

    def __call__(
        self,
        texts: Union[str, List[str]],
        motion: Tensor,
        num_frames: Optional[List[int]] = None,
    ) -> Tensor:
        return self.score(texts, motion, num_frames=num_frames)
