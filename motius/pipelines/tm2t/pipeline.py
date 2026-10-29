"""TM2T motion-to-text inference pipeline."""

from __future__ import annotations

from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class TM2TPipeline(BasePipeline):
    """Caption denormalized HumanML3D-263 motions with TM2T."""

    BUNDLE_CLS = "motius.models.tm2t.TM2TBundle"

    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, **kwargs)
        from motius.motion.conditioning.tm2t import TM2TVocabulary
        from .beam_search import Translator
        self.vocabulary = TM2TVocabulary(bundle.artifact_dir)
        self.translator = Translator(
            bundle.transformer, beam_size=bundle.beam_size,
            max_seq_len=bundle.max_text_length, src_pad_idx=bundle.motion_pad_index,
            trg_pad_idx=bundle.text_pad_index, trg_sos_idx=bundle.text_start_index,
            trg_eos_idx=bundle.text_end_index).to(bundle.device)

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @torch.no_grad()
    def infer_m2t(
        self,
        motions: Sequence[Union[np.ndarray, torch.Tensor]],
        lengths: Optional[Sequence[int]] = None,
    ) -> List[str]:
        if lengths is None:
            lengths = [int(motion.shape[0]) for motion in motions]
        if len(motions) != len(lengths):
            raise ValueError("motions and lengths must have equal length")

        outputs: List[str] = []
        bundle = self.bundle
        for motion, requested_length in zip(motions, lengths):
            value = torch.as_tensor(motion)
            if value.ndim != 2 or value.shape[-1] != 263:
                raise ValueError(
                    f"expected HML3D-263 motion shape (T,263), got {tuple(value.shape)}"
                )
            length = min(int(requested_length), int(value.shape[0]), 196)
            length = max(4, length // 4 * 4)
            from motius.motion.conditioning.tm2t import normalize_features
            normalized = normalize_features(value[:length], bundle.mean, bundle.std, device=self.device)
            latents = bundle.vq_encoder(normalized[None, ..., :-4])
            indices = bundle.quantizer.map2index(latents).flatten().tolist()
            if len(indices) > 53:
                indices = indices[:53]
            motion_tokens = [bundle.motion_start_index]
            motion_tokens.extend(int(index) for index in indices)
            motion_tokens.append(bundle.motion_end_index)
            motion_tokens.extend(
                [bundle.motion_pad_index] * (55 - len(motion_tokens))
            )
            token_tensor = torch.tensor(
                motion_tokens, dtype=torch.long, device=self.device
            ).unsqueeze(0)
            generated = self.translator.translate_sentence(token_tensor)[1:-1]
            outputs.append(" ".join(self.vocabulary.token(index) for index in generated))
        return outputs

    def __call__(self, motions, lengths=None, **kwargs):
        return self.infer_m2t(motions, lengths=lengths, **kwargs)

    infer_motion_to_text = infer_m2t


__all__ = ["TM2TPipeline"]
