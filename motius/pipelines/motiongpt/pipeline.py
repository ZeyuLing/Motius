"""MotionGPT text-to-motion pipeline."""
from __future__ import annotations
import random
from typing import List, Optional, Sequence, Union
import numpy as np
import torch
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES

@PIPELINES.register_module()
class MotionGPTPipeline(BasePipeline):
    """Inference pipeline for MotionGPT on HumanML3D-263."""
    BUNDLE_CLS = 'motius.models.motiongpt.MotionGPTBundle'

    def __init__(self, bundle, device: Optional[str]=None, **kwargs):
        super().__init__(bundle, **kwargs)
        if device is not None:
            self.to(device)
        self._decoder = None

    def to(self, device):
        self.bundle.to_device(device)
        return self

    @property
    def device(self) -> torch.device:
        return self.bundle.device

    @torch.no_grad()
    def infer_t2m(self, captions: Sequence[str], lengths: Optional[Sequence[int]]=None, prompt_mode: Optional[str]=None, seed: Optional[int]=None, do_sample: bool=True, progress: bool=False) -> List[np.ndarray]:
        outputs = self._infer_hml263(captions, lengths=lengths, prompt_mode=prompt_mode, seed=seed, do_sample=do_sample)
        if progress:
            for i, arr in enumerate(outputs):
                print(f'[motiongpt] {i + 1}/{len(outputs)} shape={arr.shape}', flush=True)
        return outputs

    @torch.no_grad()
    def infer_m2t(self, motions: Sequence[Union[np.ndarray, torch.Tensor]], lengths: Optional[Sequence[int]]=None, with_len: bool=False, pad_to_batch_max: bool=False, progress: bool=False) -> List[str]:
        """Generate text descriptions from HML3D-263 motions.

        ``motions`` are expected to be denormalized HumanML3D-263 features, the
        same format saved by the T2M baseline scripts. MotionGPT's VQ-VAE was
        trained on normalized features, so normalization is applied here before
        tokenization. Set ``pad_to_batch_max=True`` only to reproduce the
        released evaluator's batch-dependent zero-padding behavior.
        """
        if lengths is None:
            lengths = [int(m.shape[0]) for m in motions]
        if len(motions) != len(lengths):
            raise ValueError('motions and lengths must have equal length')
        bundle = self.bundle
        normalized = []
        mean = bundle.mean.to(self.device)
        std = bundle.std.to(self.device)
        for motion, length in zip(motions, lengths):
            if isinstance(motion, torch.Tensor):
                motion_t = motion.detach().to(self.device, dtype=torch.float32)
            else:
                motion_t = torch.from_numpy(np.array(motion, dtype=np.float32, copy=True)).to(self.device)
            if motion_t.ndim != 2 or motion_t.shape[-1] != 263:
                raise ValueError(f'expected HML3D-263 motion shape (T,263), got {tuple(motion_t.shape)}')
            length = max(1, min(int(length), int(motion_t.shape[0])))
            motion_t = motion_t[:length]
            motion_norm = (motion_t - mean) / std.clamp_min(1e-08)
            normalized.append(motion_norm)
        if pad_to_batch_max:
            maximum = max((int(value.shape[0]) for value in normalized))
            motion_batch = normalized[0].new_zeros((len(normalized), maximum, normalized[0].shape[-1]))
            for index, value in enumerate(normalized):
                motion_batch[index, :value.shape[0]] = value
            tokens, _ = bundle.encode_motion(motion_batch)
            motion_tokens = [tokens[index] for index in range(len(normalized))]
            token_lengths = [int(tokens.shape[1])] * len(normalized)
        else:
            motion_tokens = []
            token_lengths = []
            for motion_norm in normalized:
                token, _ = bundle.encode_motion(motion_norm.unsqueeze(0))
                motion_tokens.append(token[0])
                token_lengths.append(int(token.shape[1]))
        texts = self.decoder.generate_conditional(motion_tokens=motion_tokens, lengths=token_lengths, task='m2t', with_len=with_len, stage='test')
        texts = [str(x).strip() for x in texts]
        if progress:
            for i, text in enumerate(texts):
                print(f'[motiongpt:m2t] {i + 1}/{len(texts)} {text}', flush=True)
        return texts

    @torch.no_grad()
    def infer_motion_reconstruction(self, motions: Sequence[Union[np.ndarray, torch.Tensor]], batch_size: int=64) -> List[np.ndarray]:
        """Round-trip physical-scale HumanML3D-263 motions through MotionGPT VQ."""
        bundle = self.bundle
        groups = {}
        for index, motion in enumerate(motions):
            value = np.asarray(motion.detach().cpu().numpy() if isinstance(motion, torch.Tensor) else motion, dtype=np.float32)
            if value.ndim != 2 or value.shape[1] != 263:
                raise ValueError(f'expected (T,263), got {value.shape}')
            usable = len(value) // 4 * 4
            if usable < 4:
                raise ValueError('MotionGPT reconstruction requires at least 4 frames')
            groups.setdefault(usable, []).append((index, value[:usable]))
        outputs: List[np.ndarray] = [None] * len(motions)
        mean = bundle.mean.to(self.device)
        std = bundle.std.to(self.device).clamp_min(1e-08)
        for length, items in groups.items():
            for start in range(0, len(items), max(1, int(batch_size))):
                chunk = items[start:start + max(1, int(batch_size))]
                batch = torch.from_numpy(np.stack([value for _, value in chunk])).to(self.device)
                reconstructed, _, _ = bundle.vae((batch - mean) / std)
                reconstructed = self.denormalize(reconstructed.float()).cpu().numpy()
                for row, (index, _) in enumerate(chunk):
                    outputs[index] = reconstructed[row, :length].astype(np.float32)
        return outputs

    def __call__(self, captions, lengths=None, **kwargs):
        return self.infer_t2m(captions, lengths, **kwargs)
    infer_reconstruction = infer_motion_reconstruction
    infer_text_to_motion = infer_t2m
    infer_motion_to_text = infer_m2t

    def denormalize(self, motion_263: torch.Tensor) -> torch.Tensor:
        return motion_263 * self.bundle.std.to(motion_263) + self.bundle.mean.to(motion_263)

    def generate_motion_tokens(self, captions: Sequence[str], lengths: Optional[Sequence[int]]=None, prompt_mode: Optional[str]=None, do_sample: bool=True) -> List[torch.Tensor]:
        mode = prompt_mode or self.bundle.prompt_mode
        captions = list(captions)
        lengths = [int(x) for x in lengths or [0] * len(captions)]
        if mode == 'official_nolen':
            return self.decoder.generate_conditional(texts=captions, lengths=lengths, task='t2m', with_len=False, stage='test', tasks=None)
        if mode == 'official_len':
            return self.decoder.generate_conditional(texts=captions, lengths=lengths, task='t2m', with_len=True, stage='test', tasks=None)
        if mode == 'direct':
            outputs, _ = self.decoder.generate_direct(captions, max_length=self.bundle.max_new_tokens, num_beams=1, do_sample=do_sample)
            return outputs
        raise ValueError(f'unsupported MotionGPT prompt_mode: {mode}')

    def decode_tokens(self, tokens: torch.Tensor, fallback_length: int) -> torch.Tensor:
        tokens = torch.clamp(tokens.to(self.device), 0, self.bundle.codebook_size - 1)
        if int(tokens.numel()) > 1:
            motion = self.bundle.decode_motion(tokens)[0]
        else:
            motion = torch.zeros((max(1, int(fallback_length)), 263), device=self.device)
        return self.denormalize(motion.float())

    @torch.no_grad()
    def _infer_hml263(self, captions: Sequence[str], lengths: Optional[Sequence[int]]=None, prompt_mode: Optional[str]=None, seed: Optional[int]=None, do_sample: bool=True) -> List[np.ndarray]:
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        target_lengths = [int(x) for x in lengths or [0] * len(captions)]
        tokens = self.generate_motion_tokens(captions, target_lengths, prompt_mode=prompt_mode, do_sample=do_sample)
        outputs: List[np.ndarray] = []
        for token_ids, target_len in zip(tokens, target_lengths):
            motion = self.decode_tokens(token_ids, fallback_length=target_len or 1)
            if target_len:
                motion = motion[:max(1, min(int(motion.shape[0]), target_len))]
            outputs.append(motion.detach().cpu().numpy().astype(np.float32))
        return outputs

    @property
    def decoder(self):
        if self._decoder is None:
            from .decoding import MotionGPTDecoder
            self._decoder = MotionGPTDecoder(self.bundle)
        self._decoder.device = self.device
        return self._decoder
