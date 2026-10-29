"""UniMuMo inference composition, conditioning and output conversion."""
from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import torch
from .sampling import generate_parallel
from motius.motion.conditioning.unimumo_pattern import DelayedPattern

class UniMuMoInference:

    def normalize_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return (motion - self.bundle.motion_mean) / self.bundle.motion_std

    def denormalize_motion(self, motion: torch.Tensor) -> torch.Tensor:
        return motion * self.bundle.motion_std + self.bundle.motion_mean

    def _prepare_waveform(self, waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
        waveform = torch.as_tensor(waveform, dtype=torch.float32, device=self.device)
        if waveform.ndim == 1:
            waveform = waveform[None, None]
        elif waveform.ndim == 2:
            waveform = waveform[:, None]
        if waveform.ndim != 3 or waveform.shape[1] != 1:
            raise ValueError('waveform must have shape (N), (B,N), or (B,1,N)')
        return waveform

    @torch.inference_mode()
    def encode_audio(self, waveform: torch.Tensor | np.ndarray) -> torch.Tensor:
        waveform = self._prepare_waveform(waveform)
        frame_multiple = self.bundle.sample_rate // int(self.bundle.code_fps)
        joint_multiple = frame_multiple * 5
        target_length = waveform.shape[-1] // joint_multiple * joint_multiple
        if target_length < joint_multiple:
            raise ValueError('waveform is too short for UniMuMo')
        encoded = self.bundle.audio_codec.encode(waveform[..., :target_length])
        codes = encoded.audio_codes
        if codes.shape[0] != 1:
            raise ValueError('Chunked Encodec artifacts are not supported')
        return codes[0].contiguous()

    @torch.inference_mode()
    def decode_audio(self, codes: torch.Tensor) -> torch.Tensor:
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim == 2:
            codes = codes[None]
        if codes.ndim != 3:
            raise ValueError('audio codes must have shape (K,T) or (B,K,T)')
        decoded = self.bundle.audio_codec.decode(codes[None], [None])
        return decoded.audio_values

    def _zero_audio_embeddings(self, batch: int, motion_frames: int) -> torch.Tensor:
        samples = int(round(motion_frames / self.bundle.motion_fps * self.bundle.sample_rate))
        waveform = torch.zeros((batch, 1, samples), dtype=self.bundle.motion_mean.dtype, device=self.device)
        return self.bundle.audio_codec.encoder(waveform)

    @torch.inference_mode()
    def encode_motion(self, motion: torch.Tensor | np.ndarray) -> torch.Tensor:
        motion = torch.as_tensor(motion, dtype=torch.float32, device=self.device)
        if motion.ndim == 2:
            motion = motion[None]
        if motion.ndim != 3 or motion.shape[-1] != int(self.bundle.config['motion_dim']):
            raise ValueError('motion must have shape (T,263) or (B,T,263)')
        frame_multiple = int(round(self.bundle.motion_fps / self.bundle.code_fps * 5))
        frames = motion.shape[1] // frame_multiple * frame_multiple
        if frames < frame_multiple:
            raise ValueError('motion is too short for UniMuMo')
        normalized = self.normalize_motion(motion[:, :frames])
        music_embeddings = self._zero_audio_embeddings(len(normalized), frames)
        motion_embeddings = self.bundle.motion_codec.encode_embeddings(normalized, music_embeddings)
        return self.bundle.audio_codec.quantizer.encode(motion_embeddings).permute(1, 0, 2)

    def _decode_code_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        return self.bundle.audio_codec.quantizer.decode(codes.permute(1, 0, 2))

    @torch.inference_mode()
    def decode_motion(self, music_codes: torch.Tensor, motion_codes: torch.Tensor) -> torch.Tensor:
        music_codes = torch.as_tensor(music_codes, dtype=torch.long, device=self.device)
        motion_codes = torch.as_tensor(motion_codes, dtype=torch.long, device=self.device)
        if music_codes.ndim == 2:
            music_codes = music_codes[None]
        if motion_codes.ndim == 2:
            motion_codes = motion_codes[None]
        if music_codes.shape != motion_codes.shape:
            raise ValueError('music and motion codes must share shape')
        music_embeddings = self._decode_code_embeddings(music_codes)
        motion_embeddings = self._decode_code_embeddings(motion_codes)
        normalized = self.bundle.motion_codec.decode_embeddings(music_embeddings, motion_embeddings)
        return self.denormalize_motion(normalized)

    def _encode_descriptions(self, descriptions: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        encoded = self.tokenizer(list(descriptions), padding=True, return_tensors='pt')
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        hidden = self.bundle.text_encoder(**encoded).last_hidden_state
        hidden = self.bundle.core.text_condition_projection(hidden)
        mask = encoded['attention_mask'].bool()
        return (hidden * mask[..., None], mask)

    @staticmethod
    def _split_descriptions(descriptions: Sequence[str], mode: str) -> tuple[list[str], list[str]]:
        music, motion = ([], [])
        for description in descriptions:
            parts = description.split('<separation>')
            music_text, motion_text = (parts[0].strip(), parts[-1].strip())
            if mode == 'music2motion':
                music_text = ''
            elif mode == 'motion2music':
                motion_text = ''
            elif mode != 'music_motion':
                raise ValueError(f'Unsupported generation mode: {mode!r}')
            music.append(music_text)
            motion.append(motion_text)
        return (music, motion)

    def text_condition(self, descriptions: Sequence[str], mode: str='music_motion') -> tuple[torch.Tensor, torch.Tensor]:
        music_text, motion_text = self._split_descriptions(descriptions, mode)
        return self._condition_from_streams(music_text, motion_text)

    def _condition_from_streams(self, music_text: Sequence[str], motion_text: Sequence[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if len(music_text) != len(motion_text):
            raise ValueError('music and motion text batches must have equal length')
        music_hidden, music_mask = self._encode_descriptions(music_text)
        motion_hidden, motion_mask = self._encode_descriptions(motion_text)
        hidden = torch.cat((music_hidden, motion_hidden), dim=1)
        mask = torch.zeros((len(music_text), 2, hidden.shape[1]), dtype=torch.bool, device=self.device)
        mask[:, 0, :music_mask.shape[1]] = music_mask
        mask[:, 1, music_mask.shape[1]:] = motion_mask
        return (hidden, mask)

    def cfg_text_condition(self, descriptions: Sequence[str], mode: str='music_motion') -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode conditional and null prompts with identical padding lengths."""
        music_text, motion_text = self._split_descriptions(descriptions, mode)
        batch = len(descriptions)
        hidden, mask = self._condition_from_streams(music_text + [''] * batch, motion_text + [''] * batch)
        return (hidden[:batch], mask[:batch], hidden[batch:], mask[batch:])

    def generate_codes(self, descriptions: Sequence[str], *, timesteps: int, mode: str='music_motion', music_codes: torch.Tensor | None=None, motion_codes: torch.Tensor | None=None, guidance_scale: float=4.0, temperature: float=1.0, top_k: int=250, generator: torch.Generator | None=None) -> tuple[torch.Tensor, torch.Tensor]:
        condition, condition_mask, unconditional, unconditional_mask = self.cfg_text_condition(descriptions, mode)
        return generate_parallel(self.bundle.generator, condition=condition, condition_mask=condition_mask, unconditional_condition=unconditional, unconditional_mask=unconditional_mask, timesteps=timesteps, music_codes=music_codes, motion_codes=motion_codes, guidance_scale=guidance_scale, temperature=temperature, top_k=top_k, generator=generator)

    @torch.inference_mode()
    def caption(self, codes: torch.Tensor, *, modality: str) -> list[str]:
        codes = torch.as_tensor(codes, dtype=torch.long, device=self.device)
        if codes.ndim == 2:
            codes = codes[None]
        if modality not in {'music', 'motion'}:
            raise ValueError("modality must be 'music' or 'motion'")
        pattern = DelayedPattern(codes.shape[-1], tuple(range(self.bundle.generator.num_codebooks)))
        sequence, _ = pattern.build(codes, special_token=self.bundle.generator.special_token_id, valid_only=True)
        empty = torch.full_like(sequence, self.bundle.generator.special_token_id)
        music_sequence, motion_sequence = (sequence, empty) if modality == 'music' else (empty, sequence)
        condition, condition_mask = self.text_condition(['<separation>'] * len(codes))
        context = self.bundle.generator.forward_features(music_sequence, motion_sequence, condition=condition, condition_mask=condition_mask, caption_mode=True)
        stream_length = sequence.shape[-1]
        attention_mask = torch.zeros((len(codes), stream_length * 2), dtype=torch.long, device=self.device)
        if modality == 'music':
            attention_mask[:, :stream_length] = 1
        else:
            attention_mask[:, stream_length:] = 1
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=self.bundle.core.caption_context_projection(context))
        output = self.bundle.captioner.generate(encoder_outputs=encoder_outputs, attention_mask=attention_mask, do_sample=False, max_length=int(self.bundle.config['caption_max_length']), num_beams=int(self.bundle.config['caption_num_beams']))
        return self.tokenizer.batch_decode(output, skip_special_tokens=True)
