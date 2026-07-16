"""
Embedding modules for MotionWan Transformer.

This module provides embedding classes for processing timesteps and text conditions
in the MotionWan diffusion model architecture. The embeddings are designed to be
compatible with the Wan-style transformer blocks.
"""

from typing import Optional, Tuple

from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from torch import nn
import torch


def _module_dtype_or_fallback(*modules: nn.Module, fallback: torch.dtype) -> torch.dtype:
    for module in modules:
        for param in module.parameters(recurse=True):
            return param.dtype
        for buffer in module.buffers(recurse=True):
            return buffer.dtype
    return fallback


class WanTimeTextEmbedding(nn.Module):
    """
    Embedding module for processing both timestep and text conditions.

    This module combines sinusoidal timestep embeddings with text embeddings
    for conditioning the MotionWan transformer. It processes discrete timesteps
    into continuous embeddings and projects text features into the model's
    hidden dimension.

    Architecture:
        1. Timestep Processing:
           - Sinusoidal projection (Timesteps) -> frequency encoding
           - MLP embedding (TimestepEmbedding) -> time embedding
           - SiLU activation + Linear projection -> timestep projection

        2. Text Processing:
           - PixArtAlphaTextProjection with GELU-tanh activation

    Args:
        dim (int): The main hidden dimension of the model.
        time_freq_dim (int): The dimension for sinusoidal frequency encoding of timesteps.
        time_proj_dim (int): The output dimension for the timestep projection.
        text_embed_dim (int): The input dimension of the text encoder hidden states.
        pos_embed_seq_len (Optional[int]): Reserved for positional embedding sequence length.
            Currently not used but kept for future compatibility. Defaults to None.

    Example:
        >>> embedding = WanTimeTextEmbedding(
        ...     dim=128,
        ...     time_freq_dim=256,
        ...     time_proj_dim=128,
        ...     text_embed_dim=4096,
        ... )
        >>> timestep = torch.randint(0, 1000, (batch_size,))
        >>> text_hidden_states = torch.randn(batch_size, text_embed_dim)
        >>> temb, timestep_proj, text_emb = embedding(timestep, text_hidden_states)
    """

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        pos_embed_seq_len: Optional[int] = None,
    ):
        super().__init__()

        # Sinusoidal timestep projection: converts discrete timesteps to frequency encoding
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        # MLP to embed the frequency-encoded timesteps into the model dimension
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        # Activation function for timestep projection
        self.act_fn = nn.SiLU()
        # Linear projection for timestep embeddings (used in transformer blocks)
        self.time_proj = nn.Linear(dim, time_proj_dim)
        # Text embedding projection with GELU-tanh activation (PixArt-Alpha style)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process timesteps and text hidden states into embeddings.

        Args:
            timestep (torch.Tensor): Discrete diffusion timesteps.
                Shape: (batch_size,) or (batch_size * seq_len,) if timestep_seq_len is provided.
            encoder_hidden_states (torch.Tensor): Text encoder hidden states.
                Shape: (batch_size, text_embed_dim) or (batch_size, seq_len, text_embed_dim).
            timestep_seq_len (Optional[int]): If provided, reshapes the timestep tensor
                from (batch_size * seq_len,) to (batch_size, seq_len, ...).
                Useful for per-frame timestep conditioning. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                - temb: Time embedding for global conditioning.
                    Shape: (batch_size, dim) or (batch_size, seq_len, dim).
                - timestep_proj: Projected timestep embedding for transformer blocks.
                    Shape: (batch_size, time_proj_dim) or (batch_size, seq_len, time_proj_dim).
                - encoder_hidden_states: Projected text embeddings.
                    Shape: (batch_size, dim) or (batch_size, seq_len, dim).
        """
        # Step 1: Apply sinusoidal projection to timesteps
        timestep = self.timesteps_proj(timestep)

        # Step 2: Optionally reshape for sequence-level timesteps
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        # Step 3: Handle dtype compatibility for mixed precision training
        time_embedder_dtype = _module_dtype_or_fallback(
            self.time_embedder,
            self.time_proj,
            self.text_embedder,
            fallback=encoder_hidden_states.dtype,
        )
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)

        # Step 4: Compute time embedding through MLP
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        # Step 5: Project time embedding with activation for transformer conditioning
        timestep_proj = self.time_proj(self.act_fn(temb))

        # Step 6: Project text embeddings to model dimension
        # Run in fp32 to prevent GELU-tanh overflow under fp16 autocast.
        # PixArtAlphaTextProjection uses nn.GELU(approximate="tanh") which computes x^3
        # that overflows fp16 when |x| > 40.3.
        with torch.amp.autocast('cuda', dtype=torch.float32):
            encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        # Cast back to match temb dtype so downstream layers (cross-attention with
        # bf16 weights under fp16 autocast) don't encounter dtype mismatch.
        encoder_hidden_states = encoder_hidden_states.type_as(temb)

        return temb, timestep_proj, encoder_hidden_states


class WanTimeEmbedding(nn.Module):
    """
    Embedding module for processing timestep conditions only (without text).

    This is a simplified version of WanTimeTextEmbedding that only handles
    timestep embeddings. It is designed for unconditional or class-conditional
    generation scenarios where text conditioning is not required.

    Architecture:
        - Sinusoidal projection (Timesteps) -> frequency encoding
        - MLP embedding (TimestepEmbedding) -> time embedding
        - SiLU activation + Linear projection -> timestep projection

    Args:
        dim (int): The main hidden dimension of the model.
        time_freq_dim (int): The dimension for sinusoidal frequency encoding of timesteps.
        time_proj_dim (int): The output dimension for the timestep projection.

    Example:
        >>> embedding = WanTimeEmbedding(
        ...     dim=128,
        ...     time_freq_dim=256,
        ...     time_proj_dim=128,
        ... )
        >>> timestep = torch.randint(0, 1000, (batch_size,))
        >>> temb, timestep_proj = embedding(timestep)
    """

    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
    ):
        super().__init__()

        # Sinusoidal timestep projection: converts discrete timesteps to frequency encoding
        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        # MLP to embed the frequency-encoded timesteps into the model dimension
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        # Activation function for timestep projection
        self.act_fn = nn.SiLU()
        # Linear projection for timestep embeddings (used in transformer blocks)
        self.time_proj = nn.Linear(dim, time_proj_dim)

    def forward(
        self,
        timestep: torch.Tensor,
        timestep_seq_len: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process timesteps into embeddings.

        Args:
            timestep (torch.Tensor): Discrete diffusion timesteps.
                Shape: (batch_size,) or (batch_size * seq_len,) if timestep_seq_len is provided.
            timestep_seq_len (Optional[int]): If provided, reshapes the timestep tensor
                from (batch_size * seq_len,) to (batch_size, seq_len, ...).
                Useful for per-frame timestep conditioning. Defaults to None.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - temb: Time embedding for global conditioning.
                    Shape: (batch_size, dim) or (batch_size, seq_len, dim).
                - timestep_proj: Projected timestep embedding for transformer blocks.
                    Shape: (batch_size, time_proj_dim) or (batch_size, seq_len, time_proj_dim).
        """
        # Step 1: Apply sinusoidal projection to timesteps
        timestep = self.timesteps_proj(timestep)

        # Step 2: Optionally reshape for sequence-level timesteps
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        # Step 3: Handle dtype compatibility for mixed precision training
        time_embedder_dtype = _module_dtype_or_fallback(
            self.time_embedder,
            self.time_proj,
            fallback=timestep.dtype,
        )
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)

        # Step 4: Compute time embedding through MLP
        temb = self.time_embedder(timestep)

        # Step 5: Project time embedding with activation for transformer conditioning
        timestep_proj = self.time_proj(self.act_fn(temb))

        return temb, timestep_proj


if __name__ == "__main__":
    """
    Test script for WanTimeTextEmbedding and WanTimeEmbedding modules.

    This script validates:
    1. Basic forward pass for both embedding classes
    2. Output shapes match expected dimensions
    3. Sequence-level timestep handling (timestep_seq_len parameter)
    4. Different batch sizes and configurations
    """
    print("=" * 60)
    print("Testing WanTimeTextEmbedding and WanTimeEmbedding modules")
    print("=" * 60)

    # ==================== Test Configuration ====================
    batch_size = 4
    seq_len = 16
    dim = 128
    time_freq_dim = 256
    time_proj_dim = 128
    text_embed_dim = 4096

    # ==================== Test WanTimeTextEmbedding ====================
    print("\n[Test 1] WanTimeTextEmbedding - Basic Forward Pass")
    print("-" * 50)

    embedding_with_text = WanTimeTextEmbedding(
        dim=dim,
        time_freq_dim=time_freq_dim,
        time_proj_dim=time_proj_dim,
        text_embed_dim=text_embed_dim,
    )

    # Test with simple batch input
    timestep = torch.randint(0, 1000, (batch_size,))
    encoder_hidden_states = torch.randn(batch_size, text_embed_dim)

    temb, timestep_proj, text_emb = embedding_with_text(timestep, encoder_hidden_states)

    print(f"Input timestep shape: {timestep.shape}")
    print(f"Input text hidden states shape: {encoder_hidden_states.shape}")
    print(f"Output temb shape: {temb.shape} (expected: [{batch_size}, {dim}])")
    print(
        f"Output timestep_proj shape: {timestep_proj.shape} (expected: [{batch_size}, {time_proj_dim}])"
    )
    print(f"Output text_emb shape: {text_emb.shape} (expected: [{batch_size}, {dim}])")

    # Validate shapes
    assert temb.shape == (batch_size, dim), f"temb shape mismatch!"
    assert timestep_proj.shape == (
        batch_size,
        time_proj_dim,
    ), f"timestep_proj shape mismatch!"
    assert text_emb.shape == (batch_size, dim), f"text_emb shape mismatch!"
    print("✓ All shape assertions passed!")

    # ==================== Test WanTimeTextEmbedding with Sequence Timesteps ====================
    print("\n[Test 2] WanTimeTextEmbedding - Sequence-level Timesteps")
    print("-" * 50)

    # Timesteps for each frame in the sequence
    timestep_seq = torch.randint(0, 1000, (batch_size * seq_len,))
    encoder_hidden_states_seq = torch.randn(batch_size, seq_len, text_embed_dim)

    temb_seq, timestep_proj_seq, text_emb_seq = embedding_with_text(
        timestep_seq, encoder_hidden_states_seq, timestep_seq_len=seq_len
    )

    print(f"Input timestep shape: {timestep_seq.shape}")
    print(f"Input text hidden states shape: {encoder_hidden_states_seq.shape}")
    print(
        f"Output temb shape: {temb_seq.shape} (expected: [{batch_size}, {seq_len}, {dim}])"
    )
    print(
        f"Output timestep_proj shape: {timestep_proj_seq.shape} (expected: [{batch_size}, {seq_len}, {time_proj_dim}])"
    )
    print(
        f"Output text_emb shape: {text_emb_seq.shape} (expected: [{batch_size}, {seq_len}, {dim}])"
    )

    # Validate shapes
    assert temb_seq.shape == (batch_size, seq_len, dim), f"temb_seq shape mismatch!"
    assert timestep_proj_seq.shape == (
        batch_size,
        seq_len,
        time_proj_dim,
    ), f"timestep_proj_seq shape mismatch!"
    assert text_emb_seq.shape == (
        batch_size,
        seq_len,
        dim,
    ), f"text_emb_seq shape mismatch!"
    print("✓ All shape assertions passed!")

    # ==================== Test WanTimeEmbedding ====================
    print("\n[Test 3] WanTimeEmbedding - Basic Forward Pass (No Text)")
    print("-" * 50)

    embedding_no_text = WanTimeEmbedding(
        dim=dim,
        time_freq_dim=time_freq_dim,
        time_proj_dim=time_proj_dim,
    )

    timestep = torch.randint(0, 1000, (batch_size,))
    temb, timestep_proj = embedding_no_text(timestep)

    print(f"Input timestep shape: {timestep.shape}")
    print(f"Output temb shape: {temb.shape} (expected: [{batch_size}, {dim}])")
    print(
        f"Output timestep_proj shape: {timestep_proj.shape} (expected: [{batch_size}, {time_proj_dim}])"
    )

    # Validate shapes
    assert temb.shape == (batch_size, dim), f"temb shape mismatch!"
    assert timestep_proj.shape == (
        batch_size,
        time_proj_dim,
    ), f"timestep_proj shape mismatch!"
    print("✓ All shape assertions passed!")

    # ==================== Test WanTimeEmbedding with Sequence Timesteps ====================
    print("\n[Test 4] WanTimeEmbedding - Sequence-level Timesteps")
    print("-" * 50)

    timestep_seq = torch.randint(0, 1000, (batch_size * seq_len,))
    temb_seq, timestep_proj_seq = embedding_no_text(
        timestep_seq, timestep_seq_len=seq_len
    )

    print(f"Input timestep shape: {timestep_seq.shape}")
    print(
        f"Output temb shape: {temb_seq.shape} (expected: [{batch_size}, {seq_len}, {dim}])"
    )
    print(
        f"Output timestep_proj shape: {timestep_proj_seq.shape} (expected: [{batch_size}, {seq_len}, {time_proj_dim}])"
    )

    # Validate shapes
    assert temb_seq.shape == (batch_size, seq_len, dim), f"temb_seq shape mismatch!"
    assert timestep_proj_seq.shape == (
        batch_size,
        seq_len,
        time_proj_dim,
    ), f"timestep_proj_seq shape mismatch!"
    print("✓ All shape assertions passed!")

    # ==================== Test Parameter Count ====================
    print("\n[Test 5] Model Parameter Statistics")
    print("-" * 50)

    def count_parameters(model):
        """Count total and trainable parameters in a model."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total, trainable

    total_text, trainable_text = count_parameters(embedding_with_text)
    total_no_text, trainable_no_text = count_parameters(embedding_no_text)

    print(f"WanTimeTextEmbedding:")
    print(f"  Total parameters: {total_text:,}")
    print(f"  Trainable parameters: {trainable_text:,}")
    print(f"WanTimeEmbedding:")
    print(f"  Total parameters: {total_no_text:,}")
    print(f"  Trainable parameters: {trainable_no_text:,}")

    # ==================== Summary ====================
    print("\n" + "=" * 60)
    print("All tests passed successfully! ✓")
    print("=" * 60)
