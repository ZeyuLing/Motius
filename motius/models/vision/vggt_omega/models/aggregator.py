# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from motius.models.vision.vggt_omega.models.layers.ffn_layers import Mlp
from motius.models.vision.vggt_omega.models.layers.rope_position_encoding import RopePositionEmbedding
from motius.models.vision.vggt_omega.models.layers.block import SelfAttentionBlock
from motius.models.vision.vggt_omega.models.layers.utils import named_apply
from motius.models.vision.vggt_omega.models.layers.vision_transformer import DinoVisionTransformer
from motius.models.vision.vggt_omega.models.layers.vision_transformer import init_weights_vit


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(
            patch_size=patch_size, embed_dim=embed_dim, use_checkpoint=use_checkpoint
        )
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.cached_layer_indices = set(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)
        # patch_embed initialises itself in _build_patch_embed.
        named_apply(init_weights_vit, self.frame_blocks)
        named_apply(init_weights_vit, self.inter_frame_blocks)

    def _run_block(self, block: nn.Module, tokens: torch.Tensor, rope_sincos) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            return checkpoint(block, tokens, rope_sincos, use_reentrant=False)
        return block(tokens, rope_sincos)

    def forward(
        self,
        images: torch.Tensor,
        *, cached_layer_indices=None, progress=None,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        if progress:
            progress('Encode camera frames', 0, num_frames)
        patch_tokens = self.patch_embed(images)
        if progress:
            progress('Encode camera frames', num_frames, num_frames)
            progress('Estimate camera · global layers', 0, self.depth)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        retained_layers = self.cached_layer_indices if cached_layer_indices is None else set(cached_layer_indices)
        outputs = []
        for block_idx in range(self.depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                block_idx,
                self.inter_frame_attention_types[block_idx],
            )
            if block_idx in retained_layers:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)
            if progress:
                progress('Estimate camera · global layers', block_idx+1, self.depth)

        return outputs, self.patch_token_start

    def forward_camera_streamed(self, images, *, frame_batch_size=8, token_chunk_size=4096,
                                query_chunk_size=4096, execute=None, device=None, progress=None, attention_fn=None):
        """All-frame camera evaluation with bounded frame/FFN workspaces.

        ``execute`` optionally stages one module at a time; resource ownership
        lives in the inference pipeline. All inter-frame attention remains global.
        """
        if self.training or torch.is_grad_enabled():
            raise RuntimeError('Streamed camera evaluation requires eval() and disabled gradients')
        if min(frame_batch_size, token_chunk_size, query_chunk_size) < 1:
            raise ValueError('Chunk sizes must be positive')
        run = execute or (lambda module, function, *args: function(*args))
        device = torch.device(device) if device is not None else self.camera_token.device
        batch, frames, channels, height, width = images.shape
        if channels != 3:
            raise ValueError('Expected RGB frames')
        flat_images = images.reshape(batch*frames, channels, height, width)
        camera = slice_expand_and_flatten(self.camera_token.to(device), batch, frames)
        registers = slice_expand_and_flatten(self.register_token.to(device), batch, frames)
        mean = self._resnet_mean.to(device).reshape(1, 3, 1, 1)
        std = self._resnet_std.to(device).reshape(1, 3, 1, 1)
        def encode():
            tokens = None
            for start in range(0, batch*frames, frame_batch_size):
                stop = min(start+frame_batch_size, batch*frames)
                pixels = (flat_images[start:stop].to(device)-mean)/std
                patches = self.patch_embed(pixels)
                if isinstance(patches, dict):
                    patches = patches['x_norm_patchtokens']
                if tokens is None:
                    tokens = torch.empty((batch*frames, self.patch_token_start+patches.shape[1], patches.shape[2]),
                                         dtype=camera.dtype, device=device)
                    tokens[:, :1].copy_(camera)
                    tokens[:, 1:self.patch_token_start].copy_(registers)
                tokens[start:stop, self.patch_token_start:].copy_(patches)
                if progress:
                    if tokens.is_cuda:
                        torch.cuda.current_stream(tokens.device).synchronize()
                    progress('Encode camera frames', stop, batch*frames)
            return tokens
        tokens = run(self.patch_embed, encode)
        rope = run(self.rope_embed, lambda: self.rope_embed(H=height//self.patch_size, W=width//self.patch_size))
        rope = tuple(value.to(device=device, dtype=torch.float32) for value in rope)
        count, dim = tokens.shape[1:]
        final_frame_prefix = None
        if progress:
            progress('Estimate camera · global layers', 0, self.depth)
        for index in range(self.depth):
            block = self.frame_blocks[index]
            def frame_step():
                for start in range(0, batch*frames, frame_batch_size):
                    stop = min(start+frame_batch_size, batch*frames)
                    tokens[start:stop].copy_(block(tokens[start:stop], rope))
            run(block, frame_step)
            if index == self.depth-1:
                # CameraHead consumes this prefix only; never concatenate full
                # patch grids just to immediately discard them in the head.
                final_frame_prefix = tokens.view(batch, frames, count, dim)[:, :, :self.patch_token_start].clone()
            inter = self.inter_frame_blocks[index]
            if self.inter_frame_attention_types[index] == 'global':
                run(inter, lambda: inter.forward_chunked_(tokens.view(batch, frames*count, dim),
                    query_chunk_size=query_chunk_size, ffn_chunk_size=token_chunk_size, attention_fn=attention_fn))
            elif self.inter_frame_attention_types[index] == 'register':
                prefix = tokens.view(batch, frames, count, dim)[:, :, :self.patch_token_start].reshape(batch, frames*self.patch_token_start, dim).contiguous()
                run(inter, lambda: inter.forward_chunked_(prefix, query_chunk_size=query_chunk_size,
                                                         ffn_chunk_size=token_chunk_size, attention_fn=attention_fn))
                tokens.view(batch, frames, count, dim)[:, :, :self.patch_token_start].copy_(prefix.view(batch, frames, self.patch_token_start, dim))
                del prefix
            else:
                raise ValueError('Unknown inter-frame attention type')
            if progress:
                progress('Estimate camera · global layers', index+1, self.depth)
        final_prefix = tokens.view(batch, frames, count, dim)[:, :, :self.patch_token_start]
        return [None]*(self.depth-1)+[torch.cat((final_frame_prefix, final_prefix), dim=-1)], self.patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens = self._run_block(self.frame_blocks[block_idx], tokens, rope_sincos)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens = self._run_block(self.inter_frame_blocks[block_idx], tokens, None)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens = self._run_block(
            self.inter_frame_blocks[block_idx], camera_and_register_tokens, None
        )
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)


def _build_patch_embed(patch_size: int, embed_dim: int, use_checkpoint: bool = False) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        use_checkpoint=use_checkpoint,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
