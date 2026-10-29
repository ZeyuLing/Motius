# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import warnings

import torch
import torch.nn as nn

from motius.models.vision.vggt_omega.models.aggregator import Aggregator
from motius.models.vision.vggt_omega.models.heads.camera_head import CameraHead
from motius.models.vision.vggt_omega.models.heads.dense_head import DenseHead
from motius.models.vision.vggt_omega.models.heads.text_alignment_head import TextAlignmentHead


class VGGTOmega(nn.Module):
    """Minimal VGGT-Omega inference model for camera and depth prediction."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        enable_camera: bool = True,
        enable_depth: bool = True,
        enable_alignment: bool = False,
        use_checkpoint: bool = False,
        autocast: bool = True,
        autocast_dtype: str = "source",
    ) -> None:
        super().__init__()

        if autocast_dtype not in ("source", "float16"):
            raise ValueError("autocast_dtype must be source or float16")
        self.autocast = autocast
        self.autocast_dtype = autocast_dtype
        self.aggregator = Aggregator(patch_size=patch_size, embed_dim=embed_dim, use_checkpoint=use_checkpoint)
        _warn_if_rope_not_max(self.aggregator)
        self.camera_head = CameraHead(dim_in=2 * embed_dim) if enable_camera else None
        self.dense_head = DenseHead(dim_in=2 * embed_dim, patch_size=patch_size) if enable_depth else None
        self.text_alignment_head = TextAlignmentHead(dim_in=2 * embed_dim) if enable_alignment else None

    def forward(self, images: torch.Tensor, *, camera_only=False, memory_efficient=False,
                execute=None, device=None, progress=None, attention_fn=None,
                frame_batch_size=8, token_chunk_size=4096, query_chunk_size=4096) -> dict[str, torch.Tensor]:
        if memory_efficient and (not camera_only or self.training or torch.is_grad_enabled()):
            raise ValueError("Memory-efficient evaluation requires camera_only, eval() and disabled gradients")
        run = execute or (lambda module, function: function())
        if len(images.shape) == 4:
            images = images.unsqueeze(0)

        if self.autocast:
            target = torch.device(device) if device is not None else images.device
            if target.type == "cuda":
                with torch.cuda.device(target):
                    bf16_supported = torch.cuda.is_bf16_supported()
            else:
                bf16_supported = torch.cuda.is_bf16_supported()
            amp_dtype = torch.float16 if self.autocast_dtype == "float16" else (torch.bfloat16 if bf16_supported else torch.float16)
            aggregator_context = torch.autocast(device_type="cuda", dtype=amp_dtype)
        else:
            aggregator_context = contextlib.nullcontext()
        with aggregator_context:
            if memory_efficient:
                aggregated_tokens_list, patch_token_start = self.aggregator.forward_camera_streamed(
                    images, execute=execute, device=device, progress=progress, attention_fn=attention_fn,
                    frame_batch_size=frame_batch_size, token_chunk_size=token_chunk_size, query_chunk_size=query_chunk_size)
            elif camera_only:
                # CameraHead reads only the last layer. Depth/alignment consume
                # intermediate layers, but FlowHMR never uses either output.
                aggregated_tokens_list, patch_token_start = self.aggregator(
                    images, cached_layer_indices=(self.aggregator.depth-1,), progress=progress)
            else:
                aggregated_tokens_list, patch_token_start = self.aggregator(images, progress=progress)

        final_tokens = aggregated_tokens_list[-1]
        if final_tokens is None:
            raise ValueError("Aggregator did not cache the final layer, which VGGTOmega needs.")

        predictions = {
            "camera_and_register_tokens": final_tokens[:, :, :patch_token_start].contiguous(),
        }
        with torch.autocast(device_type="cuda", enabled=False):
            if self.camera_head is not None:
                predictions["pose_enc"] = run(self.camera_head, lambda: self.camera_head(
                    aggregated_tokens_list,
                    patch_token_start=patch_token_start,
                    query_chunk_size=256 if memory_efficient else None,
                ))

            if self.dense_head is not None and not camera_only:
                depth, depth_conf = self.dense_head(
                    aggregated_tokens_list,
                    images=images,
                    patch_token_start=patch_token_start,
                )
                predictions["depth"] = depth
                predictions["depth_conf"] = depth_conf

            if self.text_alignment_head is not None and not camera_only:
                predictions.update(
                    self.text_alignment_head(
                        aggregated_tokens_list,
                        patch_token_start=patch_token_start,
                    )
                )

        if not self.training:
            predictions["images"] = images
        return predictions


def _warn_if_rope_not_max(aggregator: nn.Module) -> None:
    for name, module in (("aggregator.patch_embed", aggregator.patch_embed), ("aggregator", aggregator)):
        rope_embed = getattr(module, "rope_embed", None)
        normalize_coords = getattr(rope_embed, "normalize_coords", None)
        if normalize_coords != "max":
            warnings.warn(
                f"{name} RoPE normalize_coords is {normalize_coords!r}; "
                "the released VGGT-Omega checkpoint was trained with 'max'.",
                stacklevel=2,
            )
