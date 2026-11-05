"""Original contrastive objective and distributed gradient gather."""

import torch
from torch import nn
import torch.distributed as dist


# Contrastive loss functions
def contrastive_loss(logits: torch.Tensor, offset: int = 0) -> torch.Tensor:
    """Contrastive loss. offset shifts the label for the global-batch case."""
    labels = torch.arange(logits.shape[0], device=logits.device) + offset
    return nn.functional.cross_entropy(logits, labels)


class GatherWithGrad(torch.autograd.Function):
    """all_gather that preserves gradients on the local shard."""

    @staticmethod
    def forward(ctx, tensor):
        world_size = dist.get_world_size()
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor)
        ctx.rank = dist.get_rank()
        ctx.world_size = world_size
        return torch.cat(gathered, dim=0)

    @staticmethod
    def backward(ctx, grad_output):
        chunk_size = grad_output.shape[0] // ctx.world_size
        return grad_output[ctx.rank * chunk_size : (ctx.rank + 1) * chunk_size]


def gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    return GatherWithGrad.apply(tensor)


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    """Symmetric contrastive loss for CLIP-style training."""
    caption_loss = contrastive_loss(similarity)
    image_loss = contrastive_loss(similarity.t())
    return (caption_loss + image_loss) / 2.0
