"""SentenceT5 neural backbone, mean pooling, projection and normalization."""

import torch
from torch import nn
from torch.nn import functional as F


class SentenceT5Encoder(nn.Module):
    def __init__(self, backbone, projection):
        super().__init__()
        self.backbone = backbone
        self.projection = projection

    def forward(self, input_ids, attention_mask):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=False
        )[0]
        mask = (
            attention_mask.unsqueeze(-1).expand(embeddings.size()).to(embeddings.dtype)
        )
        pooled = torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)
        return F.normalize(self.projection(pooled), p=2, dim=1)
