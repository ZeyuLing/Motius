"""Released MetaCLIP text tower and learned PromptHMR projection."""
import torch
from torch import nn

class ClipTextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        import open_clip
        self.encoder=open_clip.create_model('ViT-L-14-quickgelu',pretrained=None)
        self.encoder.visual=None
        self.encoder.eval()
        for parameter in self.encoder.parameters():parameter.requires_grad=False
        self.ln_proj=nn.Linear(self.encoder.token_embedding.weight.shape[1],1024)

    def forward(self,tokens):
        with torch.no_grad():
            features=self.encoder.encode_text(tokens).detach()
            features/=features.norm(dim=-1,keepdim=True)
        return self.ln_proj(features)
