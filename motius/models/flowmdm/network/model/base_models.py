
import torch
import clip
import torch.nn as nn
from motius.models.flowmdm.network.model.layers import PositionalEncoding, TimestepEmbedder


class TextConditionalModel(nn.Module):
    def __init__(self, latent_dim=256, cond_mode="no_cond", cond_mask_prob=0., dropout=0.0, clip_dim=512, clip_version=None, **kargs):
        super().__init__()
        self.cond_mode = cond_mode
        assert self.cond_mode in ["no_cond", "text"]
        self.cond_mask_prob = cond_mask_prob

        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)

        if cond_mode != 'no_cond':
            if 'text' in cond_mode:
                self.embed_text = nn.Linear(clip_dim, latent_dim)
                print('Loading CLIP...')
                self.clip_version = clip_version
                self.clip_model = self.load_and_freeze_clip(clip_version)
            else:
                raise NotImplementedError("only conditioning with text is implemented for now")

    def parameters_wo_clip(self):
        return [p for name, p in self.named_parameters() if not name.startswith('clip_model.')]

    def load_and_freeze_clip(self, clip_version):
        clip_model, clip_preprocess = clip.load(clip_version, device='cpu',
                                                jit=False)  # Must set jit=False for training
        clip.model.convert_weights(
            clip_model)  # Actually this line is unnecessary since clip by default already on float16

        # Freeze CLIP weights
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False

        return clip_model

    def mask_cond(self, cond, force_mask=False):
        bs, d = cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_mask_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_mask_prob).view(bs, 1)  # 1-> use null_cond, 0-> use real cond
            return cond * (1. - mask)
        else:
            return cond

    def encode_text(self, tokens):
        texts = tokens.to(next(self.parameters()).device)
        return self.clip_model.encode_text(texts).float()


    def compute_embedding(self, x, timesteps, y):
        bs, njoints, nfeats, nframes = x.shape
        multitext_mode = y.get("multi_text", False)
        time_emb = self.embed_timestep(timesteps)
        force_mask = y.get("uncond", False)

        # compute embedding
        if not multitext_mode: # --> single text training (e.g. HumanML3D dataset) / inference
            enc_text = y["text_embeddings"]
            cond_emb = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask))
            cond_emb = cond_emb.unsqueeze(0).expand(nframes, -1, -1) # [T, N, d]
        else: # --> multi-text training / inference (e.g. Babel dataset)
            enc_text = y["text_embeddings"]

            I, N, d = enc_text.shape
            enc_text = enc_text.reshape(-1, enc_text.shape[-1]) # [I*N, d]
            embedded_text = self.embed_text(self.mask_cond(enc_text, force_mask=force_mask)).reshape(I, N, d) # [I, N, d]

            conditions_mask = y['conditions_mask'] # [I, T, N]
            conditions_mask = conditions_mask.unsqueeze(-1).expand(-1, -1, -1, self.latent_dim) # [I, T, N, d]
            cond_emb = torch.zeros(conditions_mask.shape[1:], device=embedded_text.device) # [T, N, d]
            for i in range(I):
                m = conditions_mask[i] # [T, N, d]
                cond_emb = cond_emb + m * embedded_text[i].unsqueeze(0) # [T, N, d] --> [T, N, d]

        return time_emb + cond_emb
