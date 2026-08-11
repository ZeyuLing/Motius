"""ACMDM Flow backbone used by ProjFlow.

The architecture follows the official release, with artifact-local CLIP loading
and no dependency on an external ProjFlow checkout.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
import math
import time
from functools import partial, wraps
from timm.models.vision_transformer import Attention
from .rope import RopeND
from .sampler import ProjFlowSampler


def lengths_to_mask(lengths, max_len=None):
    lengths = torch.as_tensor(lengths)
    maximum = int(max_len or lengths.max().item())
    return torch.arange(maximum, device=lengths.device)[None] < lengths[:, None]


def eval_decorator(function):
    @wraps(function)
    def wrapped(model, *args, **kwargs):
        training = model.training
        model.eval()
        try:
            return function(model, *args, **kwargs)
        finally:
            model.train(training)

    return wrapped


#################################################################################
#                                      ACMDM                                    #
#################################################################################
class ACMDM(nn.Module):
    def __init__(self, input_dim, cond_mode, latent_dim=256, ff_size=1024, num_layers=8,
                 num_heads=4, dropout=0, clip_dim=512,
                 diff_model='Flow', cond_drop_prob=0.1, max_length=49,
                 patch_size=(1, 22), stride_size=(1, 22), num_joint=22,
                 clip_version='ViT-B/32', **kargs):
        super(ACMDM, self).__init__()

        self.input_dim = input_dim
        self.num_joint = num_joint
        self.latent_dim = latent_dim
        self.clip_dim = clip_dim
        self.dropout = dropout

        self.cond_mode = cond_mode
        self.cond_drop_prob = cond_drop_prob

        if self.cond_mode == 'action':
            assert 'num_actions' in kargs
            self.num_actions = kargs.get('num_actions', 1)
            self.encode_action = partial(F.one_hot, num_classes=self.num_actions)
        # --------------------------------------------------------------------------
        # Diffusion
        self.diff_model = diff_model
        if self.diff_model != 'Flow':
            raise ValueError("The released ProjFlow artifact requires an ACMDM Flow prior")
        self.train_diffusion = None
        self.gen_diffusion = ProjFlowSampler()
        # --------------------------------------------------------------------------
        # ACMDM
        self.t_embedder = TimestepEmbedder(self.latent_dim)
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.patches_per_frame = (num_joint - patch_size[1]) // stride_size[1] + 1

        # Patchification
        self.x_embedder = nn.Conv2d(self.input_dim, self.latent_dim, kernel_size=self.patch_size, stride=self.stride_size, bias=True)

        # Positional Encoding
        max_length = max_length * self.patches_per_frame
        self.max_lens = [max_length]
        self.rope = RopeND(nd=1, nd_split=[1], max_lens=self.max_lens)
        self.register_buffer(
            'position_ids_precompute',
            torch.arange(max_length).unsqueeze(0),
            persistent=False,
        )

        self.ACMDMTransformer = nn.ModuleList([
            ACMDMTransBlock(self.latent_dim, num_heads, mlp_size=ff_size, rope=self.rope, qk_norm=True) for _ in range(num_layers)
        ])

        if self.cond_mode == 'text':
            self.y_embedder = nn.Linear(self.clip_dim, self.latent_dim)
        elif self.cond_mode == 'action':
            self.y_embedder = nn.Linear(self.num_actions, self.latent_dim)
        elif self.cond_mode == 'uncond':
            self.y_embedder = nn.Identity()
        else:
            raise KeyError("Unsupported condition mode!!!")

        self.final_layer = FinalLayer(self.latent_dim, self.input_dim, patch_size=patch_size, stride_size=stride_size, patches=self.patches_per_frame, joint=num_joint)

        self.initialize_weights()

        if self.cond_mode == 'text':
            self.clip_version = clip_version
            self.clip_model = self.load_and_freeze_clip(clip_version)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in ACMDM blocks:
        for block in self.ACMDMTransformer:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def load_and_freeze_clip(self, clip_version):
        clip_model, _ = clip.load(clip_version, device='cpu', jit=False)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
        return clip_model

    def encode_text(self, raw_text):
        device = next(self.parameters()).device
        text = clip.tokenize(raw_text, truncate=True).to(device)
        feat_clip_text = self.clip_model.encode_text(text).float()
        return feat_clip_text

    def mask_cond(self, cond, force_mask=False):
        bs, d =  cond.shape
        if force_mask:
            return torch.zeros_like(cond)
        elif self.training and self.cond_drop_prob > 0.:
            mask = torch.bernoulli(torch.ones(bs, device=cond.device) * self.cond_drop_prob).view(bs, 1)
            return cond * (1. - mask)
        else:
            return cond

    def forward(self, x, t, conds, attention_mask, force_mask=False):
        t = self.t_embedder(t, dtype=x.dtype)
        conds = self.mask_cond(conds, force_mask=force_mask)
        x = self.x_embedder(x)
        x = x.flatten(2).transpose(1, 2)
        conds = self.y_embedder(conds)
        y = t.unsqueeze(1) + conds.unsqueeze(1)
        position_ids = self.position_ids_precompute[:, :x.shape[1]]
        for block in self.ACMDMTransformer:
            x = block(x, y, attention_mask, position_ids=position_ids)
        x = self.final_layer(x, y)
        return x

    def forward_with_CFG(self, x, t, conds, attention_mask, cfg=1.0, **kwargs):
        if not cfg == 1.0:
            half = x[: len(x) // 2]
            x = torch.cat([half, half], dim=0)
        x = self.forward(x, t, conds, attention_mask)

        if not cfg == 1.0:
            cond_eps, uncond_eps = torch.split(x, len(x) // 2, dim=0)
            half_eps = uncond_eps + cfg * (cond_eps - uncond_eps)
            x = torch.cat([half_eps, half_eps], dim=0)
        return x

    def forward_loss(self, latents, y, m_lens):
        latents = latents.permute(0, 2, 3, 1)
        b, l, j, d = latents.shape
        device = latents.device

        non_pad_mask = lengths_to_mask(m_lens, l)
        latents = torch.where(non_pad_mask.unsqueeze(-1).unsqueeze(-1), latents, torch.zeros_like(latents))

        target = latents.clone().permute(0, 3, 1, 2).detach()

        force_mask = False
        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(y)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(y).to(device).float()
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
            force_mask = True
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        attention_mask = non_pad_mask.unsqueeze(-1).repeat(1, 1, self.patches_per_frame).flatten(1).unsqueeze(1).unsqueeze(1)

        model_kwargs = dict(conds=cond_vector, force_mask=force_mask, attention_mask=attention_mask)
        if self.diff_model == "Flow":
            loss_dict = self.train_diffusion.training_losses(self.forward, target, model_kwargs)
        else:
            t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
            loss_dict = self.train_diffusion.training_losses(self.forward, target, t, model_kwargs)
        loss = loss_dict["loss"]
        loss = (loss * non_pad_mask).sum() / non_pad_mask.sum()

        return loss

    @torch.no_grad()
    @eval_decorator
    def generate(self,
                 conds,
                 m_lens,
                 cond_scale: int,
                 temperature=1,
                 j=None,
                 ):
        device = next(self.parameters()).device
        if j is None:
            j = self.num_joint
        elif j != self.num_joint:
            raise ValueError(f"generate() expects j={self.num_joint} to match the model configuration, got j={j}.")
        l = max(m_lens)
        b = len(m_lens)

        if self.cond_mode == 'text':
            with torch.no_grad():
                cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            cond_vector = self.enc_action(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(b, self.latent_dim).float().to(device)
        else:
            raise NotImplementedError("Unsupported condition mode!!!")

        padding_mask = ~lengths_to_mask(m_lens, l)

        noise = torch.randn(b, self.input_dim, l, j).to(device)
        if not cond_scale == 1.0:
            cond_vector = torch.cat([cond_vector, torch.zeros_like(cond_vector)], dim=0)
            noise = torch.cat([noise, noise], dim=0)

        attention_mask = (~padding_mask).unsqueeze(-1).repeat(1,1,self.patches_per_frame).flatten(1).unsqueeze(1).unsqueeze(1)
        model_kwargs = dict(conds=cond_vector, attention_mask=attention_mask, cfg=cond_scale)
        sample_fn = self.forward_with_CFG

        if not cond_scale == 1:
            model_kwargs["attention_mask"] = attention_mask.repeat(2, 1, 1, 1)

        if self.diff_model == "Flow":
            model_fn = self.gen_diffusion.sample_ode()  # default to ode sampling
            sampled_token_latent = model_fn(noise, sample_fn, **model_kwargs)[-1]
        else:
            sampled_token_latent = self.gen_diffusion.p_sample_loop(
                sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs,
                progress=False,
                temperature=temperature
            )
        if not cond_scale == 1:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)
        sampled_token_latent = sampled_token_latent.permute(0,2,3,1)

        latents = torch.where(padding_mask.unsqueeze(-1).unsqueeze(-1), torch.zeros_like(sampled_token_latent), sampled_token_latent)
        return latents.permute(0,3,1,2)

    @torch.no_grad()
    @eval_decorator
    def generate_control(
        self,
        conds,
        m_lens,
        control,
        index,
        density,
        cond_scale,
        use_projflow: bool = True,
        ):
        """
        Inputs
        ------
        conds      : raw text/action inputs (batch-aligned with m_lens)
        m_lens     : (B,) valid lengths on the raw time grid
        control    : (B, Cc, T, J) control signal on the raw grid
        index      : multi-joint spec, one of:
                    - int                         : same single joint for all samples
                    - List[int]/Tuple[int]        : same joint set for all samples
                    - torch.Tensor(K,)            : same K joints for all samples
                    - torch.Tensor(B, K)          : per-sample K joints
                    - List[List[int]]             : per-sample joint sets; len(index) == B
        density    : if in {1, 2, 5} -> absolute #frames to constrain;
                    else -> percentage [0..100] of valid frames to constrain
        cond_scale : classifier-free guidance scale (float)
        use_projflow : if False, disable metric coupling, halo augmentation, and noise mixing.

        Returns
        -------
        samples_raw : (B, D, T, J) sampled outputs on the raw grid
        control_mask: (B, D, T, J) binary mask of constrained positions
        """
        device = next(self.parameters()).device
        assert control.dim() == 4, f"'control' must be (B,Cc,T,J), got {control.shape}"
        B, Cc, T, J = control.shape
        D, L = self.input_dim, T
        assert D % Cc == 0, f"input_dim ({D}) must be divisible by Cc ({Cc})"

        if self.cond_mode == 'text':
            cond_vector = self.encode_text(conds)
        elif self.cond_mode == 'action':
            fn = getattr(self, "encode_action", None) or getattr(self, "enc_action", None)
            if fn is None:
                raise AttributeError("Action mode requires self.encode_action (or self.enc_action).")
            cond_vector = fn(conds).to(device)
        elif self.cond_mode == 'uncond':
            cond_vector = torch.zeros(B, self.latent_dim, device=device)
        else:
            raise NotImplementedError("Unsupported condition mode.")

        cond_vector = cond_vector.float()

        padding_mask = ~lengths_to_mask(m_lens, L)
        attention_mask = (
            (~padding_mask)
            .unsqueeze(-1).repeat(1, 1, self.patches_per_frame)
            .flatten(1).unsqueeze(1).unsqueeze(1)    # (B,1,1,L*patches)
        )

        noise = torch.randn(B, D, L, J, device=device)

        def _standardize_indices(idx_spec, B_, J_, dev):
            """
            Normalize 'index' into a list of length B_, where each item is a 1D LongTensor
            of unique joint indices on device 'dev', clamped into [0, J_-1].
            """
            def _to_tensor(vec):
                t = torch.as_tensor(vec, device=dev, dtype=torch.long).view(-1)
                if t.numel() == 0:
                    return torch.zeros(0, device=dev, dtype=torch.long)
                t = torch.unique(t.clamp_(0, J_ - 1))
                return t

            if isinstance(idx_spec, int):
                return [_to_tensor([idx_spec]) for _ in range(B_)]

            if torch.is_tensor(idx_spec):
                if idx_spec.dim() == 0:
                    return [_to_tensor([int(idx_spec.item())]) for _ in range(B_)]
                if idx_spec.dim() == 1:
                    return [_to_tensor(idx_spec) for _ in range(B_)]
                if idx_spec.dim() == 2:
                    assert idx_spec.size(0) == B_, "index.shape[0] must equal batch size"
                    return [_to_tensor(idx_spec[b]) for b in range(B_)]
                raise ValueError("Unsupported tensor shape for 'index'.")

            if isinstance(idx_spec, (list, tuple)):
                if len(idx_spec) == 0:
                    raise ValueError("index must not be empty")
                # Flat list of ints -> same for all samples
                if all(isinstance(x, int) for x in idx_spec):
                    return [_to_tensor(idx_spec) for _ in range(B_)]
                # List of lists/tuples/tensors -> per-sample sets
                if len(idx_spec) == B_ and all(isinstance(x, (list, tuple, torch.Tensor)) for x in idx_spec):
                    return [_to_tensor(x) for x in idx_spec]
                raise ValueError("Ambiguous 'index' specification. Provide int, List[int], List[List[int]], Tensor(K,) or Tensor(B,K).")

            raise ValueError("Unsupported type for 'index'.")

        per_sample_joints = _standardize_indices(index, B, J, device)  # List[Tensor], len B

        mask_seq = torch.zeros((B, D, T, J), device=device)
        for b in range(B):
            pool = int(torch.as_tensor(m_lens[b]).item())
            if density in (1, 2, 5):
                k = int(density)
            else:
                k = max(1, int(pool * float(density) / 100.0))

            frames = torch.sort(torch.randperm(pool, device=device)[:k]).values
            joints_b = per_sample_joints[b]
            if joints_b.numel() == 0:
                # If no joints provided for this sample, skip (mask remains zero)
                continue
            # Activate all requested joints for the selected frames (broadcast over D)
            mask_seq[b, :, frames.unsqueeze(-1), joints_b] = 1.0

        # Repeat control along channel dimension to match the model's D channels.
        control_D = control.repeat(1, D // Cc, 1, 1)  # (B, D, T, J)
        y = control_D * mask_seq                      # (B, D, T, J)

        cfg = float(cond_scale) if isinstance(cond_scale, (int, float)) else float(cond_scale[0])
        if cfg == 1.0:
            model_kwargs = dict(
                conds=cond_vector, attention_mask=attention_mask, cfg=cfg,
                A=mask_seq, y=y
            )
        else:
            zeros_like_cond = torch.zeros_like(cond_vector)
            model_kwargs = dict(
                conds=torch.cat([cond_vector, zeros_like_cond], dim=0),
                attention_mask=attention_mask.repeat(2, 1, 1, 1),
                cfg=cfg,
                A=mask_seq.repeat(2, 1, 1, 1),
                y=y.repeat(2, 1, 1, 1),
            )
            noise = torch.cat([noise, noise], dim=0)

        sample_fn = self.forward_with_CFG
        if self.diff_model != "Flow":
            raise NotImplementedError("This sampler is implemented for Flow (velocity) models only.")

        model_fn = self.gen_diffusion.sample_projflow(num_steps=100, use_projflow=use_projflow)
        x_raw = model_fn(noise, sample_fn, **model_kwargs)[-1]
        if cfg != 1.0:
            x_raw, _ = x_raw.chunk(2, dim=0)

        x_raw = x_raw.permute(0, 2, 3, 1)
        samples_raw = torch.where(
            padding_mask.unsqueeze(-1).unsqueeze(-1),
            torch.zeros_like(x_raw),
            x_raw
        )

        return samples_raw.permute(0, 3, 1, 2), mask_seq

#################################################################################
#                                     ACMDM Zoos                                #
#################################################################################
def acmdm_raw_flow_s_ps22(**kwargs):
    layer = 8
    return ACMDM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196,
                 patch_size=(1, 22), stride_size=(1, 22), **kwargs)
def acmdm_raw_flow_s_ps2(**kwargs):
    layer = 8
    return ACMDM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196,
                 patch_size=(1, 2), stride_size=(1, 2), **kwargs)
def acmdm_raw_flow_xl_ps2(**kwargs):
    layer = 20
    return ACMDM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=196,
                 patch_size=(1, 2), stride_size=(1, 2), **kwargs)
def acmdm_flow_s_ps22(**kwargs):
    layer = 8
    return ACMDM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=49,
                 patch_size=(1, 22), stride_size=(1, 22), **kwargs)
def acmdm_flow_xl_ps2(**kwargs):
    layer = 20
    return ACMDM(latent_dim=layer*64, ff_size=layer*64*4, num_layers=layer, num_heads=layer, dropout=0, clip_dim=512,
                 diff_model="Flow", cond_drop_prob=0.1, max_length=49,
                 patch_size=(1, 2), stride_size=(1, 2), **kwargs)

ACMDM_models = {
    'ACMDM-Raw-Flow-S-PatchSize22': acmdm_raw_flow_s_ps22, 'ACMDM-Flow-S-PatchSize22': acmdm_flow_s_ps22,
    'ACMDM-Flow-XL-PatchSize2': acmdm_flow_xl_ps2, 'ACMDM-Raw-Flow-S-PatchSize2': acmdm_raw_flow_s_ps2,
    'ACMDM-Raw-Flow-XL-PatchSize2': acmdm_raw_flow_xl_ps2
}

#################################################################################
#                                 Inner Architectures                           #
#################################################################################
def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class ACMDMAttention(Attention):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=True,
        rope=None,
        qk_norm=True,
        **block_kwargs,
    ):
        super().__init__(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_norm=qk_norm, **block_kwargs)
        self.rope = rope

    def forward(self, x, position_ids=None, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q, k, position_ids)

        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_drop.p
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x):
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


class ACMDMTransBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_size=1024, rope=None, qk_norm=True):
        super().__init__()
        self.norm1 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.attn = ACMDMAttention(hidden_size, num_heads=num_heads, qkv_bias=True, norm_layer=LlamaRMSNorm,
                                        qk_norm=qk_norm, rope=rope)
        self.norm2 = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, int(2 / 3 * mlp_size))
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c, attention_mask=None, position_ids=None):
        dtype = x.dtype
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        norm_x1 = self.norm1(x.to(torch.float32)).to(dtype)
        attn_input_x = modulate(norm_x1, shift_msa, scale_msa)
        attn_output_x = self.attn(attn_input_x, attention_mask=attention_mask, position_ids=position_ids)
        x = x + gate_msa * attn_output_x

        norm_x2 = self.norm2(x.to(torch.float32)).to(dtype)
        gate_input_x = modulate(norm_x2, shift_mlp, scale_mlp)
        gate_output_x = self.mlp(gate_input_x)
        x = x + gate_mlp * gate_output_x
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, output_size, patch_size=(1, 22), stride_size=(1,22), patches=1, joint=22):
        super().__init__()
        self.norm_final = LlamaRMSNorm(hidden_size, eps=1e-6)
        self.patch_size = patch_size
        self.stride_size = stride_size
        self.patches = patches
        self.joint=joint
        self.linear = nn.Linear(hidden_size, output_size*patch_size[0]*patch_size[1], bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        norm_x = self.norm_final(x.to(torch.float32)).to(x.dtype)
        x = modulate(norm_x, shift, scale)
        x = self.linear(x)
        x = x.reshape(shape=(x.shape[0], x.shape[1]//self.patches, self.patches, self.patch_size[0], self.patch_size[1], x.shape[-1] // self.patch_size[1]))
        x = torch.einsum('nljpqc->nclpjq', x)
        x = x.reshape(shape=(x.shape[0], x.shape[1], -1, self.joint))
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000, dtype=torch.float32):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=dtype) / half
        ).to(device=t.device, dtype=dtype)
        args = t[:, None] * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, dtype=torch.bfloat16):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size, dtype=dtype)
        t_emb = self.mlp(t_freq)
        return t_emb


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)
