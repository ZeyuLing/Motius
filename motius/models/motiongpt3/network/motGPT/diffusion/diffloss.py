import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
import math

from . import create_diffusion


class DiffLoss(nn.Module):
    """Diffusion Loss"""
    def __init__(self, target_channels, z_channels, depth, width, num_sampling_steps, 
                 noise_schedule='linear', use_kl=False, learn_sigma=False, sigma_small=True,
                 target_size=None, multi_hidden=False, grad_checkpointing=False,
                 pe_type=None,):
        super(DiffLoss, self).__init__()
        self.in_size = target_size
        self.in_channels = target_channels
        self.multi_hidden = multi_hidden
        out_channels = target_channels + target_channels*learn_sigma  # *2 for vlb loss， while learn_sigma=True
        self.net = SimpleMLPAdaLN(
            in_size=target_size,
            multi_hidden=multi_hidden,
            in_channels=target_channels,
            model_channels=width,
            out_channels=out_channels,  # *2 for vlb loss， while learn_sigma=True
            z_channels=z_channels,
            num_res_blocks=depth,
            grad_checkpointing=grad_checkpointing,
            # pe_type=pe_type,
        )

        self.train_diffusion = create_diffusion(timestep_respacing="", noise_schedule=noise_schedule, 
                                                use_kl=use_kl, learn_sigma=learn_sigma, sigma_small=sigma_small)
        self.gen_diffusion = create_diffusion(timestep_respacing=num_sampling_steps, noise_schedule=noise_schedule, 
                                                use_kl=use_kl, learn_sigma=learn_sigma, sigma_small=sigma_small)

    def forward(self, target, z, mask=None):
        t = torch.randint(0, self.train_diffusion.num_timesteps, (target.shape[0],), device=target.device)
        model_kwargs = dict(c=z)
        loss_dict = self.train_diffusion.training_losses(self.net, target, t, model_kwargs)
        loss = loss_dict["loss"]
        if mask is not None:
            loss = (loss * mask).sum() / mask.sum()
        return loss.mean()#, loss_dict["pred_xstart"]

    def sample(self, z, temperature=1.0, cfg=1.0):
        # diffusion loss sampling
        if not cfg == 1.0:
            bsz = z.shape[0] // 2
            if self.in_size is not None:
                noise = torch.randn(bsz, self.in_size, self.in_channels).cuda()
            else:
                noise = torch.randn(bsz, self.in_channels).cuda()
            noise = torch.cat([noise, noise], dim=0)
            model_kwargs = dict(c=z, cfg_scale=cfg)
            sample_fn = self.net.forward_with_cfg
        else:
            if self.in_size is not None:
                noise = torch.randn(z.shape[0], self.in_size, self.in_channels).cuda()
            else:
                noise = torch.randn(z.shape[0], self.in_channels).cuda()
            model_kwargs = dict(c=z)
            sample_fn = self.net.forward

        # sampled_token_latent0 = self.gen_diffusion.ddim_sample_loop(
        #     sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False)
        sampled_token_latent = self.gen_diffusion.p_sample_loop(
            sample_fn, noise.shape, noise, clip_denoised=False, model_kwargs=model_kwargs, progress=False,
            temperature=temperature
        )

        return sampled_token_latent


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
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
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class ResBlock(nn.Module):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    """

    def __init__(
        self,
        channels, 
        num_heads=4, 
        mlp_ratio: float = 4., # ff_size = int(hidden_dim*mlp_ratio)
        dropout: float = 0.1,
        activation: str = "gelu",
        normalize_before: bool = False,
        use_mlp_layer: bool = True,
    ):
        super().__init__()
        self.channels = channels

        self.in_ln = nn.LayerNorm(channels, eps=1e-6)
        self.mlp = use_mlp_layer
        if self.mlp:
            self.mlp = nn.Sequential(
                nn.Linear(channels, channels, bias=True),
                nn.SiLU(),
                nn.Linear(channels, channels, bias=True),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Conv1d(channels, channels, 3, 1, 1),
                nn.ReLU(),
                nn.Linear(channels, channels, bias=True),
            )
            # from motius.models.motiongpt3.network.motGPT.archs.operator.cross_attention import TransformerEncoderLayer
            # ff_size = int(channels*mlp_ratio)
            # self.hid_encoder_layer = TransformerEncoderLayer(
            #     channels, num_heads, ff_size, dropout, activation, normalize_before,
            #         )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(channels, 3 * channels, bias=True)
        )

    def forward(self, x, y):
        shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(y).chunk(3, dim=-1)
        h = modulate(self.in_ln(x), shift_mlp, scale_mlp)
        h = self.mlp(h)
        # if self.mlp:
        #     h = self.mlp(h)
        # else:
        #     h.permute(1,0,2)
        #     h = self.hid_encoder_layer(h)
        #     h.permute(1,0,2)
        return x + gate_mlp * h


class FinalLayer(nn.Module):
    """
    The final layer adopted from DiT.
    """
    def __init__(self, model_channels, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(model_channels, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(model_channels, out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_channels, 2 * model_channels, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class hiddenEmbedWithAtt(nn.Module):
    def __init__(self, input_dim, output_dim=256, output_latent_size=1, proj_first=False,
                 pe_type=None, num_heads=4, 
                 mlp_ratio: float = 4., # ff_size = int(hidden_dim*mlp_ratio)
                 dropout: float = 0.1,
                 activation: str = "gelu",
                 normalize_before: bool = False,
                 **kwargs):
        
        super().__init__()
        from motius.models.motiongpt3.network.motGPT.archs.operator.cross_attention import TransformerEncoderLayer
        self.proj_first = proj_first
        hidden_dim = output_dim if proj_first else input_dim
        self.cond_proj = nn.Linear(input_dim, output_dim)
        self.hidden_embedding = nn.Parameter(torch.randn(output_latent_size, hidden_dim))

        if pe_type == 'actor':
            from motius.models.motiongpt3.network.motGPT.archs.operator import PositionalEncoding
            dropout = 0.1
            self.hid_pos = PositionalEncoding(hidden_dim, dropout)
        elif pe_type == 'mld':
            from motius.models.motiongpt3.network.motGPT.archs.operator.position_encoding import build_position_encoding
            position_embedding = 'learned'
            self.hid_pos = build_position_encoding(hidden_dim, position_embedding=position_embedding)
        else:
            raise ValueError("Not Support PE type")
        
        ff_size = int(hidden_dim*mlp_ratio)
        self.hid_encoder_layer = TransformerEncoderLayer(
            hidden_dim, num_heads, ff_size, dropout, activation, normalize_before,
                )
        # attn = Attention(z_channels, num_heads=4, qkv_bias=True)

    def forward(self, hidden):
        if self.proj_first:
            hidden = self.cond_proj(hidden)
        bs, seq_len, hidden_dim = hidden.shape
        hidden = hidden.permute(1,0,2)  # [seq_len, bs, hidden_dim/input_dim]
        hidden_dist = torch.tile(self.hidden_embedding[:, None, :], (1, bs, 1))
        hidseq = torch.cat((hidden_dist, hidden), 0)
        hidseq = self.hid_pos(hidseq)  # (seq_len+1, bs, hidden_dim/input_dim)
        hidden_dist = self.hid_encoder_layer(hidseq)[:hidden_dist.shape[0]]
        hidden_dist = hidden_dist.permute(1,0,2)
        if self.proj_first:
            return hidden_dist
        return self.cond_proj(hidden_dist)
    

class SimpleMLPAdaLN(nn.Module):
    """
    The MLP for Diffusion Loss.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param z_channels: channels in the condition.
    :param num_res_blocks: number of residual blocks per downsample.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        z_channels,
        num_res_blocks,
        in_size=None,
        multi_hidden=False,
        grad_checkpointing=False,
    ):
        super().__init__()

        self.in_size = in_size
        self.multi_hidden = multi_hidden
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.grad_checkpointing = grad_checkpointing
        
        self.time_embed = TimestepEmbedder(model_channels)
        self.input_proj = nn.Linear(in_channels, model_channels)
        if not self.multi_hidden:
            self.cond_embed = nn.Linear(z_channels, model_channels)
        else:
            self.cond_embed = hiddenEmbedWithAtt(input_dim=z_channels, 
                                            output_dim=model_channels, 
                                            output_latent_size=self.in_size,
                                            proj_first=False, pe_type='mld', 
                                            num_heads=4, mlp_ratio=4., dropout=0.1)

        res_blocks = []
        for i in range(num_res_blocks):
            res_blocks.append(ResBlock(
                model_channels,
                use_mlp_layer=True,
            ))

        self.res_blocks = nn.ModuleList(res_blocks)
        self.final_layer = FinalLayer(model_channels, out_channels)

        # self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed.mlp[0].weight, std=0.02)
        nn.init.normal_(self.time_embed.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers
        for block in self.res_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def forward(self, x, t, c):
        """
        Apply the model to an input batch.
        :param x: an [N x C] Tensor of inputs.
        :param t: a 1-D batch of timesteps.
        :param c: conditioning from AR transformer.
        :return: an [N x C] Tensor of outputs.
        """

        x = self.input_proj(x)  # [bs,1,model_channels 1024]
        t = self.time_embed(t)  # [bs, model_channels 1024]
        if self.in_size is not None:
            t = t.unsqueeze(1)  # [bs, 1, model_channels 1024]
        # time_emb = self.time_proj(timesteps)
        # time_emb = self.time_embedding(time_emb).unsqueeze(0)
        c = self.cond_embed(c)  # [bs, seq_len, model_channels 1024]
        y = t + c  # [bs, seq_len, model_channels 1024]
        # if self.pe:
        #     x = self.query_pos(x)  # [bs, k, model_channels 1024]
        #     y = self.mem_pos(y)  # [bs, k, model_channels 1024]

        if self.grad_checkpointing and not torch.jit.is_scripting():
            for block in self.res_blocks:
                x = checkpoint(block, x, y)
        else:
            for block in self.res_blocks:
                x = block(x, y)

        return self.final_layer(x, y)

    def forward_with_cfg(self, x, t, c, cfg_scale):
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, c)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)
