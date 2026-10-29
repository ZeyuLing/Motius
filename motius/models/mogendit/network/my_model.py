import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


def get_window_mask(
    seq_len: int, window_size: int, device: torch.device
) -> torch.Tensor:
    half_win = window_size // 2  # 窗口半宽（如window_size=3 → 半宽1，左右各1个位置）
    # 生成序列索引：[0,1,2,...,seq_len-1]
    idx = torch.arange(seq_len, device=device)
    # 计算每个位置i与所有位置j的绝对距离：[seq_len, seq_len]
    dist = torch.abs(idx.unsqueeze(0) - idx.unsqueeze(1))
    # 窗口内（距离≤半宽）设为0，窗口外设为-1e9
    window_mask = torch.where(dist <= half_win, 0.0, -1e9)
    # 扩展维度以匹配后续合并需求：[1, 1, seq_len, seq_len]
    window_mask = window_mask.unsqueeze(0)
    return window_mask


class DiTBlock(nn.Module):
    # Description-Mask Conditioned Encoder
    """
    支持常规token序列、条件特征token和mask_label的Transformer编码器块

    Args:
        d_model: 输入特征维度
        nhead: 注意力头数
        dim_feedforward: 前馈网络隐藏层维度
        dropout: dropout概率
        activation: 激活函数类型
    """

    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_head: int,
        dropout: float = 0.1,
        activation: str = "silu",
        window_size=None,
    ):
        super(DiTBlock, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.window_size = window_size if window_size is not None else -1

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=False
        )
        self.adaLN = AdaLN(d_model=d_model)

        self.timestep_embedding = TimestepEmbedder(d_model=d_model, max_timesteps=1000)

        # 条件归一化层（支持mask_label）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.cond_embed = nn.Linear(d_cond, d_model)
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Dropout(dropout),
            _get_activation_fn(activation),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        src: torch.Tensor,  # 常规token序列 [seq_len, batch_size, d_model]
        t: torch.Tensor,  # 扩散的时间步
        cond: torch.Tensor,  # 单个条件特征token [1, batch_size, d_model]
        padding_mask: Optional[
            torch.Tensor
        ] = None,  # 新增: padding mask [batch_size, seq_len] (1表示填充)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 1. 融合条件特征token到序列中
        cond_emb = self.cond_embed(cond)
        time_emb = self.timestep_embedding(t)
        cond_fusion = cond_emb + time_emb

        gate_msa, shift_msa, scale_msa, gate_ffn, shift_ffn, scale_ffn = self.adaLN(
            cond_fusion
        )  # c包含时间步嵌入

        attn_mask = None
        seq_len = src.shape[0]

        # 1. 生成窗口可见范围mask（类似卷积核局部视野）
        if self.window_size > 0:
            attn_mask = get_window_mask(
                seq_len=seq_len,
                window_size=self.window_size,
                device=src.device,
            )  # [1,seq_len,seq_len]

        # 2. 处理原有padding mask（若存在则与窗口mask合并）
        if padding_mask is not None:
            padding_mask_expanded = padding_mask.unsqueeze(
                1
            )  # [batch_size, 1, seq_len]
            padding_mask_expanded = padding_mask_expanded.expand(
                -1, seq_len, -1
            )  # [batch_size, seq_len, seq_len]
            padding_mask_expanded = padding_mask_expanded.to(dtype=src.dtype) * -1e9
            padding_mask_expanded = padding_mask_expanded.unsqueeze(
                1
            )  # [batch_size, 1, seq_len, seq_len]
            padding_mask_expanded = padding_mask_expanded.repeat(1, self.n_head, 1, 1)
            padding_mask_expanded = padding_mask_expanded.view(-1, seq_len, seq_len)

            # 合并窗口mask和padding mask（相加实现“双重禁止”）
            if attn_mask is not None:
                # 窗口mask扩展到batch维度：[1,1,seq_len,seq_len] → [batch_size,1,seq_len,seq_len]
                attn_mask = attn_mask.expand_as(padding_mask_expanded)
                attn_mask = attn_mask + padding_mask_expanded  # 合并两种mask
            else:
                attn_mask = padding_mask_expanded
        else:
            attn_mask = (
                attn_mask.repeat(src.size(1) * self.n_head, 1, 1)
                if attn_mask is not None
                else None
            )

        x = ada_shift_scale(x=self.norm1(src), shift=shift_msa, scale=scale_msa)
        x = x + gate_msa * self.self_attn(x, x, x, attn_mask=attn_mask)[0]

        # 3. 前馈网络
        x = ada_shift_scale(x=self.norm2(x), shift=shift_ffn, scale=scale_ffn)
        x = x + gate_ffn * self.ffn(x)

        return x


class MoGenDiT_CatMask(nn.Module):
    def __init__(
        self,
        d_motion: int,
        d_model: int,
        d_cond: int,
        n_head: int,
        n_stack: int,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super(MoGenDiT_CatMask, self).__init__()
        # self.pe = LearnablePositionEncoding(d_model=d_model, max_seq_len=512)
        self.pe = PositionalEncoding(d_model=d_model, max_len=5000)
        self.motion_2_token = Embedder(d_motion * 2, d_model)
        self.token_2_motion = nn.Linear(d_model, d_motion)
        nn.init.zeros_(self.token_2_motion.weight)
        nn.init.zeros_(self.token_2_motion.bias)

        self.encoder_blocks = []
        for _ in range(n_stack):
            self.encoder_blocks.append(
                DiTBlock(
                    d_model=d_model,
                    d_cond=d_cond,
                    n_head=n_head,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

    def wrap_inputs(
        self,
        x: torch.Tensor,  # 常规token序列 [batch_size, seq_len, d_model
        cond: torch.Tensor,  # 单个条件特征token [batch_size, 1, d_model]
        mask: Optional[
            torch.Tensor
        ] = None,  # mask标签 [batch_size, seq_len, d_model] (1表示掩码)
        padding_mask: Optional[
            torch.Tensor
        ] = None,  # mask标签 [batch_size, seq_len] (1表示掩码)
    ) -> dict:
        return {"x_t": x, "cond": cond, "mask": mask, "padding_mask": padding_mask}

    def forward(
        self,
        x_wrapped: dict,  # 常规token序列 [batch_size, seq_len, d_model]
        t: torch.Tensor,  # 扩散的时间步
    ) -> torch.Tensor:

        x = x_wrapped["x_t"].permute(1, 0, 2)
        cond = x_wrapped["cond"].permute(1, 0, 2)
        mask = x_wrapped["mask"].permute(1, 0, 2)
        padding_mask = x_wrapped["padding_mask"]

        x = torch.cat([x, mask], dim=-1)

        x = self.motion_2_token(x)
        x = self.pe(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, t, cond, padding_mask)
        x = self.token_2_motion(x).permute(1, 0, 2)

        return x


class MoGenDiT_V2(nn.Module):
    def __init__(
        self,
        d_motion: int,
        d_model: int,
        d_cond: int,
        n_head: int,
        n_stack: int,
        dropout: float = 0.1,
        activation: str = "silu",
        window_size=None,
    ):
        super(MoGenDiT_V2, self).__init__()
        # self.pe = LearnablePositionEncoding(d_model=d_model, max_seq_len=512)
        self.pe = PositionalEncoding(d_model=d_model, max_len=5000)
        self.motion_2_token = nn.Linear(d_motion * 2, d_model)
        self.token_2_motion = nn.Linear(d_model, d_motion)
        nn.init.zeros_(self.token_2_motion.weight)
        nn.init.zeros_(self.token_2_motion.bias)

        self.encoder_blocks = []
        for _ in range(n_stack):
            self.encoder_blocks.append(
                DiTBlock(
                    d_model=d_model,
                    d_cond=d_cond,
                    n_head=n_head,
                    dropout=dropout,
                    activation=activation,
                    window_size=window_size,
                )
            )
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

    def wrap_inputs(
        self,
        x: torch.Tensor,  # 常规token序列 [batch_size, seq_len, d_model
        cond: torch.Tensor,  # 单个条件特征token [batch_size, 1, d_model]
        mask: Optional[
            torch.Tensor
        ] = None,  # mask标签 [batch_size, seq_len, d_model] (1表示掩码)
        padding_mask: Optional[
            torch.Tensor
        ] = None,  # mask标签 [batch_size, seq_len] (1表示掩码)
    ) -> dict:
        return {"x_t": x, "cond": cond, "mask": mask, "padding_mask": padding_mask}

    def forward(
        self,
        x_wrapped: dict,  # 常规token序列 [batch_size, seq_len, d_model]
        t: torch.Tensor,  # 扩散的时间步
    ) -> torch.Tensor:

        x = x_wrapped["x_t"].permute(1, 0, 2)
        cond = x_wrapped["cond"].permute(1, 0, 2)
        mask = x_wrapped["mask"].permute(1, 0, 2)
        padding_mask = x_wrapped["padding_mask"]

        x = torch.cat([x, mask], dim=-1)

        x = self.motion_2_token(x)
        x = self.pe(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, t, cond, padding_mask)
        x = self.token_2_motion(x).permute(1, 0, 2)

        return x


class EncoderBlock(nn.Module):
    # Description-Mask Conditioned Encoder
    """
    支持常规token序列、条件特征token和mask_label的Transformer编码器块

    Args:
        d_model: 输入特征维度
        nhead: 注意力头数
        dim_feedforward: 前馈网络隐藏层维度
        dropout: dropout概率
        activation: 激活函数类型
    """

    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
        activation: str = "silu",
    ):
        super(EncoderBlock, self).__init__()
        self.d_model = d_model
        self.n_head = n_head

        # 自注意力层
        self.self_attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=False
        )

        # 条件归一化层（支持mask_label）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Dropout(dropout),
            _get_activation_fn(activation),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(
        self,
        src: torch.Tensor,  # 常规token序列 [seq_len, batch_size, d_model]
        padding_mask: Optional[
            torch.Tensor
        ] = None,  # 新增: padding mask [batch_size, seq_len] (1表示填充)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # 自注意力计算 - 应用padding mask
        if padding_mask is not None:
            seq_len = padding_mask.shape[-1]
            attn_mask = padding_mask.unsqueeze(1)  # [batch_size, 1, seq_len]
            attn_mask = attn_mask.expand(
                -1, seq_len, -1
            )  # [batch_size, seq_len, seq_len]
            attn_mask = attn_mask.to(dtype=src.dtype) * -1e9  # 填充位置设为负无穷
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.n_head, 1, 1)
            attn_mask = attn_mask.view(-1, seq_len, seq_len)
        else:
            # 无padding mask时的常规自注意力计算
            attn_mask = None

        x = self.norm1(src)
        x = self.self_attn(x, x, x, attn_mask=attn_mask)[0]

        # 3. 前馈网络
        x = self.norm2(x)
        x = x + self.ffn(x)

        return x


class MDM(nn.Module):
    def __init__(
        self,
        d_motion: int,
        d_model: int,
        d_cond: int,
        n_head: int,
        n_stack: int,
        dropout: float = 0.1,
        activation: str = "silu",
        feat_res=False,
    ):
        super(MDM, self).__init__()
        # self.pe = LearnablePositionEncoding(d_model=d_model, max_seq_len=512)
        self.feat_res = feat_res
        self.pe = PositionalEncoding(d_model=d_model, max_len=5000)
        self.timestep_embedding = TimestepEmbedder(d_model=d_model, max_timesteps=1000)
        self.cond_embed = nn.Linear(d_cond, d_model)
        self.motion_2_token = Embedder(d_motion * 2, d_model)
        self.token_2_motion = nn.Linear(d_model, d_motion)
        nn.init.zeros_(self.token_2_motion.weight)
        nn.init.zeros_(self.token_2_motion.bias)

        self.encoder_blocks = []
        for _ in range(n_stack):
            self.encoder_blocks.append(
                EncoderBlock(
                    d_model=d_model,
                    n_head=n_head,
                    dropout=dropout,
                    activation=activation,
                )
            )
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)

    def wrap_inputs(
        self,
        x: torch.Tensor,  # 常规token序列 [batch_size, seq_len, d_model
        cond: torch.Tensor,  # 单个条件特征token [batch_size, 1, d_model]
        mask: Optional[
            torch.Tensor
        ] = None,  # mask标签 [batch_size, seq_len, d_model] (1表示掩码)
        padding_mask: Optional[
            torch.Tensor
        ] = None,  # mask标签 [batch_size, seq_len] (1表示掩码)
    ) -> dict:
        return {"x_t": x, "cond": cond, "mask": mask, "padding_mask": padding_mask}

    def forward(
        self,
        x_wrapped: dict,  # 常规token序列 [batch_size, seq_len, d_model]
        t: torch.Tensor,  # 扩散的时间步
    ) -> torch.Tensor:

        x = x_wrapped["x_t"].permute(1, 0, 2)
        cond = x_wrapped["cond"].permute(1, 0, 2)
        mask = x_wrapped["mask"].permute(1, 0, 2)
        padding_mask = x_wrapped["padding_mask"]

        x = torch.cat([x, mask], dim=-1)
        x = self.motion_2_token(x)

        # 拼接z token
        cond_emb = self.cond_embed(cond)
        time_emb = self.timestep_embedding(t)
        z = cond_emb + time_emb  # [1, batch_size, d_model]
        x = torch.cat([z, x], dim=0)  # 在时间维度拼接
        if padding_mask is not None:
            pad_z = torch.zeros_like(padding_mask[:, :1])  # z对应的padding全为0
            padding_mask = torch.cat([pad_z, padding_mask], dim=1)  #

        x = self.pe(x)
        for encoder_block in self.encoder_blocks:
            # pdb.set_trace()
            x = encoder_block(x, padding_mask)

        x = x[1:, :, :]  # 去掉z token对应的输出
        x = self.token_2_motion(x).permute(1, 0, 2)

        return x


def _get_activation_fn(activation: str):
    """获取激活函数"""
    activation = activation.lower()
    if activation == "relu":
        return nn.ReLU()
    elif activation == "gelu":
        return nn.GELU()
    elif activation == "silu":
        return nn.SiLU()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    else:
        raise RuntimeError(f"不支持的激活函数: {activation}")


class TimestepEmbedder(nn.Module):
    """
    时间步编码器，将扩散过程中的时间步转换为特征嵌入
    适用于DiT等扩散模型，将离散时间步映射到与模型维度匹配的特征空间
    """

    def __init__(self, d_model: int, max_timesteps: int = 1000):
        super().__init__()
        self.d_model = d_model

        # 首先使用正弦余弦函数对时间步进行编码（类似原始Transformer位置编码）
        self.freqs = nn.Parameter(
            torch.exp(torch.linspace(0, math.log(10000.0), d_model // 2) * -1),
            requires_grad=False,
        )

        # 用于进一步处理时间步嵌入的MLP
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model, bias=True),
            nn.SiLU(),
            nn.Linear(d_model, d_model, bias=True),
        )

        # 预计算时间步范围（0到max_timesteps-1）
        self.max_timesteps = max_timesteps

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """
        将时间步转换为嵌入向量

        参数:
            timesteps: 形状为 [batch_size] 的时间步张量

        返回:
            时间步嵌入，形状为 [1, batch_size, d_model]（适配DiT的输入格式）
        """
        # 确保时间步在有效范围内
        if (timesteps < 0).any() or (timesteps >= self.max_timesteps).any():
            raise ValueError(f"时间步必须在 [0, {self.max_timesteps - 1}] 范围内")

        # 计算正弦余弦编码
        x = timesteps.unsqueeze(1).float()  # [batch_size, 1]
        emb = torch.cat(
            [
                torch.sin(x * self.freqs),  # 正弦部分
                torch.cos(x * self.freqs),  # 余弦部分
            ],
            dim=1,
        )  # [batch_size, d_model]

        # 通过MLP进一步处理
        emb = self.mlp(emb)  # [batch_size, d_model]

        # 调整维度以适配 [1, batch_size, d_model] 格式（方便与序列特征广播相加）
        return emb.unsqueeze(0)


class Embedder(nn.Module):
    def __init__(self, n_input, d_model):
        super().__init__()
        self.embed = nn.Linear(n_input, d_model)
        self.d_model = d_model
        self.embed_scale = math.sqrt(self.d_model)

    def forward(self, x):
        """
        :param x: tokenlized sequence
        :return:
        """
        # 乘以一个较大的系数，放大词嵌入向量，
        # 希望与位置编码向量相加后，词嵌入向量本身的影响更大
        return self.embed(x) * self.embed_scale


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        # 生成位置编码，初始形状为 (max_len, 1, d_model)
        position = torch.arange(max_len).unsqueeze(1)  # 形状: (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )  # 形状: (d_model//2,)
        pe = torch.zeros(max_len, 1, d_model)  # 形状: (max_len, 1, d_model)

        # 填充正弦余弦编码
        pe[:, 0, 0::2] = torch.sin(position * div_term)  # 偶数维度用正弦
        pe[:, 0, 1::2] = torch.cos(position * div_term)  # 奇数维度用余弦

        # 不再进行permute，保持形状为 (max_len, 1, d_model)，适配(time, batch, rep_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 输入x的形状为 (time, batch, rep_dim)
        # pe的形状为 (max_len, 1, d_model)，取前x.size(0)个时间步的编码
        x = x + self.pe[: x.size(0), :, :].requires_grad_(False)  # 广播适配batch维度
        return x


class AdaLN(nn.Module):
    def __init__(self, d_model):
        super(AdaLN, self).__init__()
        self.d_model = d_model
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_model, 6 * d_model, bias=True)
        )
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        gate_msa, shift_msa, scale_msa, gate_mlp, shift_mlp, scale_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        return (
            gate_msa,
            shift_msa,
            scale_msa,
            gate_mlp,
            shift_mlp,
            scale_mlp,
        )


class ShiftScaleLN(nn.Module):
    def __init__(self, d_in, d_out):
        super(ShiftScaleLN, self).__init__()
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(d_in, 2 * d_out, bias=True)
        )
        self.sigmoid = nn.Sigmoid()
        nn.init.zeros_(self.adaLN_modulation[1].weight)
        nn.init.zeros_(self.adaLN_modulation[1].bias)

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        return shift, self.sigmoid(scale)


def ada_shift_scale(x, shift, scale):
    return x * (scale + 1) + shift


# # 使用示例
# if __name__ == "__main__":
#     # 配置参数
#     batch_size = 2
#     seq_len = 10
#     d_model = 256
#     n_head = 4
#     d_cond = 50
#     t = torch.FloatTensor([300])
#
#     model = MoGenDiT(d_motion=263, d_model=d_model, d_cond=d_cond, n_head=n_head, n_stack=4)
#
#     # 生成测试数据
#     src = torch.randn(seq_len, batch_size, 263)  # 常规token序列
#     cond = torch.randn(1, batch_size, d_cond)  # 条件特征token
#     mask_label = torch.randint(0, 2, (seq_len, batch_size, 263)).float()  # mask标签（0/1）
#
#     # 前向传播
#     output = model(x=src, t=t, cond=cond, mask=mask_label, keep_mask_state=True)
#     print(f"输入形状: {src.shape}")
#     print(f"输出形状: {output.shape}")  # 应与输入形状一致
