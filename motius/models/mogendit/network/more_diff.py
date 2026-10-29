import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


# ===================== 新增：RoPE模块（适配seq-first格式） =====================
class RoPE(nn.Module):
    def __init__(
        self,
        head_dim: int,  # 单个注意力头的维度（d_model // n_head）
        max_seq_len: int = 5000,
        base: int = 10000,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # 预计算频率矩阵（theta）：shape [head_dim//2]
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, head_dim, 2).float() / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)  # 不参与参数更新

    def _compute_rotary_angles(self, seq_len: int, device: torch.device):
        """计算旋转角度：shape [seq_len, head_dim//2]"""
        positions = (
            torch.arange(seq_len, device=device).float().unsqueeze(1)
        )  # [seq_len, 1]
        angles = positions * self.inv_freq.unsqueeze(0)  # [seq_len, head_dim//2]
        return angles

    @staticmethod
    def apply_rotary_emb(x: torch.Tensor, angles: torch.Tensor):
        """
        对Q/K应用旋转编码
        x: [seq_len, batch_size, n_head, head_dim]（seq-first格式）
        angles: [seq_len, head_dim//2]
        return: 旋转后的x，维度不变
        """
        # 拆分奇偶维度：[seq_len, batch, n_head, head_dim//2]
        x1, x2 = x[..., ::2], x[..., 1::2]

        # 扩展角度维度适配：[seq_len, 1, 1, head_dim//2]（匹配batch和n_head维度）
        cos_angles = torch.cos(angles).unsqueeze(1).unsqueeze(2)
        sin_angles = torch.sin(angles).unsqueeze(1).unsqueeze(2)

        # 旋转公式：x1' = x1*cos - x2*sin; x2' = x1*sin + x2*cos
        rotated_x1 = x1 * cos_angles - x2 * sin_angles
        rotated_x2 = x1 * sin_angles + x2 * cos_angles

        # 拼接回原维度
        rotated_x = torch.cat([rotated_x1, rotated_x2], dim=-1)
        return rotated_x

    def forward(self, x: torch.Tensor):
        """
        前向传播：适配seq-first的多头格式输入
        x: [seq_len, batch_size, n_head, head_dim]
        """
        assert (
            len(x.shape) == 4
        ), f"输入维度必须是4维 [seq_len, batch, n_head, head_dim]，当前是{len(x.shape)}维"
        seq_len, batch_size, n_head, head_dim = x.shape
        assert (
            head_dim == self.head_dim
        ), f"单头维度不匹配：输入{head_dim} vs 初始化{self.head_dim}"
        assert (
            seq_len <= self.max_seq_len
        ), f"序列长度{seq_len}超过最大支持长度{self.max_seq_len}"

        # 计算当前序列长度的旋转角度
        angles = self._compute_rotary_angles(seq_len, x.device)

        # 应用旋转编码
        x_rot = self.apply_rotary_emb(x, angles)
        return x_rot


# ===================== 修正：get_window_mask（确保返回4维mask） =====================
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
    # 修正：扩展为4维 [1, 1, seq_len, seq_len]（匹配后续padding mask的维度）
    window_mask = window_mask.unsqueeze(0).unsqueeze(0)  # 先加batch维度，再加head维度
    return window_mask


# ===================== 修正：DiTBlock（统一mask维度，修复expand逻辑） =====================
class DiTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_cond: int,
        n_head: int,
        dropout: float = 0.1,
        activation: str = "silu",
        window_size=None,
        rope: Optional[RoPE] = None,
    ):
        super(DiTBlock, self).__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.window_size = window_size if window_size is not None else -1
        self.rope = rope
        self.head_dim = d_model // n_head

        # Q/K/V投影 + 输出投影
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        self.adaLN = AdaLN(d_model=d_model)
        self.timestep_embedding = TimestepEmbedder(d_model=d_model, max_timesteps=1000)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.cond_embed = nn.Linear(d_cond, d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.Dropout(dropout),
            _get_activation_fn(activation),
            nn.Linear(d_model * 2, d_model),
        )

    def _multihead_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        手动实现多头注意力（适配RoPE）
        q/k/v: [seq_len, batch_size, d_model]
        attn_mask: [batch_size, n_head, seq_len, seq_len] （统一维度）
        return: [seq_len, batch_size, d_model]
        """
        seq_len, batch_size, _ = q.shape

        # 1. 拆分多头：[seq_len, batch_size, n_head, head_dim]
        q = q.reshape(seq_len, batch_size, self.n_head, self.head_dim)
        k = k.reshape(seq_len, batch_size, self.n_head, self.head_dim)
        v = v.reshape(seq_len, batch_size, self.n_head, self.head_dim)

        # 2. 对Q/K应用RoPE
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # 3. 调整维度计算注意力：[batch_size, n_head, seq_len, head_dim]
        q = q.permute(1, 2, 0, 3)  # [batch, n_head, seq_len, head_dim]
        k = k.permute(1, 2, 3, 0)  # [batch, n_head, head_dim, seq_len]
        v = v.permute(1, 2, 0, 3)  # [batch, n_head, seq_len, head_dim]

        # 4. 计算注意力权重
        attn_weights = (q @ k) / math.sqrt(
            self.head_dim
        )  # [batch, n_head, seq_len, seq_len]

        # 5. 应用注意力mask（确保mask维度匹配）
        if attn_mask is not None:
            attn_weights = attn_weights + attn_mask

        # 6. Softmax + 加权求和
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_output = attn_weights @ v  # [batch, n_head, seq_len, head_dim]

        # 7. 合并多头 + 输出投影
        attn_output = attn_output.permute(2, 0, 1, 3).reshape(
            seq_len, batch_size, self.d_model
        )
        attn_output = self.out_proj(attn_output)
        attn_output = self.dropout(attn_output)

        return attn_output

    def forward(
        self,
        src: torch.Tensor,  # [seq_len, batch_size, d_model]
        t: torch.Tensor,  # [batch_size]
        cond: torch.Tensor,  # [1, batch_size, d_model]
        padding_mask: Optional[torch.Tensor] = None,  # [batch_size, seq_len]
    ) -> torch.Tensor:

        # 1. 融合条件特征和时间步
        cond_emb = self.cond_embed(cond)
        time_emb = self.timestep_embedding(t)
        cond_fusion = cond_emb + time_emb
        gate_msa, shift_msa, scale_msa, gate_ffn, shift_ffn, scale_ffn = self.adaLN(
            cond_fusion
        )

        attn_mask = None
        seq_len = src.shape[0]
        batch_size = src.shape[1]  # 明确获取batch_size，避免维度错误

        # 2. 生成窗口mask（4维：[1, 1, seq_len, seq_len]）
        if self.window_size > 0:
            attn_mask = get_window_mask(
                seq_len=seq_len,
                window_size=self.window_size,
                device=src.device,
            )  # [1, 1, seq_len, seq_len]
            # 扩展到[batch_size, n_head, seq_len, seq_len]
            attn_mask = attn_mask.expand(batch_size, self.n_head, seq_len, seq_len)

        # 3. 处理padding mask（统一维度为[batch_size, n_head, seq_len, seq_len]）
        if padding_mask is not None:
            # 步骤1：将padding mask转为[batch_size, 1, 1, seq_len]
            padding_mask_expanded = padding_mask.unsqueeze(1).unsqueeze(
                1
            )  # [batch, 1, 1, seq_len]
            # 步骤2：扩展为[batch, n_head, seq_len, seq_len]（广播到seq_len和n_head）
            padding_mask_expanded = padding_mask_expanded.expand(
                batch_size, self.n_head, seq_len, seq_len
            )
            # 步骤3：填充位置设为-1e9
            padding_mask_expanded = padding_mask_expanded.to(dtype=src.dtype) * -1e9

            # 合并window mask和padding mask
            if attn_mask is not None:
                attn_mask = attn_mask + padding_mask_expanded
            else:
                attn_mask = padding_mask_expanded

        # 4. 自注意力层（修正后）
        x = ada_shift_scale(x=self.norm1(src), shift=shift_msa, scale=scale_msa)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x_attn = self._multihead_attention(q, k, v, attn_mask=attn_mask)
        x = x + gate_msa * x_attn

        # 5. 前馈网络
        x = ada_shift_scale(x=self.norm2(x), shift=shift_ffn, scale=scale_ffn)
        x = x + gate_ffn * self.ffn(x)

        return x


# ===================== 修改：MoreDiff（移除固定PE，接入RoPE） =====================
class MoreDiff(
    nn.Module
):  # 修正父类（原BaseModel→nn.Module，若有自定义BaseModel可改回）
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
        super(MoreDiff, self).__init__()  # 修正父类调用错误（原MoGenDiT_V2→MoreDiff）

        # 1. 移除原有固定位置编码
        # self.pe = PositionalEncoding(d_model=d_model, max_len=5000)

        # 2. 初始化RoPE模块（核心新增）
        self.head_dim = d_model // n_head
        self.d_cond = d_cond
        self.rope = RoPE(
            head_dim=self.head_dim, max_seq_len=5000  # 与原固定PE的max_len保持一致
        )

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
                    rope=self.rope,  # 将RoPE传入每个DiTBlock
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
        x = x_wrapped["x_t"]
        cond = x_wrapped["cond"]
        if cond is None:
            cond = torch.zeros(x.shape[0], 1, self.d_cond).to(x.device)
        x = x.permute(1, 0, 2)
        cond = cond.permute(1, 0, 2)
        mask = x_wrapped["mask"].permute(1, 0, 2)
        padding_mask = x_wrapped["padding_mask"]

        x = torch.cat([x, mask], dim=-1)

        x = self.motion_2_token(x)
        # 2. 移除原有固定位置编码的加法
        # x = self.pe(x)

        for encoder_block in self.encoder_blocks:
            x = encoder_block(x, t, cond, padding_mask)
        x = self.token_2_motion(x).permute(1, 0, 2)

        return x


# ===================== 原有函数/类：激活函数、时间步嵌入等（无改动） =====================
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


# ===================== 废弃：原有固定位置编码（保留但不再使用） =====================
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

def ada_shift_scale(x, shift, scale):
    return x * (scale + 1) + shift


def get_MoreDiff_model(data_dim, version="0.1B"):
    assert version in [
        "0.1B",
        "0.3B",
        "0.03B",
    ], f"不支持的MoreDiff版本: {version} 请选择['0.3B', '0.1B','0.03B']"
    if version == "0.3B":
        d_model = 1024
        n_head = 16
        n_stack = 18

    elif version == "0.1B":
        d_model = 768
        n_head = 12
        n_stack = 12

    elif version == "0.03B":
        d_model = 512
        n_head = 8
        n_stack = 8

    model = MoreDiff(
        d_motion=data_dim,
        d_model=d_model,
        d_cond=22 * 3,
        n_head=n_head,
        n_stack=n_stack,
        dropout=0.0,
        window_size=90,
    )

    print(
        f"Build MoreDiff-{version} params: {(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e9):.2f} B"
    )

    return model
