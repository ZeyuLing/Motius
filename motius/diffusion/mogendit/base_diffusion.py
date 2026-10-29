import pdb

import torch
import numpy as np
from enum import Enum
from typing import List, Callable, Optional
from tqdm import tqdm
import torch.nn.functional as F


def noise_remapping(noise, mode="identity"):
    if mode == "identity":
        return noise
    elif mode == "absorb":
        return torch.tanh(noise)
    elif mode == "clip":
        return torch.clamp(noise, -1, 1)
    elif mode == "sphere_norm":
        return F.normalize(noise, p=2, dim=-1) * np.sqrt(noise.shape[-1])
    else:
        raise ValueError(f"Unknown noise remapping mode: {mode}")


class ModelMeanType(Enum):
    """模型预测目标类型"""

    EPSILON = 1  # 预测噪声
    START_X = 2  # 预测初始值x0


class BetaSchedule(Enum):
    """噪声调度类型"""

    LINEAR = 1
    COSINE = 2


class GaussianDiffusion:
    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_schedule: BetaSchedule = BetaSchedule.COSINE,
        model_mean_type: ModelMeanType = ModelMeanType.START_X,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        noise_remap_mode: str = "identity",
    ):
        """
        初始化扩散模型

        Args:
            num_timesteps: 扩散步数
            beta_schedule: beta调度类型
            model_mean_type: 模型预测目标类型
            beta_start: linear调度的起始beta值
            beta_end: linear调度的结束beta值
        """
        self.num_timesteps = num_timesteps
        self.model_mean_type = model_mean_type
        self.noise_remap_mode = noise_remap_mode

        # 初始化beta序列
        if beta_schedule == BetaSchedule.LINEAR:
            self.betas = self._linear_beta_schedule(beta_start, beta_end, num_timesteps)
        elif beta_schedule == BetaSchedule.COSINE:
            self.betas = self._cosine_beta_schedule(num_timesteps)
        else:
            raise ValueError(f"不支持的beta调度: {beta_schedule}")

        # 计算扩散过程中的关键参数
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(
            self.alphas, axis=0
        )  # 累积乘积 αₜ = Π₁ⁿ (1-βᵢ)

        # 转换为torch张量并添加维度用于广播
        self.alphas_cumprod_t = torch.tensor(
            self.alphas_cumprod, dtype=torch.float32
        ).unsqueeze(0)
        self.sqrt_alphas_cumprod_t = torch.sqrt(self.alphas_cumprod_t)
        self.sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1 - self.alphas_cumprod_t)
        self.sqrt_recip_alphas_cumprod_t = torch.sqrt(1 / self.alphas_cumprod_t)
        self.sqrt_recipm1_alphas_cumprod_t = torch.sqrt(1 / self.alphas_cumprod_t - 1)

        # 后验分布参数
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef1 = (
            self.betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(self.alphas)
            / (1.0 - self.alphas_cumprod)
        )

        # 转换为torch张量
        self.posterior_variance_t = torch.tensor(
            self.posterior_variance, dtype=torch.float32
        ).unsqueeze(0)
        self.posterior_mean_coef1_t = torch.tensor(
            self.posterior_mean_coef1, dtype=torch.float32
        ).unsqueeze(0)
        self.posterior_mean_coef2_t = torch.tensor(
            self.posterior_mean_coef2, dtype=torch.float32
        ).unsqueeze(0)

    def _linear_beta_schedule(
        self, beta_start: float, beta_end: float, num_timesteps: int
    ) -> np.ndarray:
        """线性beta调度"""
        return np.linspace(beta_start, beta_end, num_timesteps, dtype=np.float64)

    def _cosine_beta_schedule(self, num_timesteps: int, s: float = 0.008) -> np.ndarray:
        """余弦beta调度"""

        def alpha_bar(t):
            return np.cos((t + s) / (1 + s) * np.pi / 2) ** 2

        betas = []
        for i in range(num_timesteps):
            t1 = i / num_timesteps
            t2 = (i + 1) / num_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return np.array(betas, dtype=np.float64)

    def q_sample(
        self,
        x0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        obs_mask: Optional[torch.Tensor] = None,
        length_mask: Optional[torch.Tensor] = None,
    ):
        """
        前向扩散过程：从x0采样xt

        Args:
            x0: 初始样本 (batch_size, *dim)
            t: 时间步 (batch_size,)
            noise: 噪声，若为None则自动生成

        Returns:
            xt: 扩散后的样本
        """
        if noise is None:
            noise = torch.randn_like(x0, device=x0.device)
        noise = noise_remapping(noise, mode=self.noise_remap_mode)
        # 提取对应时间步的参数
        sqrt_alphas_cumprod = self._extract(
            self.sqrt_alphas_cumprod_t.to(x0.device), t, x0.shape
        )
        sqrt_one_minus_alphas_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod_t.to(x0.device), t, x0.shape
        )
        x_noise = sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * noise

        noise_mask = torch.ones_like(noise, dtype=torch.bool)
        if obs_mask is not None:
            noise_mask[obs_mask >= (1 - 1e-5)] = False
        if length_mask is not None:
            noise_mask *= length_mask
        x_t = x0.clone()
        x_t[noise_mask] = x_noise[noise_mask]
        noise[~noise_mask] *= 0

        return x_t, noise

    def p_sample(self, model: Callable, x_wrap: dict, t: torch.Tensor) -> torch.Tensor:
        """
        反向扩散过程：从xt采样xt-1

        Args:
            model: 用于预测的模型，输入(x, t)输出预测值
            x: 当前样本 (batch_size, *dim)
            t: 时间步 (batch_size,)

        Returns:
            xt-1: 前一时间步的样本
        """
        # 获取模型预测结果和分布参数
        out = self.p_mean_variance(model, x_wrap, t)

        # 生成噪声 (t=0时不添加噪声)
        noise = torch.randn_like(x_wrap["x_t"])
        noise = noise_remapping(noise, mode=self.noise_remap_mode)
        nonzero_mask = (
            (t != 0).float().view(-1, *([1] * (len(x_wrap["x_t"].shape) - 1)))
        )

        # 采样
        x_wrap["x_t"] = (
            out["mean"] + nonzero_mask * torch.exp(0.5 * out["log_variance"]) * noise
        )
        return x_wrap

    def p_mean_variance(self, model: Callable, x_wrap: dict, t: torch.Tensor) -> dict:
        """计算反向扩散的均值和方差"""
        # B, *rest = x_wrap['x_t'].shape
        device = x_wrap["x_t"].device

        # 模型预测
        model_output = model(x_wrap, t)

        # 根据模型预测类型计算x0
        if self.model_mean_type == ModelMeanType.EPSILON:
            pred_x0 = self._predict_x0_from_eps(x_wrap["x_t"], t, model_output)
        elif self.model_mean_type == ModelMeanType.START_X:
            pred_x0 = model_output
        else:
            raise ValueError(f"不支持的模型预测类型: {self.model_mean_type}")

        # 计算后验均值
        posterior_mean = (
            self._extract(
                self.posterior_mean_coef1_t.to(device), t, x_wrap["x_t"].shape
            )
            * pred_x0
            + self._extract(
                self.posterior_mean_coef2_t.to(device), t, x_wrap["x_t"].shape
            )
            * x_wrap["x_t"]
        )

        # 计算方差
        posterior_variance = self._extract(
            self.posterior_variance_t.to(device), t, x_wrap["x_t"].shape
        )
        posterior_log_variance = torch.log(posterior_variance)

        return {
            "mean": posterior_mean,
            "variance": posterior_variance,
            "log_variance": posterior_log_variance,
            "pred_x0": pred_x0,
        }

    def ddim_sample(
        self,
        model: Callable,
        x_wrap: dict,
        t: torch.Tensor,
        eta: float = 0.0,
        prev_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        DDIM采样（去噪扩散隐式模型）

        Args:
            model: 预测模型
            x_wrap: 包含当前样本的字典
            t: 当前时间步张量
            eta: 随机性控制参数
            prev_t: 上一个时间步张量，如果为None且t>0，则使用t-1（连续时间步假设）
        """
        # 获取模型预测结果
        out = self.p_mean_variance(model, x_wrap, t)
        pred_x0 = out["pred_x0"]
        device = pred_x0.device

        # 获取当前时间步参数
        a_t = self._extract(self.alphas_cumprod_t.to(device), t, x_wrap["x_t"].shape)

        # 计算上一个时间步的alpha累积乘积
        # 注意：当使用自定义时间步序列时，prev_t由ddim_sample_loop正确传递
        if prev_t is None:
            # 如果没有提供prev_t，假设连续时间步（向后一步）
            # 这适用于默认连续采样或最后一个时间步
            if t[0] > 0:
                prev_t = t - 1
            else:
                # t=0时，上一个时间步不存在，a_prev设为1.0
                a_prev = torch.tensor(1.0, device=device)
                # 计算均值和方差（此时sigma应为0）
                sigma = eta * torch.sqrt((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev))
                mean = torch.sqrt(a_prev) * pred_x0 + torch.sqrt(
                    torch.clamp(1 - a_prev - sigma**2, min=0, max=1.0)
                ) * (x_wrap["x_t"] - torch.sqrt(a_t) * pred_x0) / torch.sqrt(1 - a_t)
                return mean

        # 如果prev_t不为None（由ddim_sample_loop正确传递），计算a_prev
        if prev_t is not None:
            a_prev = self._extract(self.alphas_cumprod_t.to(device), prev_t, x_wrap["x_t"].shape)

        # 计算均值和方差
        # 添加微小epsilon确保数值稳定性
        eps = 1e-8
        sigma_term = (1 - a_t / (a_prev + eps))
        # 确保sigma_term非负（数值稳定性）
        sigma_term = torch.clamp(sigma_term, min=0.0)
        sigma = eta * torch.sqrt(((1 - a_prev) / (1 - a_t + eps)) * sigma_term)

        # 计算mean，确保分母不为零
        denominator = torch.sqrt(1 - a_t + eps)
        mean = torch.sqrt(a_prev) * pred_x0 + torch.sqrt(
            torch.clamp(1 - a_prev - sigma**2, min=0, max=1.0)
        ) * (x_wrap["x_t"] - torch.sqrt(a_t) * pred_x0) / denominator

        # 添加噪声
        if t[0] > 0:
            noise = torch.randn_like(x_wrap["x_t"])
            noise = noise_remapping(noise, mode=self.noise_remap_mode)
            return mean + sigma * noise
        else:
            return mean

    def ddim_sample_loop(
        self,
        x_wrap: dict,
        model: Callable,
        num_timesteps: Optional[int] = None,
        eta: float = 0.0,
        device: torch.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
        mask: Optional[torch.Tensor] = None,
        early_stop: Optional[int] = None,
        custom_timesteps: Optional[list] = None,  # 新增：自定义时间步序列
        imputation_mode: str = "all",
    ) -> torch.Tensor:
        """
        DDIM完整采样循环：支持自定义时间步序列，提升生成速度
        新增参数：
            custom_timesteps: 自定义采样步数列表（如[0,5,10,20,50,100,250,500,750,999]）
                            若为None，则使用默认连续步数
        """
        assert imputation_mode in [
            "all",
            "skip_last",
            "none",
        ], "imputation_mode must be 'all', 'skip_last' or 'none'"
        # 处理时间步序列
        if custom_timesteps is not None:
            # 验证自定义时间步有效性（必须降序排列，建议但不强制包含0）
            assert (
                sorted(custom_timesteps, reverse=True) == custom_timesteps
            ), "自定义时间步必须按降序排列"
            # 放宽条件：如果使用early_stop，可以不包含0
            if early_stop is None:
                assert 0 in custom_timesteps, "当未指定early_stop时，自定义时间步必须包含0"
            times = torch.LongTensor(custom_timesteps)
        else:
            # 默认使用连续时间步（从num_timesteps-1到0）
            num_timesteps = num_timesteps or self.num_timesteps
            times = torch.LongTensor(
                list(range(num_timesteps - 1, -1, -1))
            )  # 如999,998,...,0
        # 从纯噪声开始
        x_t = x_wrap["x_t"].clone().to(device)  # 确保x_t在正确的设备上
        # 确保原始x_wrap["x_t"]也在正确的设备上
        if x_wrap["x_t"].device != device:
            x_wrap["x_t"] = x_wrap["x_t"].to(device)
        x_wrap["x_t"] = torch.randn_like(x_wrap["x_t"], device=device)

        # 逐步去噪（遍历自定义时间步）
        for i, t in enumerate(tqdm(times)):
            if mask is not None and imputation_mode in ["all", "skip_last"]:
                x_wrap["x_t"][mask] = x_t[mask]

            # 创建当前时间步的批量张量
            t_batch = torch.full(
                (x_wrap["x_t"].shape[0],), t, device=device, dtype=torch.long
            )

            # 提前终止判断
            if early_stop is not None and t == early_stop:
                break

            # 计算上一个时间步（当前步的前一个元素）
            # 注意：在自定义时间步序列中，下一个元素是prev_t，因为序列是降序的
            if i < len(times) - 1:
                prev_t = times[i + 1]  # 自定义序列中当前步的下一个元素即为上一步
            else:
                prev_t = None
            x_wrap["x_t"] = self.ddim_sample(
                model, x_wrap, t_batch, eta=eta, prev_t=prev_t
            )
        if mask is not None and imputation_mode in ["all"]:
            x_wrap["x_t"][mask] = x_t[mask]
        return x_wrap["x_t"]

    def denoise(
        self,
        x_wrap: dict,
        model: Callable,
        num_timesteps: int = 1,
        eta: float = 0.0,
        device: torch.device = (
            torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        ),
        mask: Optional[torch.Tensor] = None,
        imputation_mode: str = "all",
    ) -> torch.Tensor:
        """
        DDIM完整采样循环：从纯噪声逐步去噪生成x0

        Args:
            x_wrap: 初始输入, 其中‘x_t’将设置为纯噪声
            model: 用于预测的模型，输入(x, t)输出预测值
            num_timesteps: 采样步数，默认使用初始化时的num_timesteps
            eta: 控制随机性的参数，0为确定性采样，1为接近DDPM
            device: 采样设备

        Returns:
            x0: 最终生成的初始样本
        """
        assert imputation_mode in [
            "all",
            "skip_last",
            "none",
        ], "imputation_mode must be 'all', 'skip_last' or 'none'"
        x_t = x_wrap["x_t"].clone()

        # 设置采样步数，默认使用模型初始化时的步数
        x_wrap["x_t"], noise = self.q_sample(
            x0=x_wrap["x_t"],
            t=torch.ones(size=[x_wrap["x_t"].shape[0]], device=device, dtype=torch.bool)
            * num_timesteps,
            obs_mask=mask,
        )

        # 创建时间步序列（从T-1到0）
        times = torch.arange(0, num_timesteps, device=device)
        times = reversed(times)  # 从最后一步开始逆向采样

        # 逐步去噪
        for t in tqdm(times):
            # 创建当前时间步的批量张量
            if mask is not None and imputation_mode in ["all", "skip_last"]:
                x_wrap["x_t"][mask] = x_t[mask]
            t_batch = torch.full(
                (x_wrap["x_t"].shape[0],), t, device=device, dtype=torch.long
            )
            # 调用单步DDIM采样
            x_wrap["x_t"] = self.ddim_sample(model, x_wrap, t_batch, eta=eta)
        if mask is not None and imputation_mode in ["all"]:
            x_wrap["x_t"][mask] = x_t[mask]
        return x_wrap["x_t"]

    def x0_to_v_t(
        self, x: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """
        完全对齐论文定义：将x（真实x0或预测x0）转换为速度v_t，用于v-loss计算
        论文依据：3.1节公式(1)(2)、3.2节表1(3)(a)、4.3节数值稳定性处理
        核心公式：v = (x - z_t) / max(1 - t, clip_val)

        Args:
            x: 真实x0（x_gt）或模型预测的x0（x_pred），shape [B, C, H, W]
            z_t: t时刻的加噪样本（论文中的z_t，即原代码中的x_t），shape [B, C, H, W]
            t: 时间步张量（已归一化到[0,1]区间），shape [B]

        Returns:
            v_t: 对应的速度张量，shape [B, C, H, W]
        """
        # 1. 设备对齐：确保所有输入张量同设备
        device = x.device
        z_t = z_t.to(device)
        # 根据JiT论文 t为x_0的权重值
        t = self._extract(self.sqrt_alphas_cumprod_t.to(x.device), t, x.shape)

        # 3. 计算分母1-t，并clip防止零除法（论文4.3节指定默认clip到0.05）
        denominator = 1.0 - t
        denominator = torch.clamp(
            denominator, min=0.05
        )  # 核心修改：对齐论文数值稳定性处理

        # 4. 计算v_t（论文核心公式）
        v_t = (x - z_t) / denominator

        return v_t * 0.5

    def _predict_x0_from_eps(
        self, x: torch.Tensor, t: torch.Tensor, eps: torch.Tensor
    ) -> torch.Tensor:
        """从预测的噪声计算x0"""
        return (
            self._extract(self.sqrt_recip_alphas_cumprod_t, t, x.shape) * x
            - self._extract(self.sqrt_recipm1_alphas_cumprod_t, t, x.shape) * eps
        )

    def _extract(
        self,
        x,
        t,
        shape,
    ):
        """
        提取与时间步t对应的x值，并调整维度以匹配目标形状

        Args:
            x: 包含各时间步参数的张量 (num_timesteps,)
            t: 时间步索引张量 (batch_size,)
            shape: 目标形状，用于扩展维度

        Returns:
            扩展后的参数张量，形状为 (batch_size, 1, ...) 以匹配输入数据形状
        """
        # 确保索引张量t与x具有相同的维度数量（都为1D）
        t = t.view(-1)  # 强制将t转换为1D张量 (batch_size,)
        out = x[:, t]

        # 扩展维度以匹配目标形状（在通道和空间维度上扩展）
        reshape = [shape[0]] + [1] * (len(shape) - 1)
        return out.view(*reshape).to(x.device)


def visualize_2d_tensor(
    tensor, title="2D Tensor Visualization", cmap="coolwarm", figsize=(8, 6)
):
    from matplotlib import pyplot as plt

    """
    将2D张量可视化为图像，支持PyTorch张量和NumPy数组

    Args:
        tensor: 2D张量（PyTorch tensor或NumPy array），数值范围应为-5到5
        title: 图像标题
        cmap: 颜色映射方案，默认使用coolwarm（冷暖色，适合展示正负值）
        figsize: 图像尺寸，元组(宽, 高)
    """
    # 处理输入类型：将PyTorch张量转换为NumPy数组
    if isinstance(tensor, torch.Tensor):
        # 确保在CPU上且为二维
        tensor_np = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        tensor_np = tensor
    else:
        raise TypeError("输入必须是PyTorch张量或NumPy数组")

    # 验证维度
    if len(tensor_np.shape) != 2:
        raise ValueError(f"输入必须是2D张量，当前形状为{tensor_np.shape}")

    # 创建画布
    plt.figure(figsize=figsize)

    # 绘制热图，固定颜色范围为-5到5
    im = plt.imshow(tensor_np, cmap=cmap, vmin=-2, vmax=2, interpolation="bilinear")

    # 添加颜色条，显示数值对应关系
    cbar = plt.colorbar(im)
    cbar.set_label("Tensor Value")

    # 设置标题和坐标轴
    plt.title(title)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")

    # 显示网格（可选）
    plt.grid(False)  # 热图通常不需要网格，如需可改为True

    # 紧凑布局并显示
    plt.tight_layout()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 初始化扩散模型
    diffusion = GaussianDiffusion(
        num_timesteps=1000,
        beta_schedule=BetaSchedule.COSINE,
        model_mean_type=ModelMeanType.START_X,
    )

    # 模拟模型（实际使用时替换为真实模型）
    class SimpleModel(torch.nn.Module):
        def __call__(self, x, t):
            return x * (t[:, None, None] / 1000)  # 随机噪声作为示例

    model = SimpleModel()

    # 生成随机数据
    x0 = torch.randn(1, 224, 263) * 0  # 4个3通道32x32图像
    t = torch.randint(
        999,
        1000,
        [
            1,
        ],
    )  # 随机时间步

    # 前向扩散示例
    xt = diffusion.q_sample(x0, t)
    print("前向扩散结果形状:", xt.shape)

    # 反向扩散示例
    xt_prev = diffusion.p_sample(model, xt, t)
    print("反向扩散结果形状:", xt_prev.shape)

    # DDIM采样示例
    xt_prev_ddim = diffusion.ddim_sample(model, xt, t)
    print("DDIM采样结果形状:", xt_prev_ddim.shape)

    # 测试ddim_sample_loop函数
    print("\n测试ddim_sample_loop函数:")

    # 准备测试数据
    test_batch_size = 2
    test_seq_len = 100
    test_channels = 3
    test_x_wrap = {"x_t": torch.randn(test_batch_size, test_seq_len, test_channels)}

    # 测试1：默认连续时间步
    print("测试1: 默认连续时间步")
    try:
        result1 = diffusion.ddim_sample_loop(
            x_wrap=test_x_wrap.copy(),
            model=model,
            num_timesteps=10,  # 使用较少的步数加速测试
            eta=0.0,
            device=torch.device("cpu")
        )
        print(f"  成功: 输出形状 {result1.shape}")
    except Exception as e:
        print(f"  失败: {e}")

    # 测试2：自定义时间步序列
    print("测试2: 自定义时间步序列")
    try:
        custom_steps = [9, 7, 5, 3, 1, 0]  # 降序排列，包含0
        result2 = diffusion.ddim_sample_loop(
            x_wrap=test_x_wrap.copy(),
            model=model,
            custom_timesteps=custom_steps,
            eta=0.0,
            device=torch.device("cpu")
        )
        print(f"  成功: 输出形状 {result2.shape}")
    except Exception as e:
        print(f"  失败: {e}")

    # 测试3：带early_stop的自定义时间步
    print("测试3: 带early_stop的自定义时间步")
    try:
        custom_steps_no_zero = [9, 7, 5, 3, 1]  # 不包含0，但使用early_stop
        result3 = diffusion.ddim_sample_loop(
            x_wrap=test_x_wrap.copy(),
            model=model,
            custom_timesteps=custom_steps_no_zero,
            early_stop=1,  # 在t=1时停止
            eta=0.0,
            device=torch.device("cpu")
        )
        print(f"  成功: 输出形状 {result3.shape}")
    except Exception as e:
        print(f"  失败: {e}")

    # 测试4：带mask和imputation_mode
    print("测试4: 带mask和imputation_mode")
    try:
        # 创建mask（一半的数据被mask）
        mask = torch.zeros_like(test_x_wrap["x_t"], dtype=torch.bool)
        mask[:, :test_seq_len//2, :] = True

        result4 = diffusion.ddim_sample_loop(
            x_wrap=test_x_wrap.copy(),
            model=model,
            num_timesteps=10,
            mask=mask,
            imputation_mode="skip_last",
            eta=0.0,
            device=torch.device("cpu")
        )
        print(f"  成功: 输出形状 {result4.shape}")
    except Exception as e:
        print(f"  失败: {e}")

    print("\nddim_sample_loop函数测试完成!")
    #
    # visualize_2d_tensor(xt[0])
    # visualize_2d_tensor(xt_prev[0])
    # visualize_2d_tensor(xt_prev_ddim[0])

    # for i in range(10):
    #     t = torch.LongTensor([i*100])
    #     xt = diffusion.q_sample(x0, t)
    #     visualize_2d_tensor(xt[0])

    for i in range(1000):
        xt = diffusion.p_sample(model, xt, t=torch.LongTensor([1000 - i - 1]))
        if (1000 - i - 1) % 100 == 0:
            visualize_2d_tensor(xt[0])
