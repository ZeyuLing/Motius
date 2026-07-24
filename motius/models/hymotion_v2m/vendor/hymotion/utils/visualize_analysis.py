from __future__ import annotations
import os
from typing import Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
try:
    import seaborn as sns
except ImportError:
    sns = None
import torch
from torch import Tensor


def check_trg_curve(
    curve_value_dict: Dict[str, Tensor],
    img_pth: str,
    return_array: bool = False,
    vertical_line_dict: Optional[Dict[str, List[int]]] = None,
    waveform: Optional[Tensor] = None,
    sr: int = 16000,
    ylim_max: float = 1.3,
    ylim_min: float = -3e-1,
) -> Optional[np.ndarray]:

    sns.set_theme()
    [width, height] = matplotlib.rcParams["figure.figsize"]
    if width < 10:
        matplotlib.rcParams["figure.figsize"] = [width * 2.5, height]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title("Curve plot")
    ax.grid(True)
    for curve_name, curve_value in curve_value_dict.items():
        curve_value = curve_value.detach().cpu().numpy()
        ax.plot(curve_value, label=f"{curve_name}")
        if curve_value.min() < ylim_min:
            ylim_min = -ylim_max
    ax.set_xlabel("Frame Number")
    ax.set_ylabel("Value")
    ax.set_ylim(ylim_min, ylim_max)

    if waveform is not None:
        waveform = waveform.cpu().numpy()
        end_time = waveform.shape[1] / sr * 30
        time_axis = torch.linspace(0, end_time, waveform.shape[1])
        axis_wav = ax.twinx()
        axis_wav.set_ylabel("Audio amplitude")
        axis_wav.plot(time_axis, waveform[0], linewidth=1, color="gray", alpha=0.3)
    ax.legend(fontsize=16, ncol=2)

    if vertical_line_dict is not None:
        for curve_name, vertical_line_list in vertical_line_dict.items():
            for vertical_line in vertical_line_list:
                plt.axvline(
                    x=vertical_line,
                    linestyle="--",
                    color="r",
                    label=f"{curve_name}",
                )

    if not return_array:
        plt.savefig(f"{img_pth}")
        plt.close()
    else:
        fig.canvas.draw()
        image_data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_data = image_data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close()
        return image_data


@torch.no_grad()
def plot_attn_list(
    attn_list,
    real_motion_len,
    motion_len,
    real_text_len,
    save_dir="output/attn",
    prefix="layer",
):
    os.makedirs(save_dir, exist_ok=True)

    def crop_motion_text(mat, real_motion_len, motion_len, real_text_len):
        # mat: [S_q, S_k]
        S_q, S_k = mat.shape
        # 行：仅保留真实 motion query
        rm_q = int(min(max(real_motion_len, 0), S_q, motion_len))
        # 列：保留真实 motion key + 真实 text key
        rm_k_m = int(min(max(real_motion_len, 0), S_k, motion_len))
        rt_k_t = int(min(max(real_text_len, 0), max(S_k - motion_len, 0)))
        col_idx_left = torch.arange(0, rm_k_m)
        col_idx_right = torch.arange(motion_len, motion_len + rt_k_t)
        col_idx = torch.cat([col_idx_left, col_idx_right], dim=0)
        return mat[:rm_q, col_idx], rm_q, rm_k_m  # 同时返回分界位置

    for i, attn in enumerate(attn_list):
        # attn: [B, H, S_q, S_k]
        attn0 = attn[0]  # 取 batch 0
        attn_avg = attn0.mean(0)  # [S_q, S_k]

        for tag, mat in [("avg", attn_avg)]:
            m_full = mat.detach().float().cpu()
            m, rows_kept, motion_cols_kept = crop_motion_text(
                m_full, real_motion_len, motion_len, real_text_len
            )  # m: [rows_kept, motion_cols_kept + real_text_len]

            plt.figure(figsize=(6, 5))
            im = plt.imshow(m, cmap="viridis", interpolation="nearest", vmin=0.0, vmax=1.0)
            plt.colorbar(im, fraction=0.046, pad=0.04)

            # 分界线：裁剪后纵向分割位置在 motion_cols_kept，横向在 rows_kept
            if motion_cols_kept > 0:
                plt.axvline(motion_cols_kept - 0.5, color="w", lw=1)
            if rows_kept > 0:
                plt.axhline(rows_kept - 0.5, color="w", lw=1)

            plt.xlabel("key index (cropped: motion[0:rm], text[motion_len:motion_len+rt])")
            plt.ylabel("query index (cropped: motion[0:rm])")
            plt.title(f"{prefix}_{i:02d}_{tag}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{prefix}_{i:02d}_{tag}.png"))
            plt.close()
    return None
