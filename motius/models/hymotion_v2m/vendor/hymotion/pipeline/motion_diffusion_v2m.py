from __future__ import annotations
# 这个脚本用来进行动作生成；给定提取的特征
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch import Tensor
from torchdiffeq import odeint
import matplotlib.pyplot as plt

from ..bodymodels.smpl_skeleton import SMPLMesh, SMPLMeshSparseKeypoints, SMPLSkeleton
from ..datasets.geometry import angle_axis_to_rotation_matrix, rot6d_to_rotation_matrix, rotation_matrix_to_rot6d, rotation_matrix_to_angle_axis
from ..evaluation.metrics import (
    as_np_array,
    compute_global_metrics,
    get_joints_from_smpl_params,
    get_vertices_from_smpl_params,
)
from ..utils.loaders import load_object, read_yaml
from ..utils.postprocess import pp_static_joint, pp_static_joint_footonly_v2, process_ik, process_ik_mc, save_fk_end_effector_debug
from ..utils.postprocess_optimization import post_optimization
from ..utils.type_converter import get_module_device
from ..utils.net_utils import gaussian_smooth
from .motion_diffusion import randn_tensor
from ..utils.postprocess import end_vel_to_static_conf


# ==================== ODE Solvers ====================
# 以下实现了多种常用的ODE求解器，用于Flow Matching的采样过程


def euler_step(fn, t, y, dt):
    """
    Euler法（一阶）
    y_{n+1} = y_n + dt * f(t_n, y_n)
    """
    return y + dt * fn(t, y)


def midpoint_step(fn, t, y, dt):
    """
    中点法（二阶Runge-Kutta）
    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/2, y_n + dt/2 * k1)
    y_{n+1} = y_n + dt * k2
    """
    k1 = fn(t, y)
    k2 = fn(t + dt / 2, y + dt / 2 * k1)
    return y + dt * k2


def heun_step(fn, t, y, dt):
    """
    Heun法/改进欧拉法（二阶）
    k1 = f(t_n, y_n)
    k2 = f(t_n + dt, y_n + dt * k1)
    y_{n+1} = y_n + dt/2 * (k1 + k2)
    """
    k1 = fn(t, y)
    k2 = fn(t + dt, y + dt * k1)
    return y + dt / 2 * (k1 + k2)


def rk4_step(fn, t, y, dt):
    """
    经典四阶Runge-Kutta法
    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/2, y_n + dt/2 * k1)
    k3 = f(t_n + dt/2, y_n + dt/2 * k2)
    k4 = f(t_n + dt, y_n + dt * k3)
    y_{n+1} = y_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    k1 = fn(t, y)
    k2 = fn(t + dt / 2, y + dt / 2 * k1)
    k3 = fn(t + dt / 2, y + dt / 2 * k2)
    k4 = fn(t + dt, y + dt * k3)
    return y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def rk38_step(fn, t, y, dt):
    """
    3/8规则四阶Runge-Kutta法
    与经典RK4精度相同，但系数不同
    k1 = f(t_n, y_n)
    k2 = f(t_n + dt/3, y_n + dt/3 * k1)
    k3 = f(t_n + 2*dt/3, y_n - dt/3 * k1 + dt * k2)
    k4 = f(t_n + dt, y_n + dt * k1 - dt * k2 + dt * k3)
    y_{n+1} = y_n + dt/8 * (k1 + 3*k2 + 3*k3 + k4)
    """
    k1 = fn(t, y)
    k2 = fn(t + dt / 3, y + dt / 3 * k1)
    k3 = fn(t + 2 * dt / 3, y - dt / 3 * k1 + dt * k2)
    k4 = fn(t + dt, y + dt * k1 - dt * k2 + dt * k3)
    return y + dt / 8 * (k1 + 3 * k2 + 3 * k3 + k4)


def ralston_step(fn, t, y, dt):
    """
    Ralston法（二阶，最小化截断误差）
    k1 = f(t_n, y_n)
    k2 = f(t_n + 2*dt/3, y_n + 2*dt/3 * k1)
    y_{n+1} = y_n + dt * (1/4 * k1 + 3/4 * k2)
    """
    k1 = fn(t, y)
    k2 = fn(t + 2 * dt / 3, y + 2 * dt / 3 * k1)
    return y + dt * (k1 / 4 + 3 * k2 / 4)


def ssprk3_step(fn, t, y, dt):
    """
    强稳定保持三阶Runge-Kutta法 (SSPRK3 / TVD-RK3)
    常用于需要保持稳定性的问题
    y1 = y_n + dt * f(t_n, y_n)
    y2 = 3/4 * y_n + 1/4 * (y1 + dt * f(t_n + dt, y1))
    y_{n+1} = 1/3 * y_n + 2/3 * (y2 + dt * f(t_n + dt/2, y2))
    """
    y1 = y + dt * fn(t, y)
    y2 = 0.75 * y + 0.25 * (y1 + dt * fn(t + dt, y1))
    return y / 3 + 2 / 3 * (y2 + dt * fn(t + dt / 2, y2))


# 自定义求解器字典（带 _custom 后缀，用于区分 torchdiffeq 的实现）
CUSTOM_ODE_SOLVERS = {
    "heun_custom": heun_step,
    "rk4_custom": rk4_step,
    "rk38": rk38_step,  # torchdiffeq 不支持
    "ralston": ralston_step,  # torchdiffeq 不支持
    "ssprk3": ssprk3_step,  # torchdiffeq 不支持
}

# torchdiffeq 支持的方法列表
TORCHDIFFEQ_METHODS = [
    "euler",  # 一阶欧拉法
    "midpoint",  # 中点法
    "rk4",  # 经典四阶RK
    "explicit_adams",  # Adams-Bashforth
    "implicit_adams",  # Adams-Moulton
    "dopri5",  # Dormand-Prince 5(4) 自适应
    "dopri8",  # Dormand-Prince 8(7) 自适应
    "bosh3",  # Bogacki-Shampine 自适应
    "adaptive_heun",  # 自适应Heun
    "scipy_solver",  # scipy后端
]


def odeint_custom(fn, y0, t, method="euler", **kwargs):
    """
    ODE积分器，支持torchdiffeq方法和自定义方法

    Args:
        fn: 导数函数 dy/dt = fn(t, y)
        y0: 初始状态 (Tensor)
        t: 时间点序列 (Tensor)，从t[0]积分到t[-1]
        method: 求解器方法名称，支持:
            torchdiffeq 方法（推荐）:
            - "euler": 欧拉法（一阶）
            - "midpoint": 中点法（二阶）
            - "rk4": 经典四阶Runge-Kutta
            - "explicit_adams": Adams-Bashforth方法
            - "implicit_adams": Adams-Moulton方法
            - "dopri5": Dormand-Prince 5(4) 自适应步长
            - "dopri8": Dormand-Prince 8(7) 自适应步长
            - "bosh3": Bogacki-Shampine 自适应
            - "adaptive_heun": 自适应Heun

            自定义方法:
            - "euler_custom": 自定义欧拉法
            - "midpoint_custom": 自定义中点法
            - "heun_custom": Heun法/改进欧拉法
            - "rk4_custom": 自定义四阶RK
            - "rk38": 3/8规则四阶RK
            - "ralston": Ralston法（二阶）
            - "ssprk3": 强稳定保持三阶RK
        **kwargs: 传递给torchdiffeq.odeint的额外参数（仅对torchdiffeq方法有效）

    Returns:
        trajectory: 形状为 (len(t), *y0.shape) 的轨迹张量
    """
    # 对于torchdiffeq支持的方法，直接调用torchdiffeq
    if method in TORCHDIFFEQ_METHODS:
        return odeint(fn, y0, t, method=method, **kwargs)

    # 对于自定义的固定步长方法
    if method not in CUSTOM_ODE_SOLVERS:
        raise ValueError(
            f"Unknown ODE solver method: {method}. "
            f"Available torchdiffeq methods: {TORCHDIFFEQ_METHODS}\n"
            f"Available custom methods: {list(CUSTOM_ODE_SOLVERS.keys())}"
        )

    step_fn = CUSTOM_ODE_SOLVERS[method]

    # 初始化轨迹
    trajectory = [y0]
    y = y0

    # 逐步积分
    for i in range(len(t) - 1):
        dt = t[i + 1] - t[i]
        y = step_fn(fn, t[i], y, dt)
        trajectory.append(y)

    return torch.stack(trajectory, dim=0)


def length_to_mask(lengths: Tensor, max_len: int) -> Tensor:
    """
        lengths: (B, 1)
        max_len: int
    Returns: (B, max_len)
    """
    assert lengths.max() <= max_len, f"lengths.max()={lengths.max()} > max_len={max_len}"
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths
    return mask


def rollout_local_transl_vel(local_transl_vel, global_orient_R, camera_vel=0, fps=30):
    """
    transl velocity is in local coordinate (or, SMPL-coord)
    Args:
        local_transl_vel: (*, L, 3)
        global_orient_R: (*, L, 3, 3)
    Returns:
        transl: (*, L, 3)
    """
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient_R, local_transl_vel) / fps
    # rollout from start point
    transl_vel = transl_vel + camera_vel
    transl = torch.cumsum(transl_vel, dim=-2)
    return transl


class BaseV2M(torch.nn.Module):
    def load_mean_std(self, mean_std):
        with open(mean_std, "r") as f:
            mean_std = json.load(f)
        if (
            self.motion_rep == "wvrot6dstd"
            or self.motion_rep == "wvrot6d_transl_std"
            or self.motion_rep == "wvrot6d_transl_shape_std"
            or self.motion_rep == "wvrot6d_transl_shape_stationary_std"
        ):
            for key in ["root_rot6d", "body_rot6d", "transl_vel", "shapes"]:
                mean = torch.FloatTensor(mean_std[key]["mean"])
                std = torch.FloatTensor(mean_std[key]["std"])
                self.register_buffer(f"{key}_mean", mean[None, None])
                self.register_buffer(f"{key}_std", std[None, None])
            if "end_effector_vel" in mean_std:
                mean = torch.FloatTensor(mean_std["end_effector_vel"]["mean"])
                std = torch.FloatTensor(mean_std["end_effector_vel"]["std"])
                self.register_buffer(f"end_effector_vel_mean", mean[None, None])
                self.register_buffer(f"end_effector_vel_std", std[None, None])
        else:
            # dict_keys(['rot6d', 'trans', 'trans_vel', 'trans_vel_norm', 'local_trans', 'local_trans_norm', 'local_transl_vel', 'local_transl_vel_norm', 'shapes'])
            self.register_buffer("rot6d_mean", torch.FloatTensor(mean_std["rot6d"]["mean"])[None, None])
            rot6d_std = torch.FloatTensor(mean_std["rot6d"]["std"])
            print(f"[{self.__class__.__name__}] set rot6d_std to {rot6d_std.max()}")
            rot6d_std.fill_(rot6d_std.max())
            self.register_buffer("rot6d_std", rot6d_std[None, None])
            trans_vel_mean = torch.zeros_like(torch.FloatTensor(mean_std["trans_vel"]["mean"]))
            # 速度直接设置为4
            trans_vel_std = torch.ones_like(torch.FloatTensor(mean_std["trans_vel"]["std"]))
            self.register_buffer("trans_vel_mean", trans_vel_mean[None, None])
            self.register_buffer("trans_vel_std", trans_vel_std[None, None])
            # trans
            self.register_buffer("trans_mean", torch.FloatTensor(mean_std["trans"]["mean"])[None, None])
            trans_std = torch.FloatTensor(mean_std["trans"]["std"])
            self.register_buffer("trans_std", trans_std[None, None])
            # shapes
            self.register_buffer("shapes_mean", torch.FloatTensor(mean_std["shapes"]["mean"])[None])
            shapes_std = torch.FloatTensor(mean_std["shapes"]["std"])
            self.register_buffer("shapes_std", shapes_std[None])

    def set_epoch(self, epoch):
        self.epoch = epoch

    def _encode_gv_rot6d(self, batch):
        # 这个会把数据编码到GV空间
        rot6d = batch["rot6d"]
        rot6d_body = rot6d[:, :, 1:]
        rot6d_root = rot6d[:, :, 0]
        # root的在世界坐标系下的旋转
        rotation_root = rot6d_to_rotation_matrix(rot6d_root)
        # 计算在camera下的旋转
        # rot6d = rot6d - self.rot6d_mean
        # rot6d = rot6d.reshape(rot6d.shape[0], rot6d.shape[1], -1)
        rot6d_body = rot6d_body - self.rot6d_mean[:, :, 1:]
        rot6d_body = rot6d_body.reshape(rot6d_body.shape[0], rot6d_body.shape[1], -1)

        R_c2gv = batch["R_c2gv"]
        # R_c2gv @ camera_R @ rotation_root
        global_orient_R = batch["camera_R"] @ rotation_root
        global_orient_gv_r6d = rotation_matrix_to_rot6d(R_c2gv @ global_orient_R)
        #
        gv_r6d_mean = torch.FloatTensor([0, 0, 0, -1, -1, 0]).to(global_orient_gv_r6d.device).reshape(1, 1, 6)
        global_orient_gv_r6d = global_orient_gv_r6d - gv_r6d_mean
        return torch.cat([global_orient_gv_r6d, rot6d_body], dim=-1)

    def _encode_wv_rot6d(self, batch):
        # 这个直接把输入取出来，不进行任何操作，dataset里已经操作过了
        root_rot6d = batch["root_rot6d"] - self.rot6d_mean[:, :, 0]
        body_rot6d = batch["body_rot6d"] - self.rot6d_mean[:, :, 1:]
        body_rot6d = body_rot6d.reshape(body_rot6d.shape[0], body_rot6d.shape[1], -1)
        trans_vel = batch["transl_vel"]
        shapes = batch["shapes"]
        return torch.cat([root_rot6d, body_rot6d], dim=-1)

    def _encode_wv_rot6d_std(self, batch):
        root_rot6d = (batch["root_rot6d"] - self.root_rot6d_mean) / self.root_rot6d_std
        body_rot6d_std = self.body_rot6d_std.clone()
        body_rot6d_std[body_rot6d_std < 1e-3] = 1.0
        body_rot6d = (batch["body_rot6d"] - self.body_rot6d_mean) / body_rot6d_std
        body_rot6d = body_rot6d.reshape(body_rot6d.shape[0], body_rot6d.shape[1], -1)
        return torch.cat([root_rot6d, body_rot6d], dim=-1)

    def _encode_wv_rot6d_transl_std(self, batch):
        wv_rot6d_std = self._encode_wv_rot6d_std(batch)
        transl_vel = (batch["transl_vel"] - self.transl_vel_mean) / self.transl_vel_std
        return torch.cat([wv_rot6d_std, transl_vel], dim=-1)

    def _encode_wv_rot6d_transl_shape_std(self, batch):
        wv_rot6d_transl_std = self._encode_wv_rot6d_transl_std(batch)
        body_shape = (batch["shapes"] - self.shapes_mean) / self.shapes_std
        return torch.cat([wv_rot6d_transl_std, body_shape], dim=-1)

    def _encode_wv_rot6d_transl_shape_stationary_std(self, batch):
        wv_rot6d_transl_shape_std = self._encode_wv_rot6d_transl_shape_std(batch)
        end_effector_vel = (batch["end_effector_vel"] - self.end_effector_vel_mean) / self.end_effector_vel_std
        end_effector_vel = end_effector_vel.reshape(end_effector_vel.shape[0], end_effector_vel.shape[1], -1)
        return torch.cat([wv_rot6d_transl_shape_std, end_effector_vel], dim=-1)

    def _decode_gv_rot6d(self, latent, **other_params):
        gv_r6d_mean = torch.FloatTensor([0, 0, 0, -1, -1, 0]).to(latent.device).reshape(1, 1, 6)
        global_orient_gv_r6d = latent[:, :, :6] + gv_r6d_mean
        rot6d_body = latent[:, :, 6:]
        rot6d_body = rot6d_body.reshape(rot6d_body.shape[0], rot6d_body.shape[1], -1, 6)
        rot6d_body = rot6d_body + self.rot6d_mean[:, :, 1:]
        # 处理root
        R_gv = rot6d_to_rotation_matrix(global_orient_gv_r6d)
        # R_c2gv @ camera_R @ rotation_root => R_gv
        # rotation_root = (R_c2gv @ camera_R).mT @ R_gv
        rotation_root = (other_params["R_c2gv"] @ other_params["camera_R"]).transpose(-1, -2) @ R_gv
        rot6d_root = rotation_matrix_to_rot6d(rotation_root)
        rot6d = torch.cat([rot6d_root[:, :, None], rot6d_body], dim=-2)
        shapes = torch.zeros(latent.shape[0], latent.shape[1], 16, device=latent.device, dtype=latent.dtype)
        trans = torch.zeros(latent.shape[0], latent.shape[1], 3, device=latent.device, dtype=latent.dtype)
        return {"rot6d": rot6d, "shapes": shapes, "trans": trans}

    def _decode_wv_rot6d(self, latent, **other_params):
        root_rot6d = latent[:, :, :6] + self.rot6d_mean[:, :, 0]
        rot6d_body = latent[:, :, 6:]
        rot6d_body = rot6d_body.reshape(rot6d_body.shape[0], rot6d_body.shape[1], -1, 6)
        rot6d_body = rot6d_body + self.rot6d_mean[:, :, 1:]
        shapes = torch.zeros(latent.shape[0], latent.shape[1], 16, device=latent.device, dtype=latent.dtype)
        trans = torch.zeros(latent.shape[0], latent.shape[1], 3, device=latent.device, dtype=latent.dtype)
        rot6d = torch.cat([root_rot6d[:, :, None], rot6d_body], dim=-2)
        return {"rot6d": rot6d, "shapes": shapes, "trans": trans}

    def _decode_wv_rot6d_std(self, latent, **other_params):
        root_rot6d = latent[:, :, :6] * self.root_rot6d_std + self.root_rot6d_mean
        body_rot6d = latent[:, :, 6:]
        body_rot6d = body_rot6d.reshape(body_rot6d.shape[0], body_rot6d.shape[1], -1, 6)
        body_rot6d = body_rot6d * self.body_rot6d_std + self.body_rot6d_mean
        rot6d = torch.cat([root_rot6d[:, :, None], body_rot6d], dim=-2)
        shapes = torch.zeros(latent.shape[0], latent.shape[1], 16, device=latent.device, dtype=latent.dtype)
        trans = torch.zeros(latent.shape[0], latent.shape[1], 3, device=latent.device, dtype=latent.dtype)
        return {"rot6d": rot6d, "shapes": shapes, "trans": trans}

    def _decode_wv_rot6d_transl_std(self, latent, fps=30, **other_params):
        root_rot6d = latent[:, :, :6] * self.root_rot6d_std + self.root_rot6d_mean
        body_rot6d = latent[:, :, 6:-3]
        body_rot6d = body_rot6d.reshape(body_rot6d.shape[0], body_rot6d.shape[1], -1, 6)
        body_rot6d = body_rot6d * self.body_rot6d_std + self.body_rot6d_mean
        transl_vel = latent[:, :, -3:] * self.transl_vel_std + self.transl_vel_mean

        root_rotmat = rot6d_to_rotation_matrix(root_rot6d)

        trans = rollout_local_transl_vel(transl_vel, root_rotmat, fps=fps)
        rot6d = torch.cat([root_rot6d[:, :, None], body_rot6d], dim=-2)
        shapes = torch.zeros(latent.shape[0], latent.shape[1], 16, device=latent.device, dtype=latent.dtype)

        return {"rot6d": rot6d, "shapes": shapes, "trans": trans}

    def _decode_wv_rot6d_transl_shape_std(self, latent, fps=30, **other_params):
        root_rot6d = latent[:, :, :6] * self.root_rot6d_std + self.root_rot6d_mean
        body_rot6d = latent[:, :, 6:-19]
        body_rot6d = body_rot6d.reshape(body_rot6d.shape[0], body_rot6d.shape[1], -1, 6)
        body_rot6d = body_rot6d * self.body_rot6d_std + self.body_rot6d_mean
        transl_vel = latent[:, :, -19:-16] * self.transl_vel_std + self.transl_vel_mean

        root_rotmat = rot6d_to_rotation_matrix(root_rot6d)
        trans = rollout_local_transl_vel(transl_vel, root_rotmat, fps=fps)

        rot6d = torch.cat([root_rot6d[:, :, None], body_rot6d], dim=-2)
        shapes = latent[:, :, -16:] * self.shapes_std + self.shapes_mean
        # print('trans', trans.shape, trans[0, :20])
        # if self.training:
        #     breakpoint()

        return {
            "rot6d": rot6d,
            "shapes": shapes,
            "trans": trans,
            "global_orient": root_rotmat,
            "local_transl_vel": transl_vel,
        }

    def _optimizer_func(self, optimizer, closure, metrics_report_func, max_iters=100):
        # 运行多次优化迭代
        for iter_idx in range(max_iters):

            # 每10次迭代打印一次metrics
            if iter_idx == 0 or (iter_idx + 1) % 10 == 0:
                with torch.no_grad():
                    metrics = metrics_report_func()
                log_str = f"[Iter {iter_idx:3d}] "
                for key, value in metrics.items():
                    log_str += f"{key}: {value:.4f}, "
                print(log_str)
            optimizer.step(closure)


    @torch.enable_grad()
    def _optimize_feet_floating(self, rot6d, shapes, transl_vel, static_conf, fps=30, height_axis_dim=1):
        # Frame, J, 6
        root_rotmat = rot6d_to_rotation_matrix(rot6d[:, 0, :6])
        param_init = {
            "rot6d": rot6d,
            "shapes": shapes,
            "trans": torch.zeros_like(transl_vel)
        }
        vertices = self.mesh_model(param_init)["vertices"]
        height_per_frame = vertices[:, :, height_axis_dim].min(dim=-1)[0]
        transl_vel_global_init = torch.einsum("lij,lj->li", root_rotmat, transl_vel) / fps
        trans_init = torch.cumsum(transl_vel_global_init, dim=0)  # (L, 3)
        height_per_frame_wtrans_init = height_per_frame + trans_init[:, height_axis_dim]


        trans_vel_opt = transl_vel.clone().detach().requires_grad_(True)
        opt_params = [trans_vel_opt]
        optimizer = torch.optim.LBFGS(opt_params, lr=1.0, max_iter=20, line_search_fn='strong_wolfe')
        static_conf_feet_atleast = static_conf[:, :4].any(dim=-1)

        def closure():
            optimizer.zero_grad()
            transl_vel_global = torch.einsum("lij,lj->li", root_rotmat, trans_vel_opt) / fps
            trans = torch.cumsum(transl_vel_global, dim=0)  # (L, 3)
            # 增加上trans
            height_per_frame_wtrans = height_per_frame + trans[:, height_axis_dim]
            # 要求
            # 优化trans_vel到最低点
            loss_height = static_conf_feet_atleast.float() * (height_per_frame_wtrans - height_per_frame_wtrans.detach().mean()).abs()
            # print('loss_height', loss_height)
            # 静止帧最低点的均值作为地面参考，所有帧的最低点不应低于此值（防止陷入地面以下）
            static_mask = static_conf_feet_atleast.float()
            static_height_mean = (height_per_frame_wtrans.detach() * static_mask).sum() / static_mask.sum().clamp(min=1)
            loss_floor = torch.clamp(static_height_mean - height_per_frame_wtrans, min=0).mean()
            trans_global_vel = (trans[1:] - trans[:-1]) * fps
            trans_global_acc = (trans_global_vel[1:] - trans_global_vel[:-1]) * fps
            trans_global_jerk = (trans_global_acc[1:] - trans_global_acc[:-1])
            loss_jerk = torch.pow(trans_global_jerk, 2).mean()
            loss_init = torch.pow(trans_vel_opt - transl_vel_global_init, 2).mean()
            loss = loss_height.mean() + loss_floor #+ loss_jerk * 1e-6 + loss_init * 1e-4
            loss.backward()
            return loss

        def metrics_report_func():
            transl_vel_global = torch.einsum("lij,lj->li", root_rotmat, trans_vel_opt) / fps
            trans = torch.cumsum(transl_vel_global, dim=0)  # (L, 3)
            height_per_frame_wtrans = height_per_frame + trans[:, height_axis_dim]
            height_per_frame_wtrans_mean = height_per_frame_wtrans.mean().item()
            stationary_frames = static_conf_feet_atleast.sum().item()
            stationary_frames_ratio = stationary_frames / static_conf_feet_atleast.shape[0]
            stationary_frame_height_mean = ((height_per_frame_wtrans - height_per_frame_wtrans_mean) * static_conf_feet_atleast.float()).abs().mean().item()
            stationary_frame_height_max = ((height_per_frame_wtrans - height_per_frame_wtrans_mean) * static_conf_feet_atleast.float()).abs().max().item()
            # 陷地指标：静止帧最低点均值作为地面参考，统计低于地面的帧
            static_mask = static_conf_feet_atleast.float()
            static_height_mean = (height_per_frame_wtrans * static_mask).sum().item() / max(static_mask.sum().item(), 1)
            below_floor = (static_height_mean - height_per_frame_wtrans).clamp(min=0)
            floor_penetration_mean = below_floor.mean().item()
            floor_penetration_max = below_floor.max().item()
            floor_penetration_ratio = (below_floor > 0).float().mean().item()

            return {
                "stationary_frames": stationary_frames,
                "stationary_frames_ratio": stationary_frames_ratio,
                "stationary_frame_height_mean": stationary_frame_height_mean * 1000,
                "stationary_frame_height_max": stationary_frame_height_max * 1000,
                "floor_penetration_mean": floor_penetration_mean * 1000,
                "floor_penetration_max": floor_penetration_max * 1000,
                "floor_penetration_ratio": floor_penetration_ratio,
            }

        self._optimizer_func(optimizer, closure, metrics_report_func, max_iters=100)

        # 可视化优化前后height对比
        with torch.no_grad():
            trans_opt = torch.cumsum(torch.einsum("lij,lj->li", root_rotmat, trans_vel_opt) / fps, dim=0)
            h_before = (height_per_frame + trans_init[:, height_axis_dim]).cpu().numpy()
            h_after = (height_per_frame + trans_opt[:, height_axis_dim]).cpu().numpy()
            os.makedirs("debug/optimize_feet_floating", exist_ok=True)
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(h_before, label='Before', alpha=0.8)
            ax.plot(h_after, label='After', alpha=0.8)
            ax.set(xlabel='Frame', ylabel='Height (m)', title='Feet Floating: Before vs After')
            ax.legend(); ax.grid(True, alpha=0.3)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"debug/optimize_feet_floating/{timestamp}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight'); plt.close(fig)
            print(f"[DEBUG] Saved: {save_path}")

        return rot6d, trans_vel_opt.detach()

    def _save_keypoints3d_comparison(self, params, name="grounding", labels=None,
                                        debug_dir="debug/optimize_feet_floating",
                                        joint_ids=None, joint_names=None):
        """可视化优化前后end-effector关节位置对比，以及root节点rot6d变化

        Args:
            params: list of dict, 每个dict包含 "rot6d" (L, J, 6), "shapes" (L, 16), "trans" (L, 3)
            name: 图片名称前缀
            labels: list of str, 每个param对应的标签，默认 ["Before", "After", ...]
            debug_dir: 调试图片保存目录
            joint_ids: 需要可视化的关节ID列表
            joint_names: 需要可视化的关节名称列表
        """
        if joint_ids is None:
            joint_ids = [7, 10, 8, 11]
            joint_names = ["L_Ankle", "L_Foot", "R_Ankle", "R_Foot"]
        if joint_names is None:
            joint_names = [f"Joint_{jid}" for jid in joint_ids]
        if labels is None:
            labels = [f"Param_{k}" for k in range(len(params))]

        os.makedirs(debug_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # --- FK得到每组param的keypoints3d ---
        kp_list = []  # list of (L, J, 3) numpy
        rot6d_list = []  # list of (L, J, 6) numpy
        for p in params:
            fk_out = self.body_model(p)
            kp_list.append(fk_out["keypoints3d"].detach().cpu().numpy())  # (L, J, 3)
            rot6d_list.append(p["rot6d"].detach().cpu().numpy())  # (L, J, 6)

        # === 图1: end-effector 关节位置对比 ===
        axis_names = ['X', 'Y', 'Z']
        num_joints = len(joint_ids)
        num_params = len(params)

        fig, axes = plt.subplots(3, num_joints, figsize=(num_joints * 3, 9), squeeze=False)
        fig.suptitle(f'{name}: Keypoints3D Comparison', fontsize=13)

        colors = plt.cm.tab10.colors
        for col, (jid, jname) in enumerate(zip(joint_ids, joint_names)):
            for ax_i in range(3):
                ax = axes[ax_i, col]
                for k in range(num_params):
                    ax.plot(kp_list[k][:, jid, ax_i], linewidth=0.8, alpha=0.7,
                            color=colors[k % len(colors)], label=labels[k])
                if ax_i == 0:
                    ax.set_title(f'{jname} (j{jid})', fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'{axis_names[ax_i]}', fontsize=10)
                if ax_i == 2:
                    ax.set_xlabel('Frame', fontsize=9)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=6, loc='upper right')

        # 统一每行ylim
        for ax_i in range(3):
            y_min = min(axes[ax_i, c].get_ylim()[0] for c in range(num_joints))
            y_max = max(axes[ax_i, c].get_ylim()[1] for c in range(num_joints))
            for c in range(num_joints):
                axes[ax_i, c].set_ylim(y_min, y_max)

        save_path = os.path.join(debug_dir, f"{timestamp}_{name}_kp3d.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[DEBUG] Saved keypoints3d: {save_path}")

        # === 图2: root节点 rot6d 变化对比 ===
        rot6d_dim = 6
        fig2, axes2 = plt.subplots(2, 3, figsize=(12, 6), squeeze=False)
        fig2.suptitle(f'{name}: Root Rot6D Comparison', fontsize=13)

        dim_labels = [f'd{d}' for d in range(rot6d_dim)]
        for d in range(rot6d_dim):
            row, col = d // 3, d % 3
            ax = axes2[row, col]
            for k in range(num_params):
                ax.plot(rot6d_list[k][:, 0, d], linewidth=0.8, alpha=0.7,
                        color=colors[k % len(colors)], label=labels[k])
            ax.set_title(f'Root rot6d [{dim_labels[d]}]', fontsize=10)
            ax.set_ylim(-1.2, 1.2)
            ax.set_xlabel('Frame', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=6, loc='upper right')

        save_path2 = os.path.join(debug_dir, f"{timestamp}_{name}_root_rot6d.png")
        plt.tight_layout()
        plt.savefig(save_path2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[DEBUG] Saved root rot6d: {save_path2}")

    def _optimize_feet_grounding(
        self,
        rot6d,
        shapes,
        transl_vel,
        static_conf
    ):
        """
        优化脚部接地：尽量不改变transl_vel的情况下，让静止帧的最低点都在第一帧的最低点上。

        拆分后处理Node：
        1. Ground Node：抑制数据的floating情况

        Args:
            rot6d: (B, L, J, 6) 旋转表示
            shapes: (B, L, 16) 体型参数
            transl_vel: (B, L, 3) 局部坐标系下的平移速度
            root_rotmat: (B, L, 3, 3) 根节点旋转矩阵
            static_conf: (B, L, 6) 末端效应器的静止置信度
            fps: 帧率
            reg_weight: 控制保持原始transl_vel的权重
            n_outer_iters: 优化迭代次数
            verbose: 是否打印优化过程信息
            optimize_pose: 是否同时优化pose (rot6d)
            pose_reg_weight: 控制保持原始rot6d的权重

        Returns:
            dict: {
                "transl_vel": (B, L, 3) 优化后的局部坐标系下的平移速度,
                "rot6d": (B, L, J, 6) 优化后的旋转表示 (如果optimize_pose=True则为优化后的值，否则为原值)
            }
        """
        transl_vel = transl_vel.clone()
        rot6d = rot6d.clone()
        optimize_nodes = [
            {
                'name': "GroundNode",
                'optimize_params': ['trans_vel'],
                'func': self._optimize_feet_floating,
                'enabled': True
            }
        ]
        root_rotmat = rot6d_to_rotation_matrix(rot6d[:, 0, :6])
        print(f'[{self.__class__.__name__}] _optimize_feet_grounding: {rot6d.shape[0]} batches')
        for i in range(rot6d.shape[0]):
            rot6d_init = rot6d[i].clone().detach()  # (L, J, 6)
            root_rotmat_i = rot6d_to_rotation_matrix(rot6d_init[:, 0, :6])
            transl_vel_init = transl_vel[i].clone().detach()  # (L, 3)
            rot6d_i = rot6d_init.clone().detach()
            transl_vel_i = transl_vel_init.clone().detach()
            shapes_i = shapes[i]  # (L, 16)
            static_conf_i = static_conf[i]  # (L, 6)
            trans_i = rollout_local_transl_vel(transl_vel_i, root_rotmat_i, fps=30)
            param_init = {
                "rot6d": rot6d_i.clone().detach(),
                "shapes": shapes_i.clone().detach(),
                "trans": trans_i.clone().detach()
            }
            if False:
                # === DEBUG: 优化前，用mesh_model获取vertices，每隔100帧保存OBJ ===
                with torch.no_grad():
                    debug_mesh_out = self.mesh_model({
                        "rot6d": rot6d_i,        # (L, J, 6)
                        "shapes": shapes_i,      # (L, 16)
                        "trans": trans_i,         # (L, 3)
                    })
                import trimesh
                debug_vertices = debug_mesh_out["vertices"].detach().cpu().numpy()  # (L, V, 3)
                debug_faces = self.mesh_model.faces  # (F, 3)
                debug_obj_dir = f"debug/optimize_feet_floating/mesh_before_opt_b{i}"
                os.makedirs(debug_obj_dir, exist_ok=True)
                num_frames = debug_vertices.shape[0]
                for fi in range(0, num_frames, 100):
                    mesh = trimesh.Trimesh(vertices=debug_vertices[fi], faces=debug_faces, process=False)
                    obj_path = os.path.join(debug_obj_dir, f"frame_{fi:06d}.obj")
                    mesh.export(obj_path)
                    print(f"[DEBUG] Saved mesh OBJ: {obj_path}")
                print(f"[DEBUG] Saved {len(range(0, num_frames, 100))} OBJ files to {debug_obj_dir}")

            for node in optimize_nodes:
                if not node['enabled']:
                    continue
                rot6d_i, transl_vel_i = node['func'](rot6d_i, shapes_i, transl_vel_i, static_conf_i)

            trans_i_new = rollout_local_transl_vel(transl_vel_i, rot6d_to_rotation_matrix(rot6d_i[:, 0, :6]), fps=30)

            param_after = {
                "rot6d": rot6d_i.clone().detach(),
                "shapes": shapes_i.clone().detach(),
                "trans": trans_i_new.clone().detach()
            }

            # self._save_keypoints3d_comparison([param_init, param_after], name=f"grounding_b{i}", labels=["Before", "After"])
            transl_vel[i] = transl_vel_i.detach()
            rot6d[i] = rot6d_i.detach()
        return {"transl_vel": transl_vel, "rot6d": rot6d}

        feet_joint_ids = [7, 10, 8, 11]  # left_ankle, left_toe, right_ankle, right_toe
        ankle_indices = [0, 2]  # left_ankle, right_ankle
        toe_indices = [1, 3]    # left_toe, right_toe

        for i in range(rot6d.shape[0]):
            # 初始化参数
            static_conf_i = static_conf[i][:, :4]  # (L, 4) 只取脚的置信度
            transl_vel_init = transl_vel[i].clone().detach()  # (L, 3)
            rot6d_init = rot6d[i].clone().detach()  # (L, J, 6)
            root_rotmat_i = root_rotmat[i]  # (L, 3, 3)
            shapes_i = shapes[i]  # (L, 16)

            # 计算初始脚部位置用于确定地面高度
            param_init = {
                "rot6d": rot6d_init,
                "shapes": shapes_i,
                "trans": torch.zeros(rot6d.shape[1], 3, device=rot6d.device, dtype=rot6d.dtype)
            }
            smpl24_wotrans_init = self.body_model(param_init)["keypoints3d"]
            feet_local_init = smpl24_wotrans_init[:, feet_joint_ids, :]  # (L, 4, 3)

            # 计算第一帧的地面高度（脚踝和脚尖分别统计）
            transl_vel_global_init = torch.einsum("lij,lj->li", root_rotmat_i, transl_vel_init) / fps
            trans_init = torch.cumsum(transl_vel_global_init, dim=0)  # (L, 3)
            feet_global_init = feet_local_init + trans_init[:, None, :]  # (L, 4, 3)

            ground_height_ankle = feet_global_init[0, ankle_indices, 1].min()  # 第一帧脚踝的最低y值
            ground_height_toe = feet_global_init[0, toe_indices, 1].min()      # 第一帧脚尖的最低y值
            ankle_conf = static_conf_i[:, ankle_indices]  # (L, 2)
            toe_conf = static_conf_i[:, toe_indices]      # (L, 2)

            # 创建可优化的参数
            transl_vel_opt = transl_vel_init.clone().detach().requires_grad_(True)

            if optimize_pose:
                rot6d_opt = rot6d_init.clone().detach().requires_grad_(True)
                opt_params = [transl_vel_opt, rot6d_opt]
            else:
                rot6d_opt = rot6d_init  # 不优化pose时直接使用原值
                opt_params = [transl_vel_opt]

            with torch.enable_grad():
                optimizer = torch.optim.LBFGS(opt_params, lr=1.0, max_iter=20, line_search_fn='strong_wolfe')

                def closure():
                    optimizer.zero_grad()

                    # 如果优化pose，需要重新计算feet_local
                    if optimize_pose:
                        param_opt = {
                            "rot6d": rot6d_opt,
                            "shapes": shapes_i,
                            "trans": torch.zeros(rot6d.shape[1], 3, device=rot6d.device, dtype=rot6d.dtype)
                        }
                        smpl24_wotrans = self.body_model(param_opt)["keypoints3d"]
                        feet_local = smpl24_wotrans[:, feet_joint_ids, :]  # (L, 4, 3)
                    else:
                        feet_local = feet_local_init

                    # 从transl_vel_opt计算trans
                    transl_vel_global = torch.einsum("lij,lj->li", root_rotmat_i, transl_vel_opt) / fps
                    trans_opt = torch.cumsum(transl_vel_global, dim=0)  # (L, 3)

                    # 计算全局脚部位置
                    feet_global = feet_local + trans_opt[:, None, :]  # (L, 4, 3)

                    # 获取脚的y坐标 (y是垂直方向)
                    feet_y = feet_global[:, :, 1]  # (L, 4)

                    # 分别计算脚踝和脚尖的ground loss
                    ankle_y = feet_y[:, ankle_indices]  # (L, 2)
                    toe_y = feet_y[:, toe_indices]      # (L, 2)

                    # Ground loss: 静止的脚踝和脚尖分别在各自的地面高度上
                    ankle_loss = (ankle_conf * (ankle_y - ground_height_ankle).abs()).sum() / (ankle_conf.sum() + 1e-6)
                    toe_loss = (toe_conf * (toe_y - ground_height_toe).abs()).sum() / (toe_conf.sum() + 1e-6)
                    static_ground_loss = ankle_loss + toe_loss

                    # Penetration loss: 非静止的脚不能穿透地面（高度不能小于地面高度）
                    non_static_ankle_conf = 1.0 - ankle_conf  # (L, 2)
                    non_static_toe_conf = 1.0 - toe_conf      # (L, 2)
                    ankle_penetration = torch.relu(ground_height_ankle - ankle_y)  # (L, 2)
                    toe_penetration = torch.relu(ground_height_toe - toe_y)        # (L, 2)
                    penetration_loss = (non_static_ankle_conf * ankle_penetration).sum() / (non_static_ankle_conf.sum() + 1e-6) + \
                                      (non_static_toe_conf * toe_penetration).sum() / (non_static_toe_conf.sum() + 1e-6)

                    ground_loss = static_ground_loss + penetration_loss

                    # Regularization loss: 保持和原始transl_vel接近
                    transl_reg_loss = ((transl_vel_opt - transl_vel_init) ** 2).mean()

                    # Pose regularization loss: 保持和原始rot6d接近
                    if optimize_pose:
                        pose_reg_loss = ((rot6d_opt - rot6d_init) ** 2).mean()
                        total_loss = ground_loss + reg_weight * transl_reg_loss + pose_reg_weight * pose_reg_loss
                    else:
                        total_loss = ground_loss + reg_weight * transl_reg_loss

                    total_loss.backward()
                    return total_loss

                # 运行多次优化迭代
                for iter_idx in range(n_outer_iters):
                    optimizer.step(closure)

                    # 每10次迭代打印一次metrics
                    if verbose and (iter_idx % 10 == 0 or iter_idx == n_outer_iters - 1):
                        with torch.no_grad():
                            # 如果优化pose，需要重新计算feet_local
                            if optimize_pose:
                                param_opt = {
                                    "rot6d": rot6d_opt,
                                    "shapes": shapes_i,
                                    "trans": torch.zeros(rot6d.shape[1], 3, device=rot6d.device, dtype=rot6d.dtype)
                                }
                                smpl24_wotrans = self.body_model(param_opt)["keypoints3d"]
                                feet_local = smpl24_wotrans[:, feet_joint_ids, :]
                            else:
                                feet_local = feet_local_init

                            transl_vel_global = torch.einsum("lij,lj->li", root_rotmat_i, transl_vel_opt) / fps
                            trans_opt = torch.cumsum(transl_vel_global, dim=0)
                            feet_global = feet_local + trans_opt[:, None, :]
                            feet_y = feet_global[:, :, 1]

                            ankle_y = feet_y[:, ankle_indices]
                            toe_y = feet_y[:, toe_indices]

                            # 静止脚踝/脚尖到地面的平均距离
                            static_ankle_dist = (ankle_conf * (ankle_y - ground_height_ankle).abs()).sum() / (ankle_conf.sum() + 1e-6)
                            static_toe_dist = (toe_conf * (toe_y - ground_height_toe).abs()).sum() / (toe_conf.sum() + 1e-6)

                            # 穿透点数量
                            ankle_penetration_mask = ankle_y < ground_height_ankle
                            toe_penetration_mask = toe_y < ground_height_toe
                            n_ankle_penetration = ankle_penetration_mask.sum().item()
                            n_toe_penetration = toe_penetration_mask.sum().item()

                            print(f"[Iter {iter_idx:3d}] static_ankle_dist: {static_ankle_dist.item():.4f}, static_toe_dist: {static_toe_dist.item():.4f}, "
                                  f"penetration_count: ankle={n_ankle_penetration}, toe={n_toe_penetration}")

            # 更新transl_vel
            transl_vel[i] = transl_vel_opt.detach()

            # 更新rot6d (如果优化了pose)
            if optimize_pose:
                rot6d[i] = rot6d_opt.detach()

        return {"transl_vel": transl_vel, "rot6d": rot6d}

    def _decode_wv_rot6d_transl_shape_stationary_std(self, latent, fps=30, is_gt=False, **other_params):
        root_rot6d = latent[:, :, :6] * self.root_rot6d_std + self.root_rot6d_mean
        body_rot6d = latent[:, :, 6:-37]
        body_rot6d = body_rot6d.reshape(body_rot6d.shape[0], body_rot6d.shape[1], -1, 6)
        body_rot6d = body_rot6d * self.body_rot6d_std + self.body_rot6d_mean
        transl_vel = latent[:, :, -37:-34] * self.transl_vel_std + self.transl_vel_mean

        root_rotmat = rot6d_to_rotation_matrix(root_rot6d)
        # trans = rollout_local_transl_vel(transl_vel, root_rotmat, fps=fps)
        # if "camera_T" in other_params:
        #     camera_vel = other_params["camera_T"] / fps
        #     trans = rollout_local_transl_vel(transl_vel, root_rotmat, camera_vel, fps=fps)
        # else:

        rot6d = torch.cat([root_rot6d[:, :, None], body_rot6d], dim=-2)
        shapes = latent[:, :, -34:-18] * self.shapes_std + self.shapes_mean
        # ATTN: shapes必须取均值
        shapes = shapes.mean(dim=-2, keepdim=True)
        end_effector_vel = latent[:, :, -18:]
        end_effector_vel = end_effector_vel.reshape(end_effector_vel.shape[0], end_effector_vel.shape[1], 6, 3)
        end_effector_vel = end_effector_vel * self.end_effector_vel_std + self.end_effector_vel_mean
        static_conf = end_vel_to_static_conf(end_effector_vel)
        # 在这里判断要不要优化脚
        # 尽量不改变transl_vel的情况下，让静止帧的最低点都在第一帧的最低点上
        optimize_feet_grounding = False
        if optimize_feet_grounding:
            opt_result = self._optimize_feet_grounding(
                rot6d=rot6d,
                shapes=shapes,
                transl_vel=transl_vel,
                static_conf=static_conf
            )
            transl_vel = opt_result["transl_vel"]
            rot6d = opt_result["rot6d"]
        trans = rollout_local_transl_vel(transl_vel, root_rotmat, fps=fps)

        # 打印一下metric
        shapes = shapes.mean(dim=-2, keepdim=True)
        for i in range(transl_vel.shape[0]):
            param_i = {
                "rot6d": rot6d[i],
                "shapes": shapes[i],
                "trans": trans[i],
            }
            vertices = self.mesh_model(param_i)["vertices"]
            if not is_gt and False:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # === DEBUG: 优化前，用mesh_model获取vertices，每隔100帧保存OBJ ===
                with torch.no_grad():
                    debug_mesh_out = self.mesh_model({
                        "rot6d": rot6d[i],        # (L, J, 6)
                        "shapes": shapes[i],      # (L, 16)
                        "trans": trans[i],         # (L, 3)
                    })
                print(f'[{self.__class__.__name__}] shapes[i].shape: {shapes[i].shape}')
                print(f'[{self.__class__.__name__}] trans[i].shape: {trans[i].shape}')
                import trimesh
                debug_vertices = debug_mesh_out["vertices"].detach().cpu().numpy()  # (L, V, 3)
                debug_faces = self.mesh_model.faces  # (F, 3)
                debug_obj_dir = f"debug/optimize_feet_floating/mesh_after_opt_b{i}_{timestamp}"
                os.makedirs(debug_obj_dir, exist_ok=True)
                num_frames = debug_vertices.shape[0]
                for fi in range(0, num_frames, 100):
                    mesh = trimesh.Trimesh(vertices=debug_vertices[fi], faces=debug_faces, process=False)
                    obj_path = os.path.join(debug_obj_dir, f"frame_{fi:06d}.obj")
                    mesh.export(obj_path)
                    print(f"[DEBUG] Saved mesh OBJ: {obj_path}")
                print(f"[DEBUG] Saved {len(range(0, num_frames, 100))} OBJ files to {debug_obj_dir}")
                #
                from ..utils.smplh2fbx import SMPLH2FBX
                smplh2fbx = SMPLH2FBX()
                poses = rotation_matrix_to_angle_axis(rot6d_to_rotation_matrix(rot6d[i]))
                param_for_fbx = {
                    "poses": poses.cpu().numpy(),
                    "betas": shapes[i, 0:1].detach().cpu().numpy(),
                    "trans": trans[i].detach().cpu().numpy(),
                    "mocap_framerate": 30,
                    "num_frames": rot6d[i].shape[0],
                }
                smplh2fbx.convert_params_to_fbx(param_for_fbx, f"debug/optimize_feet_floating/mesh_after_opt_b{i}_{timestamp}/fbx_after_opt_b{i}.fbx")
                breakpoint()

            height_axis_dim = 1
            height_per_frame = vertices[:, :, height_axis_dim].min(dim=-1)[0]
            height_per_frame_mean = height_per_frame.mean().item()
            height_per_frame_max = height_per_frame.max().item()
            height_per_frame_min = height_per_frame.min().item()
            static_foot_atleast = static_conf[i, :, :4].any(dim=-1)
            height_mean_static = height_per_frame[static_foot_atleast].mean().item()
            height_max_static = height_per_frame[static_foot_atleast].max().item()
            height_min_static = height_per_frame[static_foot_atleast].min().item()
            print(f"batch {i} height_per_frame_mean: {height_per_frame_mean:.4f}, height_per_frame_max: {height_per_frame_max:.4f}, height_per_frame_min: {height_per_frame_min:.4f}, height_mean_static: {height_mean_static:.4f}, height_max_static: {height_max_static:.4f}, height_min_static: {height_min_static:.4f}")

        return {
            "rot6d": rot6d,
            "shapes": shapes,
            "trans": trans,
            "global_orient": root_rotmat,
            "local_transl_vel": transl_vel,
            "end_effector_vel": end_effector_vel,
        }

    def encode_motion(self, batch):
        if self.motion_rep == "rot6d":
            rot6d = batch["rot6d"]
            rot6d = rot6d - self.rot6d_mean
            rot6d = rot6d.reshape(rot6d.shape[0], rot6d.shape[1], -1)
            return rot6d
        elif self.motion_rep == "gvrot6d":
            # 这个实现GV坐标系下的rot6d的编码
            return self._encode_gv_rot6d(batch)
        elif self.motion_rep == "wvrot6d":
            return self._encode_wv_rot6d(batch)
        elif self.motion_rep == "wvrot6dstd":
            return self._encode_wv_rot6d_std(batch)
        elif self.motion_rep == "wvrot6d_transl_std":
            return self._encode_wv_rot6d_transl_std(batch)
        elif self.motion_rep == "wvrot6d_transl_shape_std":
            return self._encode_wv_rot6d_transl_shape_std(batch)
        elif self.motion_rep == "wvrot6d_transl_shape_stationary_std":
            return self._encode_wv_rot6d_transl_shape_stationary_std(batch)
        else:
            raise ValueError(f"Unsupported motion representation: {self.motion_rep}")

    def decode_motion(self, latent, **other_params) -> Dict[str, Tensor]:
        if self.motion_rep == "rot6d":
            rot6d = latent.reshape(latent.shape[0], latent.shape[1], -1, 6)
            rot6d = rot6d + self.rot6d_mean
            shapes = torch.zeros(latent.shape[0], latent.shape[1], 16, device=latent.device, dtype=latent.dtype)
            trans = torch.zeros(latent.shape[0], latent.shape[1], 3, device=latent.device, dtype=latent.dtype)
            return {"rot6d": rot6d, "shapes": shapes, "trans": trans}
        elif self.motion_rep == "gvrot6d":
            return self._decode_gv_rot6d(latent, **other_params)
        elif self.motion_rep == "wvrot6d":
            return self._decode_wv_rot6d(latent, **other_params)
        elif self.motion_rep == "wvrot6dstd":
            return self._decode_wv_rot6d_std(latent, **other_params)
        elif self.motion_rep == "wvrot6d_transl_std":
            return self._decode_wv_rot6d_transl_std(latent, **other_params)
        elif self.motion_rep == "wvrot6d_transl_shape_std":
            return self._decode_wv_rot6d_transl_shape_std(latent, **other_params)
        elif self.motion_rep == "wvrot6d_transl_shape_stationary_std":
            return self._decode_wv_rot6d_transl_shape_stationary_std(latent, **other_params)
        else:
            raise ValueError(f"Unsupported motion representation: {self.motion_rep}")

    def calculate_keypoints(self, batch):
        rot6d = batch["rot6d"]
        transl = batch["trans"]
        shapes = batch["shapes"]
        rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
        transl_flat = torch.zeros(transl.shape[0] * transl.shape[1], 3, device=transl.device, dtype=transl.dtype)
        shapes_flat = shapes.reshape(shapes.shape[0] * shapes.shape[1], -1)

        params = {
            "rot6d": rot6d_flat,
            "trans": transl_flat,
            "shapes": shapes_flat,
        }

        out_keypoints = self.body_model(params)
        out_keypoints = out_keypoints["keypoints3d"]
        out_keypoints = out_keypoints.reshape(rot6d.shape[0], rot6d.shape[1], -1, 3)
        return out_keypoints

    def calculate_smpl_mesh_vertices(self, batch):
        rot6d = batch["rot6d"]
        transl = batch["trans"]
        shapes = batch["shapes"]
        rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
        transl_flat = torch.zeros(transl.shape[0] * transl.shape[1], 3, device=transl.device, dtype=transl.dtype)
        shapes_flat = shapes.reshape(shapes.shape[0] * shapes.shape[1], -1)

        params = {
            "rot6d": rot6d_flat,
            "trans": transl_flat,
            "shapes": shapes_flat,
        }

        out_vertices = self.mesh_model(params)
        out_vertices = out_vertices["vertices_wotrans"]
        out_vertices = out_vertices.reshape(rot6d.shape[0], rot6d.shape[1], -1, 3)
        return out_vertices

    def calculate_joint_jitter(self, batch_keypoints, fps=30):
        jitter = (
            (
                (
                    batch_keypoints[:, 3:]
                    - 3 * batch_keypoints[:, 2:-1]
                    + 3 * batch_keypoints[:, 1:-2]
                    - batch_keypoints[:, :-3]
                )
                * (fps * 1**3)
            )
            .norm(dim=-1)
            .mean(dim=-1)
        )
        zero_padding = torch.zeros(jitter.shape[0], 3).to(jitter.device)
        jitter = torch.cat([jitter, zero_padding], dim=1)
        return jitter


class FeedForwardV2M(BaseV2M):
    """
    纯前馈版本的 Video-to-Motion 模型（非生成式）。
    使用可学习的 embedding 替代 noise 和 timestep，实现单步推理。
    """

    def __init__(
        self,
        network_module: str,
        network_module_args: dict,
        losses_cfg: dict,
        motion_rep: str = "wvrot6d_transl_shape_stationary_std",
        mean_std: str = "assets/sft_mean_std.json",
    ):
        super().__init__()
        self.motion_transformer = load_object(network_module, network_module_args)
        self.body_model = SMPLSkeleton()
        self.motion_rep = motion_rep
        self.losses_cfg = losses_cfg
        self.load_mean_std(mean_std)

        # 可学习的 timestep（标量，初始化为 1.0）
        self.learnable_timestep = torch.nn.Parameter(torch.ones(1))

        # 可学习的输入 embedding（维度与模型输出维度一致）
        output_dim = self.motion_transformer.output_dim
        self.learnable_x_emb = torch.nn.Parameter(torch.randn(output_dim) * 0.02)

    def forward_in_training(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """单步前向传播，计算预测和 GT 之间的 loss"""
        gt_motion = self.encode_motion(batch["target"])
        device = gt_motion.device
        B, L, D = gt_motion.shape
        length = batch["length"]
        max_length = length.max().item()

        # 裁剪到最大长度
        gt_motion = gt_motion[:, :max_length]
        ctxt_input = batch["inputs"]["feature"]
        if isinstance(ctxt_input, dict):
            for key in ctxt_input:
                ctxt_input[key] = ctxt_input[key][:, :max_length]
        else:
            ctxt_input = ctxt_input[:, :max_length]

        # 使用可学习的 x embedding 广播到 (B, L, D)
        x = self.learnable_x_emb[None, None, :].expand(B, max_length, -1).to(device)

        # 使用可学习的 timestep 广播到 (B,)
        timesteps = self.learnable_timestep.expand(B).to(device)

        # vtxt_input 设为零（与 MotionGenerationV2M 一致）
        vtxt_input = torch.zeros((B, 1, self.motion_transformer.vtxt_input_dim), device=device)

        # 创建时间 mask
        x_mask_temporal = length_to_mask(length, max_length)

        # 单步前向传播
        pred = self.motion_transformer(
            x=x,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=x_mask_temporal,
        )

        # 计算 loss（直接比较预测和 GT）
        loss_fns = {"SmoothL1Loss": torch.nn.functional.smooth_l1_loss, "MSELoss": torch.nn.functional.mse_loss}
        loss_dict = {}

        recon_type = self.losses_cfg["recons"]["name"]
        recon_loss = loss_fns[recon_type](pred, gt_motion, reduction="none").mean(dim=-1)
        recon_loss = (recon_loss * x_mask_temporal).sum() / x_mask_temporal.sum()
        loss_dict["recons"] = recon_loss

        loss_weight = {key: cfg["weight"] for key, cfg in self.losses_cfg.items()}
        loss = sum([loss_dict[k] * loss_weight[k] for k in loss_dict.keys()])

        return {
            "latent": gt_motion,
            "model_output": pred,
            "loss": loss,
            "loss_dict": loss_dict,
            "tensor_results": {
                "index": batch.get("index", None),
            },
        }

    @torch.no_grad()
    def validate(self, batch: Dict[str, Any], seeds: List[int] = [0], do_postproc: bool = False, cfg=1):
        """单步推理并计算评估指标"""
        length = batch["length"]
        camera_is_static = batch.get(
            "camera_is_static", torch.ones(length.shape[0], dtype=torch.bool, device=length.device)
        )
        gt_motion = self.encode_motion(batch["target"])
        device = gt_motion.device
        B, L, D = gt_motion.shape

        # 获取输入特征
        feature = batch["inputs"]["feature"]

        # 使用可学习的 x embedding 广播到 (B, L, D)
        x = self.learnable_x_emb[None, None, :].expand(B, L, -1).to(device)

        # 使用可学习的 timestep 广播到 (B,)
        timesteps = self.learnable_timestep.expand(B).to(device)

        # vtxt_input 设为零
        vtxt_input = torch.zeros((B, 1, self.motion_transformer.vtxt_input_dim), device=device)

        # 创建时间 mask
        x_mask_temporal = length_to_mask(length, L)

        # 单步前向传播
        sampled = self.motion_transformer(
            x=x,
            ctxt_input=feature,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=x_mask_temporal,
        )

        # 解码 motion
        if self.motion_rep == "wvrot6d_transl_shape_stationary_std":
            output = self.decode_motion(sampled, camera_T=feature["camera_T"])
            gt_decode = self.decode_motion(gt_motion)
        elif self.motion_rep in ["wvrot6d", "wvrot6dstd", "wvrot6d_transl_std", "wvrot6d_transl_shape_std"]:
            output = self.decode_motion(sampled)
            gt_decode = self.decode_motion(gt_motion)
        else:
            raise NotImplementedError(f"Unsupported motion representation: {self.motion_rep}")

        # forward kinematics
        pred_j3d = get_joints_from_smpl_params(self.body_model, output, joint_num=52)
        pred_j3d_local = pred_j3d["local_joints"]
        gt_j3d = get_joints_from_smpl_params(self.body_model, gt_decode, joint_num=52)
        gt_j3d_local = gt_j3d["local_joints"]

        # 计算评估指标
        dist_local_body = torch.norm(pred_j3d_local[:, :, :22, :] - gt_j3d_local[:, :, :22, :], dim=-1).mean(dim=-1)
        dist_local_hand = torch.norm(pred_j3d_local[:, :, 22:, :] - gt_j3d_local[:, :, 22:, :], dim=-1).mean(dim=-1)
        dist_local_body = (dist_local_body * x_mask_temporal).sum() / x_mask_temporal.sum()
        dist_local_hand = (dist_local_hand * x_mask_temporal).sum() / x_mask_temporal.sum()

        angular_error_root = torch.norm(output["rot6d"][:, :, 0, :] - gt_decode["rot6d"][:, :, 0, :], dim=-1)
        angular_error_body = torch.norm(
            output["rot6d"][:, :, 1:22, :] - gt_decode["rot6d"][:, :, 1:22, :], dim=-1
        ).mean(dim=-1)
        angular_error_hand = torch.norm(output["rot6d"][:, :, 22:, :] - gt_decode["rot6d"][:, :, 22:, :], dim=-1).mean(
            dim=-1
        )
        angular_error_root = (angular_error_root * x_mask_temporal).sum() / x_mask_temporal.sum()
        angular_error_body = (angular_error_body * x_mask_temporal).sum() / x_mask_temporal.sum()
        angular_error_hand = (angular_error_hand * x_mask_temporal).sum() / x_mask_temporal.sum()

        jitter_local_body = self.calculate_joint_jitter(pred_j3d_local[:, :, :22, :])
        jitter_local_hand = self.calculate_joint_jitter(pred_j3d_local[:, :, 22:, :])
        jitter_local_body = (jitter_local_body * x_mask_temporal).sum() / x_mask_temporal.sum()
        jitter_local_hand = (jitter_local_hand * x_mask_temporal).sum() / x_mask_temporal.sum()

        metrics = {
            "dist_local_body": dist_local_body,
            "dist_local_hand": dist_local_hand,
            "angular_error_root": angular_error_root,
            "angular_error_body": angular_error_body,
            "angular_error_hand": angular_error_hand,
            "jitter_local_body": jitter_local_body,
            "jitter_local_hand": jitter_local_hand,
        }

        if "_transl_" in self.motion_rep:
            dist_abs_transl = torch.norm(output["trans"] - gt_decode["trans"], dim=-1)
            dist_abs_transl = (dist_abs_transl * x_mask_temporal).sum() / x_mask_temporal.sum()
            jitter_abs_transl = self.calculate_joint_jitter(output["trans"].unsqueeze(2))
            jitter_abs_transl = (jitter_abs_transl * x_mask_temporal).sum() / x_mask_temporal.sum()
            metrics.update({"dist_abs_transl": dist_abs_transl, "jitter_abs_transl": jitter_abs_transl})

        if "_shape_" in self.motion_rep:
            shape_error = torch.norm(output["shapes"] - gt_decode["shapes"][:, 0:1], dim=-1)
            shape_error = (shape_error * x_mask_temporal).sum() / x_mask_temporal.sum()
            metrics.update({"body_shape_error": shape_error})

        return {
            "metrics": metrics,
            "output": output,
            "gt_decode": gt_decode,
            "length": length,
            "camera_is_static": camera_is_static,
        }


class CleanLoss(torch.nn.Module):
    def __init__(self, name="SmoothL1Loss", default_weight=1.0):
        super().__init__()
        if name == "SmoothL1Loss":
            self.loss_func = torch.nn.functional.smooth_l1_loss
        elif name == "MSELoss":
            self.loss_func = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unsupported loss function: {name}")

        self.default_weight = default_weight

    def forward(self, pred, gt):
        loss = self.loss_func(pred, gt, reduction="none").mean(dim=-1)
        return loss


class FKLoss(CleanLoss):
    def __init__(self, body_model: SMPLSkeleton, name="SmoothL1Loss", default_weight=1.0):
        super().__init__(name=name, default_weight=default_weight)
        self.body_model = body_model

    def forward_kinematics(self, pred, gt):
        pred_joints = get_joints_from_smpl_params(self.body_model, pred, joint_num=52)
        gt_joints = get_joints_from_smpl_params(self.body_model, gt, joint_num=52)
        return {
            "pred_joint_local": pred_joints["local_joints"],
            "pred_joint_global": pred_joints["global_joints"],
            "gt_joint_local": gt_joints["local_joints"],
            "gt_joint_global": gt_joints["global_joints"],
        }

    def forward(self, pred, gt, global_step):
        weight = self.default_weight

        fk_results = self.forward_kinematics(pred, gt)

        loss = (
            self.loss_func(fk_results["pred_joint_local"], fk_results["gt_joint_local"], reduction="none")
            .sum(dim=-1)
            .mean(dim=-1)
        )
        loss = loss * weight
        return loss


class VertexLoss(torch.nn.Module):
    def __init__(
        self,
        body_model: SMPLMesh,
        start_step=10000,
        overlap_step=10000,
        name="SmoothL1Loss",
        weight=1.0,
        num_sample_points=-1,
    ):
        super().__init__()
        self.body_model = body_model
        self.start_step = start_step
        self.overlap_step = overlap_step
        self.num_sample_points = num_sample_points
        if name == "SmoothL1Loss":
            self.loss_func = torch.nn.functional.smooth_l1_loss
        elif name == "MSELoss":
            self.loss_func = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unsupported loss function: {name}")

    def calculate_smpl_mesh_vertices(self, batch, sample_indices=None):
        rot6d = batch["rot6d"]
        transl = batch["trans"]
        shapes = batch["shapes"]
        rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
        transl_flat = torch.zeros(transl.shape[0] * transl.shape[1], 3, device=transl.device, dtype=transl.dtype)
        shapes_flat = shapes.reshape(shapes.shape[0] * shapes.shape[1], -1)

        params = {
            "rot6d": rot6d_flat,
            "trans": transl_flat,
            "shapes": shapes_flat,
        }

        out_vertices = self.body_model(params, sample_indices=sample_indices)
        out_vertices = out_vertices["vertices_wotrans"]
        out_vertices = out_vertices.reshape(rot6d.shape[0], rot6d.shape[1], -1, 3)
        return out_vertices

    def forward(self, pred, gt, global_step):
        if global_step < self.start_step:
            trans = pred["trans"]
            return torch.zeros(trans.shape[0], trans.shape[1], device=trans.device, dtype=trans.dtype)
        if global_step > self.start_step + self.overlap_step:
            weight = 1.0
        else:
            weight = (global_step - self.start_step) / self.overlap_step

        if self.num_sample_points > 0:
            # 每次随机从6890里采样出N个点来，只计算N个点的
            # 这个是所有batch、frame共享的，在当前的 B x F里就只使用这些点
            # ATTN: magic number 6890 is the number of vertices in the SMPL mesh
            sample_indices = torch.randint(0, self.body_model.v_template.shape[0], (self.num_sample_points,))
        else:
            sample_indices = None
        pred_vertices = self.calculate_smpl_mesh_vertices(pred, sample_indices)
        gt_vertices = self.calculate_smpl_mesh_vertices(gt, sample_indices)

        loss = self.loss_func(pred_vertices, gt_vertices, reduction="none").sum(dim=-1).mean(dim=-1)
        loss = loss * weight
        return loss


class TransRollLoss(torch.nn.Module):
    def __init__(self, start_step=10000, overlap_step=10000, name="SmoothL1Loss", weight=1.0):
        super().__init__()
        self.start_step = start_step
        self.overlap_step = overlap_step
        if name == "SmoothL1Loss":
            self.loss_func = torch.nn.functional.smooth_l1_loss
        elif name == "MSELoss":
            self.loss_func = torch.nn.functional.mse_loss
        else:
            raise ValueError(f"Unsupported loss function: {name}")

    def forward(self, pred, gt, global_step):
        if global_step < self.start_step:
            trans = pred["trans"]
            return torch.zeros(trans.shape[0], trans.shape[1], device=trans.device, dtype=trans.dtype)
        if global_step > self.start_step + self.overlap_step:
            weight = 1.0
        else:
            weight = (global_step - self.start_step) / self.overlap_step

        gt_transl_w = gt["trans"]
        gt_global_orient_w = gt["global_orient"]
        local_transl_vel = pred["local_transl_vel"]

        pred_transl_w = rollout_local_transl_vel(local_transl_vel, gt_global_orient_w, fps=30)

        trans_w_loss = self.loss_func(pred_transl_w, gt_transl_w, reduction="none").mean(dim=-1)
        return trans_w_loss * weight


class Joint2DLoss(FKLoss):
    def __init__(
        self, body_model: SMPLMesh, start_step=10000, overlap_step=10000, name="SmoothL1Loss", default_weight=1.0
    ):
        super().__init__(body_model, name=name, default_weight=default_weight)
        self.start_step = start_step
        self.overlap_step = overlap_step

    def forward(self, pred, gt, global_step):
        if global_step < self.start_step:
            trans = pred["trans"]
            return torch.zeros(trans.shape[0], trans.shape[1], device=trans.device, dtype=trans.dtype)
        if global_step > self.start_step + self.overlap_step:
            weight = 1.0
        else:
            weight = (global_step - self.start_step) / self.overlap_step

        fk_results = self.forward_kinematics(pred, gt)
        # TODO: pred_j2d = normalize(fk_results["pred_joint_global"])  what is T(wv -> camera)
        # TODO: gt_j2d = normalize(fk_results["gt_joint_global"])
        pred_j2d = torch.zeros(1)
        gt_j2d = torch.zeros(1)

        loss = self.loss_func(pred_j2d, gt_j2d, reduction="none").sum(dim=-1).mean(dim=-1)
        loss = loss * weight
        return loss


def map_scale_logits(scale):
    return torch.exp(scale)


class MotionGenerationV2M(BaseV2M):
    def __init__(
        self,
        network_module: str,
        network_module_args: dict,
        losses_cfg: dict,
        noise_scheduler_cfg: dict,
        infer_noise_scheduler_cfg: dict,
        train_cfg: dict,
        test_cfg: dict,
        train_frames: int,
        motion_rep="rot6d",
        mean_std="assets/sft_mean_std.json",
        pred_type="velocity",
        body_model_path="checkpoints/body_models/smplh/neutral/model.npz",
        **kwargs,
    ):
        super().__init__()
        self.motion_transformer = load_object(network_module, network_module_args)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.train_frames = train_frames
        self.losses_cfg = losses_cfg
        self._noise_scheduler_cfg = noise_scheduler_cfg
        self._infer_noise_scheduler_cfg = infer_noise_scheduler_cfg
        self.body_model = SMPLSkeleton(model_path=body_model_path)
        if pred_type == "x1" or pred_type == "x1raw":
            if "vertex" in losses_cfg:
                self.mesh_model = SMPLMesh(model_path=body_model_path)
                self.vertex_loss = VertexLoss(self.mesh_model, **losses_cfg["vertex"])
            if "transroll" in losses_cfg:
                self.transroll_loss = TransRollLoss(**losses_cfg["transroll"])
            if "joint2d" in losses_cfg:
                self.joint2d_loss = Joint2DLoss(**losses_cfg["joint2d"])
            if "clean" in losses_cfg:
                self.clean_loss = CleanLoss(**losses_cfg["clean"])
        self.motion_rep = motion_rep
        self.pred_type = pred_type
        self._parse_train_cfg()
        self._parse_test_cfg()
        self.load_mean_std(mean_std)
        self.global_iteration = -1

    def build_body_model_sparse(self):
        if hasattr(self, "body_model_sparse"):
            return
        self.body_model_sparse = SMPLMeshSparseKeypoints()
        self.body_model_sparse.to(self.body_model.j_template.device)

    def _parse_train_cfg(self) -> None:
        self.cond_mask_prob = self.train_cfg.get("cond_mask_prob", 0.0)

    def _parse_test_cfg(self) -> None:
        self.validation_steps = self._infer_noise_scheduler_cfg["validation_steps"]
        self.text_guidance_scale = self.test_cfg.get("text_guidance_scale", 1)
        self.evaluate_on = self.test_cfg.get("evaluate_on", None)

    @staticmethod
    def noise_from_seeds(latent: Tensor, seeds: Union[int, List[int]], seed_start: int = 0) -> Tensor:
        if isinstance(seeds, int):
            seeds = list(range(seeds))
        noise_list = []
        B = latent.shape[0]
        shape = (B, *latent.shape[1:])
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed + seed_start)
            noise_sample = randn_tensor(shape, generator=generator, dtype=latent.dtype).to(latent.device)
            noise_list.append(noise_sample)
        return torch.cat(noise_list, dim=0)

    def compute_loss(
        self,
        pred: Tensor,
        gt: Tensor,
        pred_decode: Optional[Dict[str, Tensor]] = None,
        gt_decode: Optional[Dict[str, Tensor]] = None,
        data_mask_temporal: Optional[Tensor] = None,
    ) -> tuple[dict[str, Tensor], dict]:
        # loss function options and loss dict
        loss_fns = {"SmoothL1Loss": torch.nn.functional.smooth_l1_loss, "MSELoss": torch.nn.functional.mse_loss}
        loss_dict = {}

        # constrcut the data mask matrix
        recon_type = self.losses_cfg["recons"]["name"]
        recon_loss = loss_fns[recon_type](pred, gt, reduction="none").mean(dim=-1)
        loss_dict["recons"] = recon_loss
        # apply fk loss and mesh vertex loss only when predicting x1
        # if self.pred_type == "x1" or self.pred_type == "x1raw":
        # fk loss
        # if self.losses_cfg.get("fk", {}).get("weight", 0.0) > 0:
        #     fk_type = self.losses_cfg["fk"]["name"]
        #     fk_weight = self.losses_cfg["fk"]["weight"]
        #     pred_keypoints = self.calculate_keypoints(pred_decode)
        #     gt_keypoints = self.calculate_keypoints(gt_decode)
        #     fk_loss = loss_fns[fk_type](pred_keypoints, gt_keypoints, reduction="none").sum(dim=-1).mean(dim=-1)
        #     fk_loss_persample = (fk_loss * data_mask_temporal).sum(dim=-1) / data_mask_temporal.sum(dim=-1)
        #     fk_loss_mean = fk_weight * fk_loss_persample.mean()
        #     if torch.isnan(fk_loss_mean):
        #         breakpoint()
        #     loss_dict["loss"] += fk_loss_mean
        #     loss_dict["per_sample_loss"] += fk_loss_persample
        #     loss_dict.update({"fk_loss": fk_loss_mean})

        # mesh vertex loss
        if "vertex" in self.losses_cfg:
            loss_dict["vertex"] = self.vertex_loss(pred_decode, gt_decode, self.global_iteration)

        # roll-out translation loss
        if "transroll" in self.losses_cfg:
            loss_dict["transroll"] = self.transroll_loss(pred_decode, gt_decode, self.global_iteration)

        # joint2d loss
        if "joint2d" in self.losses_cfg:
            loss_dict["joint2d"] = self.joint2d_loss(pred_decode, gt_decode, self.global_iteration)

        loss_weight = {key: cfg["weight"] for key, cfg in self.losses_cfg.items()}
        loss_dict_mean = {}
        for key, val in loss_dict.items():
            if data_mask_temporal is not None:
                loss_dict_mean[key] = (val * data_mask_temporal).sum() / data_mask_temporal.sum()
            else:
                loss_dict_mean[key] = val.mean()
        return loss_dict_mean, loss_weight

    def forward_in_training(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        self.global_iteration += 1
        gt_motion = self.encode_motion(batch["target"])
        device = gt_motion.device
        length = batch["length"]
        max_length = length.max().item()
        ctxt_input = batch["inputs"]["feature"]
        # clip
        gt_motion = gt_motion[:, :max_length]
        if isinstance(ctxt_input, dict):
            for key in ctxt_input:
                ctxt_input[key] = ctxt_input[key][:, :max_length]
        else:
            ctxt_input = ctxt_input[:, :max_length]

        vtxt_input = torch.zeros(
            (gt_motion.shape[0], 1, self.motion_transformer.vtxt_input_dim), device=gt_motion.device
        )
        # x0 is gaussian noise
        # 使用CPU生成noise
        # x0 = torch.randn_like(gt_motion)
        x0 = torch.randn(gt_motion.shape).to(gt_motion.device)
        x1 = gt_motion
        # Sample a random timestep for each image
        # time step
        # 1000 与MDM兼容
        if "timestep_sample_method" in self.train_cfg:
            if self.train_cfg["timestep_sample_method"] == "logit_normal":
                timesteps = (
                    torch.randn(gt_motion.shape[0]) * self.train_cfg["t_sample_P_std"]
                    + self.train_cfg["t_sample_P_mean"]
                )
                timesteps = torch.sigmoid(timesteps)
            else:
                raise NotImplementedError(
                    f"Unsupported timestep sample method: {self.train_cfg['timestep_sample_method']}"
                )
            timesteps = timesteps.to(device)
        else:
            timesteps = torch.rand(
                (gt_motion.shape[0],),
                dtype=gt_motion.dtype,
            ).to(device)

        # str = f"[{self.__class__.__name__}] noise: {x0[:, 0, 0]}, timesteps: {timesteps}"
        # print(str)
        # sample xt (phi_t(x) in the paper)
        t = timesteps.unsqueeze(-1).unsqueeze(-1)
        phi = (1 - t) * x0 + t * x1
        flow = x1 - x0

        x_mask_temporal = length_to_mask(length, gt_motion.shape[1])

        # pred = self.motion_transformer( x=phi, ctxt_input=ctxt_input, vtxt_input=vtxt_input, timesteps=timesteps, x_mask_temporal=x_mask_temporal, ctxt_mask_temporal=x_mask_temporal, cond_mask_prob=self.cond_mask_prob)
        pred = self.motion_transformer(
            x=phi,
            ctxt_input=ctxt_input,
            vtxt_input=vtxt_input,
            timesteps=timesteps,
            x_mask_temporal=x_mask_temporal,
            ctxt_mask_temporal=x_mask_temporal,
            cond_mask_prob=self.cond_mask_prob,
        )
        if self.pred_type == "velocity":
            # do nothing
            pred_decode = None
            gt_decode = None
        elif self.pred_type == "x1":
            if "clean" in self.losses_cfg:
                clean_loss = self.clean_loss(pred, x1)
                clean_loss = (clean_loss * x_mask_temporal).sum() / x_mask_temporal.sum()
            pred_decode = self.decode_motion(pred)
            gt_decode = self.decode_motion(x1)
            # predict the original x1
            # https://github.com/LTH14/JiT/blob/main/denoiser.py#L56
            t_eps = 1 / self.validation_steps
            # check一下新的flow是否和老的flow一致的
            # check了基本一致的
            flow = (x1 - phi) / (1 - t).clamp_min(t_eps)
            # pred0 = pred
            pred = (pred - phi) / (1 - t).clamp_min(t_eps)
        elif self.pred_type == "x1raw":
            pred_decode = self.decode_motion(pred)
            gt_decode = self.decode_motion(x1)
            x1_pred = pred
            pred = x1_pred - x0
        else:
            raise NotImplementedError(f"Unsupported pred_type: {self.pred_type}")

        loss_dict, loss_weight = self.compute_loss(
            pred,
            flow,
            pred_decode=pred_decode,
            gt_decode=gt_decode,
            data_mask_temporal=x_mask_temporal,
        )
        if "clean" in self.losses_cfg:
            loss_dict["clean"] = clean_loss

        loss = sum([loss_dict[k] * loss_weight[k] for k in loss_dict.keys()])
        # print(losses.pop("per_sample_loss"))
        return {
            "latent": gt_motion,
            "model_output": pred,
            "loss": loss,
            "loss_dict": loss_dict,
            "tensor_results": {
                "index": batch.get("index", None),
            },
        }

    def create_closure_root_reproj_loss(self, optimized_params, optimizer, optimized_info, optimize_in_world=False):
        body25_root = optimized_info["body25_wotrans"]
        smpl24_root = optimized_info["smpl24_wotrans"]
        static_mask = optimized_info["static_mask"]
        static_conf = optimized_info["static_conf"]
        # 直接测试每一帧的单独的一个trans_camera
        feet_joint_ids = [7, 10, 8, 11]

        def gmof(x, sigma=100):
            """
            Geman-McClure error function
            """
            x_squared = x**2
            sigma_squared = sigma**2
            return (sigma_squared * x_squared) / (sigma_squared + x_squared)

        def reproj_loss(keypoints3d_camera_space, camera_K, keypoints2d, bbox_height):
            root_KRT = torch.einsum("fij,fkj->fki", camera_K, keypoints3d_camera_space)
            root_2d = root_KRT[..., :2] / (root_KRT[..., 2:3] + 1e-5)
            #
            if True:
                loss_kp = gmof(root_2d - keypoints2d[..., :2]) / bbox_height[..., None, None]

                conf = keypoints2d[..., 2]
                loss_kp = ((conf > 0.7) * loss_kp.mean(-1)).mean(-1).mean()
                loss_kp = loss_kp * 100.0
            else:
                conf = keypoints2d[..., 2]
                # 10 pixel
                loss = (
                    torch.nn.functional.smooth_l1_loss(root_2d, keypoints2d[..., :2], reduction="none", beta=10).sum(dim=-1)
                    * conf
                )
                loss_kp = loss.mean()
            return loss_kp

        def trans_physics_loss(trans, fps=30):
            # 1. 计算速度 (Velocity, 1阶)
            # shape: (..., F-1, 3)
            vel = (trans[..., 1:, :] - trans[..., :-1, :]) * fps

            # 2. 计算加速度 (Acceleration, 2阶)
            # shape: (..., F-2, 3)
            acc = (vel[..., 1:, :] - vel[..., :-1, :]) * fps

            # 3. 计算加加速度 (Jerk, 3阶)
            # shape: (..., F-3, 3)
            jerk = (acc[..., 1:, :] - acc[..., :-1, :]) * fps

            # 返回 Acc 和 Jerk 的 Loss
            loss_acc = torch.norm(acc, dim=-1).mean()
            loss_jerk = torch.norm(jerk, dim=-1).mean()

            return loss_acc, loss_jerk

        def rotation_physics_loss(R_vec, fps=30):
            # 1. 一阶：角速度平滑
            # rot_vel = (R_vec[1:] - R_vec[:-1]) * fps
            # loss_rot_vel = torch.norm(rot_vel[1:] - rot_vel[:-1], dim=-1).mean() # 实际上是角加速度的趋势

            # 2. 二阶：角加速度 (Acceleration)
            # 捕捉旋转的受力平滑度
            rot_acc = (R_vec[2:] - 2 * R_vec[1:-1] + R_vec[:-2]) * (fps**2)
            loss_rot_acc = torch.norm(rot_acc, dim=-1).mean()

            # 3. 三阶：加角加速度 (Jerk/Jitter)
            # 消除高频微颤
            rot_jerk = (R_vec[3:] - 3 * R_vec[2:-1] + 3 * R_vec[1:-2] - R_vec[:-3]) * (fps**3)
            loss_rot_jerk = torch.norm(rot_jerk, dim=-1).mean()

            return loss_rot_acc, loss_rot_jerk

        def closure_camera_trans():
            optimizer.zero_grad()
            R_rel = angle_axis_to_rotation_matrix(optimized_params["R_rel_vec"])[0]

            loss_skating = 0.0
            loss_floor = 0.0
            loss_rot_smooth = 0.0
            loss_rot_acc = 0.0
            loss_rot_jerk = 0.0

            if optimize_in_world:
                # 优化trans_in_world，同时增加物理约束
                trans_in_world = optimized_params["trans_in_world"]

                # 1. 提取脚部位置
                feet_local = smpl24_root[:, feet_joint_ids, :]  # (T, 4, 3)

                if R_rel.ndim == 2:
                    R_rel_expanded = R_rel[None, None, ...].expand(feet_local.shape[0], 4, -1, -1)
                elif R_rel.shape[0] == 1:
                    R_rel_expanded = R_rel[:, None, ...].expand(feet_local.shape[0], 4, -1, -1)

                # Pos_World = R_rel @ (Pos_Local + Trans)
                feet_global = feet_local + trans_in_world[:, None, :]
                delta_pos = (feet_global[1:] - feet_global[:-1]) ** 2

                loss_contact_vel = delta_pos.sum(dim=-1) * static_conf[:-1, :4]
                # feet_pos_world = torch.einsum("tvij,tvj->tvi", R_rel_expanded, pos_translated)

                # # 2. 计算有梯度的速度
                # feet_vel_norm = (feet_pos_world[1:] - feet_pos_world[:-1]).norm(2, dim=-1) * 30

                # 3. 计算滑步 Loss

                # 4. 地面穿透 Loss
                # min_h = feet_pos_world[..., 1].min(dim=-1)[0]  # 取4个脚点的最低点，y=0为地面
                # breakpoint()
                # loss_floor = torch.nn.functional.relu(-min_h).mean()
                # # torch.abs(min_h[1:] - min_h[:-1]).mean() # 平滑floor的loss

                # --- 计算投影和 Body25 ---
                R_compose = optimized_info["camera_RT"][:, :3, :3] @ R_rel[None]
                k3d = (
                    torch.einsum("fij,fkj->fki", R_compose, body25_root + optimized_params["trans_in_world"][:, None])
                    + optimized_info["camera_RT"][:, :3, 3][:, None]
                )
                # loss_acc, loss_jitter = trans_physics_loss(trans_in_world, fps=30)
                # loss_rot_acc, loss_rot_jerk = rotation_physics_loss(optimized_params["R_rel_vec"][0], fps=30) # smooth
                extra_loss = loss_contact_vel.mean() * 300

            else:
                # Camera Space 优化 (不加物理约束，只优化重投影trans_camera)
                trans_camera = optimized_params["trans_in_camera"]
                R_compose = optimized_info["camera_RT"][:, :3, :3] @ R_rel[None]
                k3d = torch.einsum("fij,fkj->fki", R_compose, body25_root) + trans_camera[:, None]
                # loss_acc, loss_jitter = trans_physics_loss(trans_camera, fps=30)
                # loss_rot_acc, loss_rot_jerk = rotation_physics_loss(optimized_params["R_rel_vec"][0], fps=30) # smooth
                extra_loss = 0.0

            loss_reproj = reproj_loss(k3d, optimized_info["camera_K"], optimized_info["keypoints2d"], optimized_info["bbox_height"])

            # 组合 Loss
            # loss_smooth = (
            #     loss_acc * 0.01 + loss_jitter * 0.001 + loss_rot_acc * 0.0 + loss_rot_jerk * 0.00
            # )  # R的平滑暂时为0
            # loss = loss_reproj + loss_smooth + loss_skating * 30.0 + loss_floor * 0.0  # 权重待定,目前版本没加地面loss
            loss = loss_reproj + extra_loss
            loss.backward()
            return loss

        return closure_camera_trans

    def _run_optimizer(self, optimizer_params, optimized_keys, optimized_info, **other_flags):
        optimizer = torch.optim.LBFGS(
            [optimizer_params[key] for key in optimized_keys], lr=1.0, max_iter=100, line_search_fn="strong_wolfe"
        )
        closure = self.create_closure_root_reproj_loss(optimizer_params, optimizer, optimized_info, **other_flags)
        loss_prev = 1e10
        for i in range(100):
            loss = optimizer.step(closure)
            print(f"Iteration {i}, loss: {loss}")
            if abs(loss - loss_prev) < 1e-4:
                break
            loss_prev = loss
        return loss

    @staticmethod
    def prepare_optimized_infos(body_model_body25, body_model_smpl24, batch, output, i):
        # 固定参数：相机的RT
        camera_RT = batch["meta"]["camera_wv_RT"]
        assert len(camera_RT.shape) == 4 and camera_RT.shape[0] == 1, "camera_RT should be (1, nframes, 4, 4)"
        camera_RT = camera_RT[0]
        # Print camera center coordinate ranges and first/last frame positions
        # camera_RT is world2camera, so we need to invert it to get camera2world
        camera2world_RT = torch.inverse(camera_RT)
        camera_centers = camera2world_RT[:, :3, 3]  # Extract translation part (camera centers in world coordinates)
        print(f"Camera center X range: [{camera_centers[:, 0].min():.4f}, {camera_centers[:, 0].max():.4f}]")
        print(f"Camera center Y range: [{camera_centers[:, 1].min():.4f}, {camera_centers[:, 1].max():.4f}]")
        print(f"Camera center Z range: [{camera_centers[:, 2].min():.4f}, {camera_centers[:, 2].max():.4f}]")
        print(
            f"First frame camera center: [{camera_centers[0, 0]:.4f}, {camera_centers[0, 1]:.4f}, {camera_centers[0, 2]:.4f}]"
        )
        print(
            f"Last frame camera center: [{camera_centers[-1, 0]:.4f}, {camera_centers[-1, 1]:.4f}, {camera_centers[-1, 2]:.4f}]"
        )
        camera_K = batch["meta"]["camera_origin_K"]
        assert len(camera_K.shape) == 4 and camera_K.shape[0] == 1, "camera_K should be (1, nframes, 3, 3)"
        camera_K = camera_K[0]
        # rot6d: (nframes, 52, 6)
        rot6d = output["rot6d"][i]
        shapes = output["shapes"][i]
        # trans_init: (nframes, 3)
        trans_init = output["trans"][i]
        # 目标：
        keypoints2d = batch["inputs"]["feature"]["keypoints2d"][0]
        bbox = batch["inputs"]["feature"]["bbox_value"][0]
        bbox_height = bbox[:, 3] - bbox[:, 1]
        assert len(keypoints2d.shape) == 3 and keypoints2d.shape[1] == 25, "keypoints2d should be (nframes, 25, 2)"
        params_fix = {
            "rot6d": rot6d,
            # "trans": trans_init,
            "shapes": shapes[:1],
        }
        smpl24_wotrans = body_model_smpl24(params_fix)["keypoints3d"]
        body25_wotrans = body_model_body25(params_fix)["vertices_wotrans"]

        pred_end_vel = output["end_effector_vel"][i].clone()

        pred_vel_norm = pred_end_vel.norm(p=2, dim=-1) / 30
        static_mask = (pred_vel_norm[:, :4] < 1e-2).float()
        static_conf = end_vel_to_static_conf(pred_end_vel)

        return {
            "camera_RT": camera_RT,
            "camera_K": camera_K,
            "keypoints2d": keypoints2d,
            "bbox_height": bbox_height,
            "trans_init": trans_init,
            "body25_wotrans": body25_wotrans,
            "smpl24_wotrans": smpl24_wotrans,
            "static_mask": static_mask,
            "static_conf": static_conf,
        }

    @staticmethod
    def _optimize_estimate_z(camera_K, keypoints2d):
        # 1. 获取焦距 fx
        focal_length = camera_K[0, 0, 0]  # fx

        # 2. 提取躯干关键点 (Neck: 1, MidHip: 8)
        # keypoints2d shape: (Frames, 25, 3) -> [x, y, conf]
        kp_neck = keypoints2d[:, 1, :2]
        kp_mid_hip = keypoints2d[:, 8, :2]
        kp_conf = torch.sqrt(torch.clamp(keypoints2d[:, 1, 2] * keypoints2d[:, 8, 2], min=0.0))

        # 3. 计算图像上的躯干长度 (像素)
        # 加上 1e-5 防止除零
        torso_pixel_len = (kp_neck - kp_mid_hip).norm(dim=1) + 1e-5

        # 4. 设定真实世界的参考长度
        # 成年人从颈部到骨盆的平均长度大约是 0.55 米 ~ 0.6 米
        avg_torso_len_meter = 0.55

        # 5. 利用相似三角形反推 Z
        estimated_z_raw = (focal_length * avg_torso_len_meter) / torso_pixel_len

        # --- 6. 鲁棒性处理 ---
        # 过滤掉置信度太低的点（检测错误的点会导致 Z 飞到无穷远或无穷近）
        valid_mask = (kp_conf > 0.3) & (estimated_z_raw > 0.5) & (estimated_z_raw < 15.0)

        if valid_mask.sum() > 0:

            median_z = torch.median(estimated_z_raw[valid_mask])
            # 默认填充中值
            estimated_z = torch.ones_like(estimated_z_raw) * median_z
            estimated_z[valid_mask] = estimated_z_raw[valid_mask]
        else:
            estimated_z = torch.ones_like(estimated_z_raw) * 2.0
        return estimated_z

    def optimize_motion(self, body_model, batch, output, debug=False, optimize_scale=False):
        # RTE指标是不考虑scale的，所以不能在这里优化scale
        # 对每个结果单独优化
        for i in range(output["rot6d"].shape[0]):
            # 只优化translation和整体的rotation, scale
            device, dtype = output["rot6d"].device, output["rot6d"].dtype
            optimized_infos = self.prepare_optimized_infos(body_model, self.body_model, batch, output, i)
            # nframes, 3
            # 需要计算一个最小化loss的旋转
            optimized_params = {
                "R_rel_vec": torch.zeros((1, 3), device=device, dtype=dtype),
                "T_rel_vec": torch.zeros((1, 3), device=device, dtype=dtype),
                "S_logit": torch.zeros((1, 1, 1), device=device, dtype=dtype),
                "trans_in_camera": torch.zeros_like(optimized_infos["trans_init"]),
            }
            estimated_z = self._optimize_estimate_z(optimized_infos["camera_K"], optimized_infos["keypoints2d"])

            # 赋值
            optimized_params["trans_in_camera"][:, 2] = estimated_z.detach()
            # optimized_params["trans_in_camera"][:, 2] = 2.0  # 大约放到2米处

            def require_grad(params, keys, flag=True):
                for key in keys:
                    params[key].requires_grad = flag

            optimized_keys = ["R_rel_vec", "trans_in_camera"]
            if optimize_scale:
                optimized_keys.append("S_logit")
            require_grad(optimized_params, optimized_keys, True)
            loss = self._run_optimizer(
                optimized_params,
                optimized_keys,
                optimized_info=optimized_infos,
            )
            # 求解trans
            with torch.no_grad():
                Rrel = angle_axis_to_rotation_matrix(optimized_params["R_rel_vec"])[0].detach()
                trans_camera = optimized_params["trans_in_camera"].detach()
                # Rcompose: (nframes, 3, 3)
                Rcompose = optimized_infos["camera_RT"][:, :3, :3] @ Rrel[None]
                # Tc: (nframes, 3)
                Tc = optimized_infos["camera_RT"][:, :3, 3]
                trans_world = torch.inverse(Rcompose) @ (trans_camera.reshape(-1, 3, 1) - Tc.reshape(-1, 3, 1))
                trans_world = trans_world.reshape(-1, 3)
            optimized_params.pop("trans_in_camera")
            optimized_params["trans_in_world"] = trans_world
            optimized_keys = ["R_rel_vec", "trans_in_world"]
            require_grad(optimized_params, optimized_keys, True)
            loss = self._run_optimizer(
                optimized_params,
                optimized_keys,
                optimized_info=optimized_infos,
                optimize_in_world=True,
            )
            trans_world = optimized_params["trans_in_world"].detach()

            if debug:
                from ..datasets.v2m_generation.base import _check_by_matplotlib, _check_by_reproj

                smpl24_wotrans_optimized = torch.einsum(
                    "ij,fkj->fki", Rrel, optimized_infos["smpl24_wotrans"] + trans_world[:, None]
                )
                start_frame = batch["meta"]["start_frame"][0]
                _check_by_matplotlib(
                    batch["meta"]["video_name"][0],
                    smpl24_wotrans_optimized,
                    optimized_infos["camera_K"],
                    optimized_infos["camera_RT"],
                    postfix="_optimized_RT_smpl24",
                    start_frame=start_frame,
                )

                # _check_by_reproj(
                #     batch["meta"]["video_name"][0],
                #     smpl24_wotrans_optimized,
                #     optimized_infos["camera_K"],
                #     optimized_infos["camera_RT"],
                #     keypoints2d=optimized_infos["keypoints2d"],
                #     postfix="_optimized_RT",
                #     start_frame=start_frame,
                # )
                breakpoint()
            # if True:
            #     optimized_keys = ["R_rel_vec", "T_rel_vec"]
            #     if optimize_scale:
            #         optimized_keys.append("S_logit")
            #     optimized_keys.append("trans")
            #     require_grad(optimized_params, optimized_keys, True)
            #     loss = self._run_optimizer(
            #         optimized_params,
            #         optimized_keys,
            #         camera_RT=camera_RT,
            #         camera_K=camera_K,
            #         keypoints2d=keypoints2d,
            #         body25_wotrans=body25_wotrans,
            #     )

            #     if debug:
            #         R = angle_axis_to_rotation_matrix(optimized_params["R_rel_vec"])[0]
            #         T = optimized_params["T_rel_vec"]
            #         S = map_scale_logits(optimized_params["S_logit"])
            #         smpl24_wotrans_optimized = (
            #             S * torch.einsum("ij,fkj->fki", R, smpl24_wotrans + optimized_params["trans"][:, None])
            #             + T[:, None]
            #         )
            #         _check_by_matplotlib(
            #             batch["meta"]["video_name"][0],
            #             smpl24_wotrans_optimized,
            #             camera_K,
            #             camera_RT,
            #             postfix="_optimized_trans_smpl24",
            #             start_frame=start_frame,
            #         )

            #         _check_by_reproj(
            #             batch["meta"]["video_name"][0],
            #             smpl24_wotrans_optimized,
            #             camera_K,
            #             camera_RT,
            #             keypoints2d=keypoints2d,
            #             postfix="_optimized_trans",
            #             start_frame=start_frame,
            #         )
            # if debug:
            #     print("final scale: ", S)
            #     print("final trans: ", T)
            #     print("final R: ", R)
            # output["trans"][i] = optimized_params["trans"].detach()
            output["trans"][i] = trans_world.detach()

        output["trans"] = gaussian_smooth(output["trans"], 3, -2)
        return output

    @torch.no_grad()
    def validate(self, batch: Dict[str, Any], seeds: List[int] = [0, 1, 2, 3], do_postproc: bool = False, cfg=1):
        length = batch["length"]
        camera_is_static = batch[
            "camera_is_static"
        ]  # NOTE: use camera_is_static to mask the eval loss for moving camera samples
        gt_motion = self.encode_motion(batch["target"])
        device = gt_motion.device
        # inputs feature: (B, L, D)
        # TODO: 应该把2D关键点也塞进去；但是2D关键点可能需要做normalize
        feature = batch["inputs"]["feature"]
        repeat = len(seeds)
        if cfg == 1:
            if isinstance(feature, dict):
                for key in feature.keys():
                    feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
            else:
                feature = feature.repeat((repeat,) + (1,) * (feature.dim() - 1))
        else:
            # 为了CFG进行准备数据
            # 把0放到前面
            if isinstance(feature, dict):
                for key in feature.keys():
                    feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
                    # FIXME: 只有key = feature 的时候需要drop
                    if key == "feature":
                        feature[key] = torch.cat([torch.zeros_like(feature[key]), feature[key]], dim=0)
                    else:
                        feature[key] = torch.cat([feature[key], feature[key]], dim=0)
            else:
                # 只有一个feature的时候，默认需要直接drop成0
                feature = feature.repeat((repeat,) + (1,) * (feature.dim() - 1))
                feature = torch.cat([torch.zeros_like(feature), feature], dim=0)

        vtxt_input = torch.zeros((repeat, 1, self.motion_transformer.vtxt_input_dim), device=device)
        x_mask_temporal = length_to_mask(length, gt_motion.shape[1])
        x_mask_temporal = x_mask_temporal.repeat((repeat,) + (1,) * (x_mask_temporal.dim() - 1))
        if cfg != 1:
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)
            vtxt_input = torch.cat([vtxt_input] * 2, dim=0)

        # mask for moving camera samples
        camera_is_static_repeated = camera_is_static.repeat(repeat)
        if cfg != 1:
            camera_is_static_repeated = torch.cat([camera_is_static_repeated] * 2, dim=0)
        moving_camera_mask = (~camera_is_static_repeated).unsqueeze(-1) * x_mask_temporal  # (B * repeat, L)

        do_classifier_free_guidance = cfg > 1

        dtype = gt_motion.dtype
        y0 = self.noise_from_seeds(gt_motion, seeds)

        def fn(t: Tensor, x: Tensor) -> Tensor:
            # predict flow
            x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
            x_pred = self.motion_transformer(
                x=x_input,
                ctxt_input=feature,
                vtxt_input=vtxt_input,
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=x_mask_temporal,
            )
            if self.pred_type == "velocity":
                pass
            elif self.pred_type == "x1":
                # predict the original x1
                # https://github.com/LTH14/JiT/blob/main/denoiser.py#L94
                t_eps = 1 / self.validation_steps
                x_pred = (x_pred - x_input) / (1.0 - t).clamp_min(t_eps)
            elif self.pred_type == "x1raw":
                # 直接减去最开始的噪声；方向始终是指向x1
                x_pred = x_pred - y0
            else:
                raise NotImplementedError(f"Unsupported pred_type: {self.pred_type}")

            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + cfg * (x_pred_text - x_pred_basic)
            return x_pred

        t = torch.linspace(0, 1, self.validation_steps + 1, device=device, dtype=dtype)
        trajectory = odeint_custom(fn, y0, t, **self._noise_scheduler_cfg)
        sampled: Tensor = trajectory[-1]

        # decode motion representation
        if self.motion_rep == "gvrot6d":
            output = self.decode_motion(sampled, R_c2gv=batch["target"]["R_c2gv"], camera_R=batch["target"]["camera_R"])
            gt_decode = self.decode_motion(
                gt_motion, R_c2gv=batch["target"]["R_c2gv"], camera_R=batch["target"]["camera_R"]
            )
        elif self.motion_rep == "wvrot6d":
            output = self.decode_motion(sampled)
            gt_decode = self.decode_motion(gt_motion)
        elif self.motion_rep == "wvrot6dstd":
            output = self.decode_motion(sampled)
            gt_decode = self.decode_motion(gt_motion)
        elif self.motion_rep == "wvrot6d_transl_std":
            output = self.decode_motion(sampled)
            gt_decode = self.decode_motion(gt_motion)
        elif self.motion_rep == "wvrot6d_transl_shape_std":
            output = self.decode_motion(sampled)
            gt_decode = self.decode_motion(gt_motion)
        elif self.motion_rep == "wvrot6d_transl_shape_stationary_std":
            output = self.decode_motion(sampled, camera_T=feature["camera_T"])
            gt_decode = self.decode_motion(gt_motion, is_gt=True)
        else:
            raise NotImplementedError(f"Unsupported motion representation: {self.motion_rep}")

        # post-processing
        if do_postproc:
            if self.evaluate_on == "EMDB":
                print(batch["meta"]["sequence_name"][0])
                output["trans"] = pp_static_joint(self.body_model, output, replace_ground_y=False)
                output["rot6d"][..., 1:22, :] = process_ik(self.body_model, output)
                self.build_body_model_sparse()
                output = self.optimize_motion(self.body_model_sparse, batch, output)
                if 'skateboard' in batch["meta"]["sequence_name"][0] or '57_outdoor_rock_chair' in batch["meta"]["sequence_name"][0]:
                    pass
                else:
                    output["trans"] = pp_static_joint(self.body_model, output, replace_ground_y=True)
                    output["rot6d"][..., 1:22, :] = process_ik(self.body_model, output)
            else:
                output["trans"] = pp_static_joint_footonly_v2(self.body_model, output, replace_ground_y=True)
                # 默认需要打开后处理IK
                if batch["meta"].get("do_post_ik", True):
                    save_fk_end_effector_debug(self.body_model, output, name="before_ik")
                    output["rot6d"][..., 1:22, :] = process_ik(self.body_model, output, debug=True)
                    save_fk_end_effector_debug(self.body_model, output, name="after_ik")

                if batch["meta"].get("do_camera_fitting", False):
                    self.build_body_model_sparse()
                    output = self.optimize_motion(self.body_model_sparse, batch, output)


        # 推理完成，计算指标的时候需要恢复
        if do_classifier_free_guidance:
            x_mask_temporal = x_mask_temporal[: x_mask_temporal.shape[0] // 2]

        # forward kinematics
        pred_j3d = get_joints_from_smpl_params(self.body_model, output, joint_num=52)
        pred_j3d_local = pred_j3d["local_joints"]
        pred_j3d_global = pred_j3d["global_joints"]
        gt_j3d = get_joints_from_smpl_params(self.body_model, gt_decode, joint_num=52)
        gt_j3d_local = gt_j3d["local_joints"]
        gt_j3d_global = gt_j3d["global_joints"]

        # evaluate local joint distance error with zero translation
        dist_local_body = torch.norm(pred_j3d_local[:, :, :22, :] - gt_j3d_local[:, :, :22, :], dim=-1).mean(dim=-1)
        dist_local_hand = torch.norm(pred_j3d_local[:, :, 22:, :] - gt_j3d_local[:, :, 22:, :], dim=-1).mean(dim=-1)
        dist_local_body = (dist_local_body * x_mask_temporal).sum() / x_mask_temporal.sum()
        dist_local_hand = (dist_local_hand * x_mask_temporal).sum() / x_mask_temporal.sum()

        # evaluate root joint, body joint, and hand joint angle error
        angular_error_root = torch.norm(output["rot6d"][:, :, 0, :] - gt_decode["rot6d"][:, :, 0, :], dim=-1)
        angular_error_body = torch.norm(
            output["rot6d"][:, :, 1:22, :] - gt_decode["rot6d"][:, :, 1:22, :], dim=-1
        ).mean(dim=-1)
        angular_error_hand = torch.norm(output["rot6d"][:, :, 22:, :] - gt_decode["rot6d"][:, :, 22:, :], dim=-1).mean(
            dim=-1
        )
        angular_error_root = (angular_error_root * x_mask_temporal).sum() / x_mask_temporal.sum()
        angular_error_body = (angular_error_body * x_mask_temporal).sum() / x_mask_temporal.sum()
        angular_error_hand = (angular_error_hand * x_mask_temporal).sum() / x_mask_temporal.sum()

        # evaluate local joint jitter with zero translation
        jitter_local_body = self.calculate_joint_jitter(pred_j3d_local[:, :, :22, :])
        jitter_local_hand = self.calculate_joint_jitter(pred_j3d_local[:, :, 22:, :])
        jitter_local_body = (jitter_local_body * x_mask_temporal).sum() / x_mask_temporal.sum()
        jitter_local_hand = (jitter_local_hand * x_mask_temporal).sum() / x_mask_temporal.sum()

        # collect all evaluation metrics
        metrics = {
            "dist_local_body": dist_local_body,
            "dist_local_hand": dist_local_hand,
            "angular_error_root": angular_error_root,
            "angular_error_body": angular_error_body,
            "angular_error_hand": angular_error_hand,
            "jitter_local_body": jitter_local_body,
            "jitter_local_hand": jitter_local_hand,
        }

        # evaluate global translation error and jitter if involve translation in motion representation
        if "_transl_" in self.motion_rep:
            dist_abs_transl = torch.norm(output["trans"] - gt_decode["trans"], dim=-1)
            dist_abs_transl = (dist_abs_transl * x_mask_temporal).sum() / x_mask_temporal.sum()
            jitter_abs_transl = self.calculate_joint_jitter(output["trans"].unsqueeze(2))
            jitter_abs_transl = (jitter_abs_transl * x_mask_temporal).sum() / x_mask_temporal.sum()
            metrics.update({"dist_abs_transl": dist_abs_transl, "jitter_abs_transl": jitter_abs_transl})

        # evaluate body shape error if involve shapes in motion representation
        if "_shape_" in self.motion_rep:
            shape_error = torch.norm(output["shapes"] - gt_decode["shapes"][:, 0:1], dim=-1)
            shape_error = (shape_error * x_mask_temporal).sum() / x_mask_temporal.sum()
            metrics.update({"body_shape_error": shape_error})

        # evaluate metrics for moving camera samples only
        if locals().get("moving_camera_mask", None) is not None and moving_camera_mask.sum() > 0:
            dist_local_body_moving = torch.norm(pred_j3d_local[:, :, :22, :] - gt_j3d_local[:, :, :22, :], dim=-1).mean(
                dim=-1
            )
            dist_local_hand_moving = torch.norm(pred_j3d_local[:, :, 22:, :] - gt_j3d_local[:, :, 22:, :], dim=-1).mean(
                dim=-1
            )
            dist_local_body_moving = (dist_local_body_moving * moving_camera_mask).sum() / moving_camera_mask.sum()
            dist_local_hand_moving = (dist_local_hand_moving * moving_camera_mask).sum() / moving_camera_mask.sum()

            angular_error_root_moving = torch.norm(output["rot6d"][:, :, 0, :] - gt_decode["rot6d"][:, :, 0, :], dim=-1)
            angular_error_body_moving = torch.norm(
                output["rot6d"][:, :, 1:22, :] - gt_decode["rot6d"][:, :, 1:22, :], dim=-1
            ).mean(dim=-1)
            angular_error_hand_moving = torch.norm(
                output["rot6d"][:, :, 22:, :] - gt_decode["rot6d"][:, :, 22:, :], dim=-1
            ).mean(dim=-1)
            angular_error_root_moving = (
                angular_error_root_moving * moving_camera_mask
            ).sum() / moving_camera_mask.sum()
            angular_error_body_moving = (
                angular_error_body_moving * moving_camera_mask
            ).sum() / moving_camera_mask.sum()
            angular_error_hand_moving = (
                angular_error_hand_moving * moving_camera_mask
            ).sum() / moving_camera_mask.sum()

            jitter_local_body_moving = self.calculate_joint_jitter(pred_j3d_local[:, :, :22, :])
            jitter_local_hand_moving = self.calculate_joint_jitter(pred_j3d_local[:, :, 22:, :])
            jitter_local_body_moving = (jitter_local_body_moving * moving_camera_mask).sum() / moving_camera_mask.sum()
            jitter_local_hand_moving = (jitter_local_hand_moving * moving_camera_mask).sum() / moving_camera_mask.sum()

            metrics.update(
                {
                    "dist_local_body_moving": dist_local_body_moving,
                    "dist_local_hand_moving": dist_local_hand_moving,
                    "angular_error_root_moving": angular_error_root_moving,
                    "angular_error_body_moving": angular_error_body_moving,
                    "angular_error_hand_moving": angular_error_hand_moving,
                    "jitter_local_body_moving": jitter_local_body_moving,
                    "jitter_local_hand_moving": jitter_local_hand_moving,
                }
            )

            if "_transl_" in self.motion_rep:
                dist_abs_transl_moving = torch.norm(output["trans"] - gt_decode["trans"], dim=-1)
                dist_abs_transl_moving = (dist_abs_transl_moving * moving_camera_mask).sum() / moving_camera_mask.sum()
                jitter_abs_transl_moving = self.calculate_joint_jitter(output["trans"].unsqueeze(2))
                jitter_abs_transl_moving = (
                    jitter_abs_transl_moving * moving_camera_mask
                ).sum() / moving_camera_mask.sum()
                metrics.update(
                    {
                        "dist_abs_transl_moving": dist_abs_transl_moving,
                        "jitter_abs_transl_moving": jitter_abs_transl_moving,
                    }
                )

            if "_shape_" in self.motion_rep:
                shape_error_moving = torch.norm(output["shapes"] - gt_decode["shapes"][:, 0:1], dim=-1)
                shape_error_moving = (shape_error_moving * moving_camera_mask).sum() / moving_camera_mask.sum()
                metrics.update({"body_shape_error_moving": shape_error_moving})

        return {
            "metrics": metrics,
            "output": {
                "pred": output,
                "gt": gt_decode,
            },
        }

    @torch.no_grad()
    def generate(
        self,
        feature: Dict[str, Tensor],
        seeds: List[int],
        length: int,
        camera_is_static: bool,
        cfg_scale: float,
        do_postproc: bool = False,
        debug: bool = False,
    ) -> Dict[str, Tensor]:
        assert isinstance(feature, dict), "feature must be a dict"
        assert (
            "feature" in feature and "camera_R" in feature and "camera_T" in feature
        ), "feature must contain feature, camera_R and camera_T"
        device = get_module_device(self)

        repeat = len(seeds)
        camera_T = feature["camera_T"].clone()
        for key in feature.keys():
            feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
        vtxt_input = torch.zeros((repeat, 1, self.motion_transformer.vtxt_input_dim), device=device)
        x_mask_temporal = length_to_mask(torch.tensor([length]), self.train_frames)
        x_mask_temporal = x_mask_temporal.repeat((repeat,) + (1,) * (x_mask_temporal.dim() - 1))

        # mask for moving camera samples
        camera_is_static_repeated = torch.tensor(camera_is_static).repeat(repeat)
        moving_camera_mask = (~camera_is_static_repeated).unsqueeze(-1) * x_mask_temporal  # (B * repeat, L)

        text_guidance_scale = cfg_scale if cfg_scale is not None else self.text_guidance_scale
        do_classifier_free_guidance = text_guidance_scale > 1.0
        if do_classifier_free_guidance is True:
            if isinstance(feature, dict):
                for key in feature.keys():
                    silent_feat = torch.zeros_like(feature[key])
                    feature[key] = torch.cat([silent_feat, feature[key]], dim=0)
            else:
                silent_feat = torch.zeros_like(feature)
                feature = torch.cat([silent_feat, feature], dim=0)
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)

        t = torch.linspace(0, 1, self.validation_steps + 1, device=device)
        y0 = self.noise_from_seeds(
            torch.zeros(
                1,
                self.train_frames,
                self.motion_transformer.motion_input_dim,
                device=device,
            ),
            seeds,
        )

        def fn(t: Tensor, x: Tensor) -> Tensor:
            # predict flow
            x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
            x_pred = self.motion_transformer(
                x=x_input,
                ctxt_input=feature,
                vtxt_input=vtxt_input,
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=x_mask_temporal,
            )
            if self.pred_type == "velocity":
                pass
            elif self.pred_type == "x1":
                # predict the original x1
                # https://github.com/LTH14/JiT/blob/main/denoiser.py#L94
                t_eps = 1 / self.validation_steps
                x_pred = (x_pred - x_input) / (1.0 - t).clamp_min(t_eps)
            elif self.pred_type == "x1raw":
                # 直接减去最开始的噪声；方向始终是指向x1
                x_pred = x_pred - y0
            else:
                raise NotImplementedError(f"Unsupported pred_type: {self.pred_type}")

            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + self.text_guidance_scale * (x_pred_text - x_pred_basic)
            return x_pred

        with torch.no_grad():
            trajectory = odeint_custom(fn, y0, t, **self._noise_scheduler_cfg)
        sampled: Tensor = trajectory[-1]
        sampled = sampled[:, :length, ...].clone()

        # decode motion representation
        if self.motion_rep == "wvrot6d_transl_shape_stationary_std":
            output_dict = self.decode_motion(sampled, camera_T=camera_T)
        else:
            raise NotImplementedError(f"Unsupported motion representation: {self.motion_rep}")

        # post-processing
        if do_postproc:
            self.build_body_model_sparse()
            # TODO: add optimize motion later on
            # output_dict = self.optimize_motion(self.body_model_sparse, batch, output_dict)

        return {**output_dict}


if __name__ == "__main__":
    # python -m hymotion.pipeline.motion_diffusion_v2m
    pass
