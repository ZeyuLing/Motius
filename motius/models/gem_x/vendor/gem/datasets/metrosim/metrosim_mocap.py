# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import io
import json
import os
import pickle
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import Dataset

from gem.utils.cam_utils import create_camera_sensor
from gem.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    normalize_T_w2c,
)
from gem.utils.motion_utils import (
    get_c_rootparam,
    get_R_c2gv,
    get_static_joint_mask,
    get_tgtcoord_rootparam,
)
from gem.utils.net_utils import (
    get_valid_mask,
    repeat_to_max_len,
    repeat_to_max_len_dict,
)
from gem.utils.pylogger import Log
from gem.utils.rotation_conversions import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)
from gem.utils.soma_utils.soma_layer import SomaLayer

from .cam_traj_utils import CameraAugmentorV11

contact_joint_ids = [69, 70, 74, 75, 14, 42]  # [L_Ankle, L_Foot, R_Ankle, R_Foot, L_Wrist, R_Wrist]


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


def aa_to_r6d(x):
    return matrix_to_rotation_6d(axis_angle_to_matrix(x))


def r6d_to_aa(x):
    return matrix_to_axis_angle(rotation_6d_to_matrix(x))


def rotate_around_axis(global_orient, transl, axis="y"):
    """Global coordinate augmentation. Random rotation around y-axis"""
    angle = torch.rand(1) * 2 * torch.pi
    if axis == "y":
        aa = torch.tensor([0.0, angle, 0.0]).float().unsqueeze(0)
    rmat = axis_angle_to_matrix(aa)
    global_orient = matrix_to_axis_angle(rmat @ axis_angle_to_matrix(global_orient))
    transl = (rmat.squeeze(0) @ transl.T).T
    return global_orient, transl


def augment_identity_coeffs(identity_coeffs, std=0.1):
    noise = torch.normal(mean=torch.zeros(45), std=torch.ones(45) * std)
    identity_coeffs_aug = identity_coeffs + noise[None]
    return identity_coeffs_aug


def augment_scale_params(scale_params, scale_comps, std=0.1):
    noise = torch.normal(mean=torch.zeros(68), std=torch.ones(68) * std)
    global_scale_noise = torch.normal(mean=torch.zeros(1), std=torch.ones(1) * std * 0.1)
    noise[18:] = 0
    noise[:18] = noise[:18] @ scale_comps[:18, :18]

    noise = torch.cat([global_scale_noise, noise])
    scale_params_aug = scale_params + noise[None]
    scale_params_aug[:, 0] = torch.clamp(scale_params_aug[:, 0], min=0.7, max=1.0)
    return scale_params_aug


def interpolate_soma_params(soma_params, tgt_len):
    """Speed augmentation via temporal interpolation.
    soma_params['body_pose'] (L, 76, 3)
    tgt_len: L -> L'
    """
    identity_coeffs = soma_params["identity_coeffs"]
    scale_params = soma_params["scale_params"]
    body_pose = soma_params["body_pose"]
    global_orient = soma_params["global_orient"]  # (L, 3)
    transl = soma_params["transl"]  # (L, 3)

    body_pose = rearrange(aa_to_r6d(body_pose.reshape(-1, 76, 3)), "l j c -> c j l")
    body_pose = F.interpolate(body_pose, tgt_len, mode="linear", align_corners=True)
    body_pose = r6d_to_aa(rearrange(body_pose, "c j l -> l j c")).reshape(-1, 76, 3)

    identity_coeffs = rearrange(identity_coeffs, "l c -> c 1 l")
    identity_coeffs = F.interpolate(identity_coeffs, tgt_len, mode="linear", align_corners=True)
    identity_coeffs = rearrange(identity_coeffs, "c 1 l -> l c")

    scale_params = rearrange(scale_params, "l c -> c 1 l")
    scale_params = F.interpolate(scale_params, tgt_len, mode="linear", align_corners=True)
    scale_params = rearrange(scale_params, "c 1 l -> l c")

    global_orient = rearrange(aa_to_r6d(global_orient.reshape(-1, 1, 3)), "l j c -> c j l")
    global_orient = F.interpolate(global_orient, tgt_len, mode="linear", align_corners=True)
    global_orient = r6d_to_aa(rearrange(global_orient, "c j l -> l j c")).reshape(-1, 3)

    transl = rearrange(transl, "l c -> c 1 l")
    transl = F.interpolate(transl, tgt_len, mode="linear", align_corners=True)
    transl = rearrange(transl, "c 1 l -> l c")

    return {
        "body_pose": body_pose,
        "identity_coeffs": identity_coeffs,
        "scale_params": scale_params,
        "global_orient": global_orient,
        "transl": transl,
    }


class MetrosimMocapDataset(Dataset):
    def __init__(
        self,
        data_root,
        motion_frames=120,
        l_factor=1.5,
        cam_augmentation="v11",
        debug=False,
        downsample_factor=1,
    ):
        super().__init__()
        self.data_root = Path(data_root)
        self.dataset_name = os.path.basename(data_root)
        self.motion_frames = motion_frames
        self.l_factor = l_factor
        self.cam_augmentation = cam_augmentation
        self.min_motion_frames = 60
        self.max_motion_frames = 120
        self.debug = debug
        self.downsample_factor = downsample_factor

        from gem.utils.hf_utils import download_soma_data

        soma_data_dir = download_soma_data()
        self.scale_comps = torch.load(f"{soma_data_dir}/scale_comps.pth")
        self.soma_lite = SomaLayer(data_root=soma_data_dir, low_lod=True, device="cpu")

        self._load_dataset()
        self._get_idx2meta()

    def _load_dataset(self):
        tic = Log.time()
        db = json.load(open(self.data_root / "train_val_split.json"))
        self.meta = db["meta"]
        self.db = db["train"]
        Log.info(
            f"[{self.dataset_name}] {len(self.db)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

    def _get_idx2meta(self):
        seq_lengths = []
        self.idx2meta = []
        max_num_samples = None
        if self.debug:
            max_num_samples = 1

        for sid in self.db:
            if max_num_samples is not None and len(self.idx2meta) > max_num_samples:
                break
            for agent_name in self.meta[sid]["agent_lengths"]:
                seq_length = self.meta[sid]["agent_lengths"][agent_name]
                num_samples = max(seq_length // self.max_motion_frames, 1)
                if seq_length < self.min_motion_frames:
                    continue
                seq_lengths.append(seq_length)
                self.idx2meta.extend([(sid, agent_name, 0, seq_length)] * num_samples)
        minutes = sum(seq_lengths) / 30 / 60
        if self.debug:
            self.idx2meta = self.idx2meta[1:]
        Log.info(
            f"[{self.dataset_name}] has {minutes:.1f} minutes motion -> Resampled to {len(self.idx2meta)} samples."
        )

    def __len__(self):
        return len(self.idx2meta) // self.downsample_factor

    def __getitem__(self, idx):
        """Override to handle data validation failures with retry logic"""
        max_retries = 10
        idx = random.randint(0, len(self.idx2meta) - 1)

        for _attempt in range(max_retries):
            try:
                data = self._load_data(idx)
                processed_data = self._process_data(data, idx)
                if processed_data is not None:
                    return processed_data
                else:
                    idx = (idx + 1) % len(self.idx2meta)
            except Exception as e:
                Log.warning(f"Error processing sample {idx}: {e}, trying next sample")
                idx = (idx + 1) % len(self.idx2meta)

        raise RuntimeError(f"Failed to load valid sample after {max_retries} attempts")

    def _load_data(self, idx):
        sid, agent_name, start, end = self.idx2meta[idx]
        mlength = end - start

        if mlength < self.min_motion_frames:
            pass
        elif mlength > self.max_motion_frames:
            effect_max_motion_len = min(self.max_motion_frames, mlength)
            length = np.random.randint(self.min_motion_frames, effect_max_motion_len + 1)
            start = np.random.randint(start, end - length + 1)
            end = start + length
        assert (
            end - start >= self.min_motion_frames
        ), f"end - start: {end - start} < {self.min_motion_frames}, mlength: {mlength}"

        vid = f"{sid}-{agent_name}"

        with open(os.path.join(self.data_root, "pose_pkls", f"{sid}.pkl"), "rb") as f:
            data = CPU_Unpickler(f).load()
            pose_data = data["pose_data"][agent_name]
            config_dir = data.get("config_dir", None)

        asset_name = pose_data["asset_name"]

        with open(os.path.join(self.data_root, "betas_69scale", f"{asset_name}.pkl"), "rb") as f:
            shapes = pickle.load(f)
            identity_coeffs = torch.from_numpy(shapes["identity_coeffs"]).float()
            scale_params = torch.from_numpy(shapes["scale_params"]).float()

        length = end - start
        poses = torch.from_numpy(pose_data["pose"][start:end]).float()
        body_pose = poses[:, 1:]  # (L, 76, 3)
        global_orient = poses[:, 0]  # (L, 3)
        pose_data = {
            "body_pose": body_pose,
            "global_orient": global_orient,
            "identity_coeffs": identity_coeffs.repeat(length, 1),
            "scale_params": scale_params.repeat(length, 1),
            "transl": torch.from_numpy(pose_data["transl"][start:end]).float(),
        }

        # Speed augmentation: select a random-length subset, then interpolate to tgt_len
        tgt_len = self.motion_frames
        raw_len = length
        raw_subset_len = np.random.randint(
            int(tgt_len / self.l_factor), int(tgt_len * self.l_factor)
        )
        if raw_subset_len <= raw_len:
            start = np.random.randint(0, raw_len - raw_subset_len + 1)
            end = start + raw_subset_len
        else:
            start = 0
            end = raw_len
        pose_data = {k: v[start:end] for k, v in pose_data.items()}

        pose_data_interpolated = interpolate_soma_params(pose_data, tgt_len)

        # AZ -> AY coordinate transform
        pose_data_interpolated["global_orient"], pose_data_interpolated["transl"], _ = (
            get_tgtcoord_rootparam(
                pose_data_interpolated["global_orient"],
                pose_data_interpolated["transl"],
                tsf="az->ay",
            )
        )

        return {
            "pose_data": pose_data_interpolated,
            "sid": sid,
            "vid": vid,
            "agent_name": agent_name,
            "config_dir": config_dir,
            "data_name": "metrosim_mocap",
        }

    def _process_data(self, data, idx):
        """
        Args:
            data: dict with pose_data in AY coordinates
        """
        data_name = data["data_name"]
        length = data["pose_data"]["body_pose"].shape[0]
        vid = data["vid"]
        config_dir = data["config_dir"]

        # Augmentation: identity/scale, y-axis rotation
        body_pose = data["pose_data"]["body_pose"]  # (L, 76, 3)
        identity_coeffs = augment_identity_coeffs(data["pose_data"]["identity_coeffs"])
        scale_params = augment_scale_params(data["pose_data"]["scale_params"], self.scale_comps)
        global_orient_w, transl_w = rotate_around_axis(
            data["pose_data"]["global_orient"], data["pose_data"]["transl"], axis="y"
        )
        del data

        soma_params_w = {
            "body_pose": body_pose,  # (L, 76, 3)
            "identity_coeffs": identity_coeffs,  # (L, 45)
            "scale_params": scale_params,  # (L, 69)
            "global_orient": global_orient_w,  # (L, 3)
            "transl": transl_w,  # (L, 3)
        }

        # Camera trajectory augmentation via SOMA joints
        if self.cam_augmentation == "v11":
            N = 10
            with torch.no_grad():
                soma_out = self.soma_lite(
                    global_orient=soma_params_w["global_orient"][::N],
                    body_pose=soma_params_w["body_pose"][::N],
                    identity_coeffs=soma_params_w["identity_coeffs"][::N],
                    scale_params=soma_params_w["scale_params"][::N],
                    transl=torch.zeros_like(soma_params_w["transl"][::N]),
                )
            w_j3d = soma_out["joints"]  # (K, 77, 3)
            w_j3d = (
                w_j3d.repeat_interleave(N, dim=0) + soma_params_w["transl"][:, None]
            )  # (L, 77, 3)

            width, height, K_fullimg = create_camera_sensor(1000, 1000, 43.3)
            cam_augmentor = CameraAugmentorV11()
            T_w2c = cam_augmentor(w_j3d, length)  # (L, 4, 4)
        else:
            raise NotImplementedError(f"cam_augmentation={self.cam_augmentation!r} not supported")

        normed_T_w2c = normalize_T_w2c(T_w2c)

        # Soma params in camera space
        soma_params_w["body_pose"] = soma_params_w["body_pose"].reshape(length, 76 * 3)
        offset = torch.zeros(3)
        global_orient_c, transl_c = get_c_rootparam(
            soma_params_w["global_orient"],
            soma_params_w["transl"],
            T_w2c,
            offset,
        )
        soma_params_c = {
            "body_pose": soma_params_w["body_pose"].clone(),
            "identity_coeffs": soma_params_w["identity_coeffs"].clone(),
            "scale_params": soma_params_w["scale_params"].clone(),
            "global_orient": global_orient_c,
            "transl": transl_c,
        }

        # World (gravity-aligned view) params
        gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)  # AY world
        R_c2gv = get_R_c2gv(T_w2c[:, :3, :3], gravity_vec)  # (L, 3, 3)

        # Camera motion features
        K_fullimg = K_fullimg.repeat(length, 1, 1)  # (L, 3, 3)
        cam_angvel = compute_cam_angvel(normed_T_w2c[:, :3, :3])  # (L, 6)
        cam_tvel = compute_cam_tvel(normed_T_w2c[:, :3, 3])  # (L, 3)

        # Static joint mask from world joints
        static_gt = get_static_joint_mask(w_j3d, vel_thr=0.15, repeat_last=True)  # (L, J)
        static_gt = static_gt[:, contact_joint_ids].float()  # (L, 6)

        max_len = length
        return_data = {
            "meta": {
                "data_name": data_name,
                "dataset_id": "metrosim_mocap",
                "vid": vid,
                "config_dir": config_dir,
                "idx": idx,
                "frame_start": 0,
                "frame_end": length,
            },
            "length": length,
            "soma_params_c": soma_params_c,
            "soma_params_w": soma_params_w,
            "R_c2gv": R_c2gv,
            "gravity_vec": gravity_vec,
            "bbx_xys": torch.zeros((length, 3)),  # placeholder
            "K_fullimg": K_fullimg,
            "f_imgseq": torch.zeros((length, 1024)),  # placeholder
            "kp2d": torch.zeros(length, 77, 3),  # placeholder
            "cam_angvel": cam_angvel,
            "cam_tvel": cam_tvel,
            "noisy_cam_tvel": cam_tvel,  # same as cam_tvel (no real image noise)
            "T_w2c": normed_T_w2c,
            "static_gt": static_gt,
            "mask": {
                "valid": get_valid_mask(length, length),
                "humanoid": get_valid_mask(max_len, 0),
                "has_img_mask": get_valid_mask(length, 0),
                "has_2d_mask": get_valid_mask(length, length),
                "has_cam_mask": get_valid_mask(length, length),
                "has_audio_mask": get_valid_mask(length, 0),
                "has_music_mask": get_valid_mask(length, 0),
                "valid_contact_mask": get_valid_mask(length, length),
                "2d_only": False,
                "vitpose": False,
                "bbx_xys": torch.zeros(length, dtype=torch.bool),
                "f_imgseq": False,
                "spv_incam_only": False,
                "invalid_contact": False,
            },
        }

        # Pad to batchable fixed length
        return_data["soma_params_c"] = repeat_to_max_len_dict(return_data["soma_params_c"], max_len)
        return_data["soma_params_w"] = repeat_to_max_len_dict(return_data["soma_params_w"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
        return_data["cam_tvel"] = repeat_to_max_len(return_data["cam_tvel"], max_len)
        return_data["noisy_cam_tvel"] = repeat_to_max_len(return_data["noisy_cam_tvel"], max_len)
        return_data["T_w2c"] = repeat_to_max_len(return_data["T_w2c"], max_len)
        return return_data
