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

from gem.datasets.imgfeat_motion.base_dataset import ImgfeatMotionDatasetBase
from gem.utils.geo_transform import (
    compute_cam_angvel,
    compute_cam_tvel,
    normalize_T_w2c,
    transform_mat,
)
from gem.utils.motion_utils import get_c_rootparam, get_R_c2gv, get_static_joint_mask
from gem.utils.net_utils import (
    get_valid_mask,
    pad_to_max_len,
    repeat_to_max_len,
    repeat_to_max_len_dict,
)
from gem.utils.pylogger import Log
from gem.utils.rotation_conversions import axis_angle_to_matrix

nvskel93to77_idx = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25,
    26,
    27,
    28,
    29,
    30,
    31,
    32,
    33,
    34,
    35,
    36,
    37,
    38,
    43,
    44,
    45,
    46,
    47,
    48,
    49,
    50,
    51,
    52,
    53,
    54,
    55,
    56,
    57,
    58,
    59,
    60,
    61,
    62,
    63,
    64,
    65,
    66,
    67,
    68,
    69,
    70,
    75,
    76,
    77,
    78,
    79,
    84,
    85,
    86,
    87,
    88,
]
nvskel77to33_idx = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    18,
    28,
    39,
    40,
    41,
    42,
    46,
    56,
    67,
    68,
    69,
    70,
    71,
    72,
    73,
    74,
    75,
    76,
]

# contact_joint_ids = [
#     7,
#     10,
#     8,
#     11,
#     20,
#     21,
# ]  # [L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist]


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda b: torch.load(io.BytesIO(b), map_location="cpu")
        else:
            return super().find_class(module, name)


contact_joint_ids = [69, 70, 74, 75, 14, 42]  # [L_Ankle, L_Foot, R_Ankle, R_Foot, L_Wrist, R_Wrist]


def get_bbx_xys(i_j2d, i_j2d_mask=None, bbx_ratio=None, do_augment=False, base_enlarge=1.2):
    """
    Args:
        i_j2d: (L, J, 3) [x,y,c] or (L, J, 2) [x,y]
        i_j2d_mask: (L, J) boolean mask indicating valid joints, if None use all joints
        bbx_ratio: [width, height] ratio for the bounding box
        do_augment: whether to apply random augmentation
        base_enlarge: factor to enlarge the bounding box
    Returns:
        bbx_xys: (L, 3) [center_x, center_y, size]
    """
    if bbx_ratio is None:
        bbx_ratio = [192, 256]

    """
    Args:
        i_j2d: (B, L, J, 3) [x,y,c] or (B, L, J, 2) [x,y]
        i_j2d_mask: (B, L, J) boolean mask indicating valid joints, if None use all joints
        bbx_ratio: [width, height] ratio for the bounding box
        do_augment: whether to apply random augmentation
        base_enlarge: factor to enlarge the bounding box
    Returns:
        bbx_xys: (B, L, 3) [center_x, center_y, size]
    """
    # Apply mask if provided
    if i_j2d_mask is not None:
        # Create a masked version of i_j2d for min/max calculations
        # For min calculation, set masked-out joints to large positive values
        # For max calculation, set masked-out joints to large negative values
        mask_expanded = i_j2d_mask.unsqueeze(-1)  # (B, L, J, 1)

        # Create copies for min and max calculations
        i_j2d_for_min = i_j2d.clone()
        i_j2d_for_max = i_j2d.clone()

        # Set coordinates of masked joints appropriately
        invalid_mask = ~mask_expanded.expand_as(i_j2d[..., :2])
        i_j2d_for_min[..., :2][invalid_mask] = float("inf")  # For min, set to large positive
        i_j2d_for_max[..., :2][invalid_mask] = float("-inf")  # For max, set to large negative

        # Calculate min/max using the filtered joints
        min_x = i_j2d_for_min[..., 0].min(-1)[0]
        max_x = i_j2d_for_max[..., 0].max(-1)[0]
        min_y = i_j2d_for_min[..., 1].min(-1)[0]
        max_y = i_j2d_for_max[..., 1].max(-1)[0]
    else:
        # Use all joints
        min_x = i_j2d[..., 0].min(-1)[0]
        max_x = i_j2d[..., 0].max(-1)[0]
        min_y = i_j2d[..., 1].min(-1)[0]
        max_y = i_j2d[..., 1].max(-1)[0]

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Size
    h = max_y - min_y  # (B, L)
    w = max_x - min_x  # (B, L)

    if True:  # fit w and h into aspect-ratio
        aspect_ratio = bbx_ratio[0] / bbx_ratio[1]
        mask1 = w > aspect_ratio * h
        h[mask1] = w[mask1] / aspect_ratio
        mask2 = w < aspect_ratio * h
        w[mask2] = h[mask2] * aspect_ratio

    # apply a common factor to enlarge the bounding box
    bbx_size = torch.max(h, w) * base_enlarge

    if do_augment:
        L = bbx_size.shape[0]
        device = bbx_size.device
        if True:
            # Smooth augmentation using temporally smoothed random noise (1D convolution)
            def smooth_noise(L, sigma=1.0, kernel_size=9, device=None):
                # Create base random noise
                noise = torch.randn(L + kernel_size - 1, device=device)
                # Build 1D Gaussian kernel

                t = torch.arange(kernel_size, device=device) - (kernel_size - 1) / 2
                gauss = torch.exp(-0.5 * (t / sigma) ** 2)
                gauss = gauss / gauss.sum()
                gauss = gauss.view(1, 1, kernel_size)
                # Apply 1D conv with padding='valid'
                noise = noise.unsqueeze(0).unsqueeze(0)  # (1, 1, L+ks-1)
                smooth = torch.nn.functional.conv1d(noise, gauss, padding=0)  # (1,1,L)
                return smooth.squeeze(0).squeeze(0)  # (L,)

            # Smooth random values for scale, tx, ty (each in [0,1])
            scale_smooth = smooth_noise(L, sigma=2.0, kernel_size=9, device=device) * 0.5 + 0.5
            tx_smooth = smooth_noise(L, sigma=2.0, kernel_size=9, device=device) * 0.5 + 0.5
            ty_smooth = smooth_noise(L, sigma=2.0, kernel_size=9, device=device) * 0.5 + 0.5

            # Now map to appropriate ranges
            scaleFactor = scale_smooth * 0.1 + 1.15  # 1.15~1.25, smooth
            txFactor = tx_smooth * 0.4 - 0.2  # -0.2 ~ 0.2, smooth
            tyFactor = ty_smooth * 0.4 - 0.2  # -0.2 ~ 0.2, smooth

        raw_bbx_size = bbx_size / base_enlarge
        bbx_size = raw_bbx_size * scaleFactor
        center_x += raw_bbx_size / 2 * ((scaleFactor - 1) * txFactor)
        center_y += raw_bbx_size / 2 * ((scaleFactor - 1) * tyFactor)

    return torch.stack([center_x, center_y, bbx_size], dim=-1)


class MetrosimDataset(ImgfeatMotionDatasetBase):
    def __init__(self, data_root, mode="train", debug=False, aug_bbx=True, aug_hb=False):
        self.data_root = Path(data_root)
        self.dataset_name = os.path.basename(data_root)
        self.aug_bbx = aug_bbx
        # self.soma = torch.jit.load("inputs/soma_data/soma_model_scripted.pt", map_location="cpus")
        self.min_motion_frames = 60
        self.max_motion_frames = 120
        self.mode = mode
        self.debug = debug
        self.aug_hb = aug_hb

        super().__init__()

    def __getitem__(self, idx):
        """Override to handle data validation failures with retry logic"""
        max_retries = 10
        # if self.debug:
        #     idx = 0
        for _attempt in range(max_retries):
            try:
                data = self._load_data(idx)
                processed_data = self._process_data(data, idx)
                if processed_data is not None:
                    return processed_data
                else:
                    # Try next sample if current one is invalid
                    idx = (idx + 1) % len(self.idx2meta)
            except Exception as e:
                Log.warning(f"Error processing sample {idx}: {e}, trying next sample")
                idx = (idx + 1) % len(self.idx2meta)

        # If all retries failed, raise error
        raise RuntimeError(f"Failed to load valid sample after {max_retries} attempts")

    def _load_dataset(self):
        tic = Log.time()
        db = json.load(open(self.data_root / "train_val_split.json"))
        self.meta = db["meta"]
        self.db = db["val" if self.mode in ["val", "test"] else "train"]
        Log.info(
            f"[{self.dataset_name} ({self.mode})] {len(self.db)} sequences. Elapsed: {Log.time() - tic:.2f}s"
        )

    def __len__(self):
        if self.mode in ["val"]:
            return 10
        elif self.mode in ["test"]:
            return 500
        return len(self.idx2meta)

    def _get_idx2meta(self):
        seq_lengths = []
        self.idx2meta = []
        if self.mode in ["val"]:
            max_num_samples = 500
        elif self.mode in ["test"]:
            max_num_samples = 500
        else:
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

    def _load_data(self, idx):
        if self.mode in ["val", "test"]:
            idx = random.randint(0, len(self.idx2meta) - 1)
        sid, agent_name, start, end = self.idx2meta[idx]
        mlength = end - start

        if mlength < self.min_motion_frames:
            pass
        elif mlength > self.max_motion_frames:
            effect_max_motion_len = min(self.max_motion_frames, mlength)
            length = np.random.randint(
                self.min_motion_frames, effect_max_motion_len + 1
            )  # [low, high)
            start = np.random.randint(start, end - length + 1)
            end = start + length
        assert (
            end - start >= self.min_motion_frames
        ), f"end - start: {end - start} < {self.min_motion_frames}, mlength: {mlength}"

        vid = f"{sid}-{agent_name}"

        with open(os.path.join(self.data_root, "pose_pkls", f"{sid}.pkl"), "rb") as f:
            data = CPU_Unpickler(f).load()
            # data = pickle.load(f)
            pose_data = data["pose_data"][agent_name]
            camera_data = data["camera_data"]
            config_dir = data.get("config_dir", None)

        data = {
            "pose_data": pose_data,
            "camera_data": camera_data,
            "sid": sid,
            "vid": vid,
            "agent_name": agent_name,
            "start": start,
            "end": end,
            "config_dir": config_dir,
        }
        return data

    def _process_data(self, data, idx):
        start = data["start"]
        end = data["end"]
        length = end - start
        sid = data["sid"]
        agent_name = data["agent_name"]

        pose_data = data["pose_data"]
        config_dir = data.get("config_dir", None)
        asset_name = pose_data["asset_name"]

        with open(os.path.join(self.data_root, "betas_69scale", f"{asset_name}.pkl"), "rb") as f:
            shapes = pickle.load(f)
            identity_coeffs = torch.from_numpy(shapes["identity_coeffs"]).float()
            scale_params = torch.from_numpy(shapes["scale_params"]).float()

        with open(os.path.join(self.data_root, "feats_v2", f"{sid}.pkl"), "rb") as f:
            f_imgseq = CPU_Unpickler(f).load()[agent_name]["pose_tokens"][start:end]
            # feat_data = pickle.load(f)
            # f_imgseq = feat_data[agent_name]["pose_tokens"][start:end]
            assert not torch.isnan(
                f_imgseq
            ).any(), f"f_imgseq is nan, {sid}, {torch.isnan(f_imgseq).sum()}"

        with open(os.path.join(self.data_root, "feats_det_v2", f"{sid}.pkl"), "rb") as f:
            noisy_data = pickle.load(f)
            noisy_f_imgseq = noisy_data[agent_name]["pose_tokens"][start:end]
            noisy_bbx_xys = torch.from_numpy(
                noisy_data[agent_name]["det_bbx_xys"][start:end]
            ).float()

        use_hb = False
        if self.mode == "train" and self.aug_hb:
            prob = 0.2
            if torch.rand(1) < prob:
                use_hb = True
        if use_hb:
            with open(os.path.join(self.data_root, "feats_halfbody", f"{sid}.pkl"), "rb") as f:
                hb_data = pickle.load(f)
                hb_f_imgseq = hb_data[agent_name]["pose_tokens"][start:end]
                hb_bbx_xys = torch.from_numpy(hb_data[agent_name]["bbx_xys"][start:end]).float()
                hb_transls = hb_data[agent_name]["transls"][start:end]

        with open(os.path.join(self.data_root, "vitpose77_v2", f"{sid}.pkl"), "rb") as f:
            vitposedata = CPU_Unpickler(f).load()
            # vitposedata = pickle.load(f)
            kp2d = vitposedata[agent_name]["vitpose"][start:end]
            gtkp2d = vitposedata[agent_name]["gt_joints_2d"][start:end, 1:]
            if gtkp2d.shape[-2] == 93:
                gtkp2d = gtkp2d[:, nvskel93to77_idx]
            assert gtkp2d.shape[-2] == 77, f"gtkp2d.shape: {gtkp2d.shape}"
            bbx_xys = torch.from_numpy(vitposedata[agent_name]["bbx_xys"][start:end]).float()

        if use_hb:
            with open(os.path.join(self.data_root, "vitpose77_halfbody", f"{sid}.pkl"), "rb") as f:
                hb_vitposedata = CPU_Unpickler(f).load()
                hb_kp2d = hb_vitposedata[agent_name]["vitpose"][start:end]
                hb_gtkp2d = hb_vitposedata[agent_name]["gt_joints_2d"][start:end, 1:]
                if hb_gtkp2d.shape[-2] == 93:
                    hb_gtkp2d = hb_gtkp2d[:, nvskel93to77_idx]
                assert hb_gtkp2d.shape[-2] == 77, f"hb_gtkp2d.shape: {hb_gtkp2d.shape}"
                hb_bbx_xys = torch.from_numpy(
                    hb_vitposedata[agent_name]["bbx_xys"][start:end]
                ).float()

        if use_hb:
            bbx_xys = hb_bbx_xys
            f_imgseq = hb_f_imgseq
            kp2d = hb_kp2d
            transl = hb_transls

        bbx_mask = torch.ones(bbx_xys.shape[0], dtype=torch.bool)
        gtkp2d = torch.from_numpy(gtkp2d).float()
        assert not torch.isnan(
            bbx_xys
        ).any(), f"bbx_xys is nan, {sid}, {torch.isnan(bbx_xys).sum()}"
        assert not torch.isinf(
            bbx_xys
        ).any(), f"bbx_xys is inf, {sid}, {torch.isinf(bbx_xys).sum()}"
        assert not torch.isnan(kp2d).any(), f"kp2d is nan, {sid}, {torch.isnan(kp2d).sum()}"
        assert not torch.isinf(kp2d).any(), f"kp2d is inf, {sid}, {torch.isinf(kp2d).sum()}"
        assert not torch.isnan(gtkp2d).any(), f"gtkp2d is nan, {sid}, {torch.isnan(gtkp2d).sum()}"
        assert not torch.isinf(gtkp2d).any(), f"gtkp2d is inf, {sid}, {torch.isinf(gtkp2d).sum()}"
        # gt_bbx_xys = get_bbx_xys(gtkp2d[:, nvskel77to33_idx, :], do_augment=False, base_enlarge=1.2)
        # detacted_bbx_xys = get_bbx_xys(kp2d, do_augment=False, base_enlarge=1.2)
        # kp2d_mask = kp2d[..., 2] > 0.5
        # assert kp2d.shape[-1] == 3, f"kp2d.shape: {kp2d.shape}"
        # det_bbx_mask = kp2d_mask[..., nvskel77to33_idx].sum(dim=-1) > 8
        # bbx_xys = gt_bbx_xys.clone()

        if (self.mode == "train") and self.aug_bbx:
            prob = 0.5
            if torch.rand(1) < prob:
                bbx_xys = noisy_bbx_xys
                f_imgseq = noisy_f_imgseq

        joints3d = pose_data["joints"][start:end][:, 1:]  # (F, 77, 3)
        if joints3d.shape[1] != 77:
            assert joints3d.shape[1] == 93, f"joints3d.shape[1]: {joints3d.shape[1]} != 77 or 93"
            joints3d = joints3d[:, nvskel93to77_idx, :]
        static_gt = get_static_joint_mask(
            torch.from_numpy(joints3d).float(), vel_thr=0.15, repeat_last=True
        )  # (F, J)
        static_gt = static_gt[:, contact_joint_ids].float()  # (F, J')

        # kp2d = pose_data["joints_2d"][start:end]
        # if kp2d.shape[-2] == 94:
        #     kp2d = kp2d[:, 1:, :]
        # kp2d_conf = np.ones_like(kp2d[..., 0])
        # kp2d = np.concatenate([kp2d, kp2d_conf[..., None]], axis=-1)
        # kp2d = torch.from_numpy(kp2d).float()
        assert (
            pose_data["pose"].shape[1] == 77
        ), f'pose_data["pose"].shape: {pose_data["pose"].shape}'

        body_pose = torch.from_numpy(pose_data["pose"])[start:end, 1:, :].float()
        global_orient = torch.from_numpy(pose_data["pose"])[start:end, 0, :].float()
        transl = torch.from_numpy(pose_data["transl"])[start:end].float()
        body_pose = body_pose.reshape(length, 76 * 3)

        global_orient_w = global_orient
        transl_w = transl

        camera_data = data["camera_data"]
        R_w2c = torch.from_numpy(camera_data["R_view"])[start:end].float()
        t_w2c = torch.from_numpy(camera_data["t_view"])[start:end].float()
        K_fullimg = torch.from_numpy(camera_data["K"]).float()

        T_w2c = torch.eye(4).repeat(length, 1, 1)
        T_w2c[:, :3, :3] = R_w2c
        T_w2c[:, :3, 3] = t_w2c

        normed_T_w2c = normalize_T_w2c(T_w2c)

        identity_coeffs = identity_coeffs.repeat(length, 1)
        scale_params = scale_params.repeat(length, 1)

        soma_params_w = {
            "body_pose": body_pose,
            "identity_coeffs": identity_coeffs,
            "scale_params": scale_params,
            "global_orient": global_orient_w,
            "transl": transl_w,
        }

        offset = torch.zeros(3)
        global_orient_c, transl_c = get_c_rootparam(
            soma_params_w["global_orient"],
            soma_params_w["transl"],
            T_w2c,
            offset,
        )

        soma_params_c = {
            "body_pose": body_pose,
            "identity_coeffs": identity_coeffs,
            "scale_params": scale_params,
            "global_orient": global_orient_c,
            "transl": transl_c,
        }

        # Data validation - check for problematic Z coordinates
        # Get 3D joints to check depth values
        min_z = transl_c[..., 2].min()
        vid = f"{sid}-{agent_name}"

        # Skip samples with depth issues that could cause numerical instability
        if min_z < 0.1:  # Too close to camera
            Log.info(f"[Metrosim] Skipping sample {vid} due to min_z={min_z:.3f} < 0.1")
            return None

        # Check for NaN/Inf in critical data
        if torch.isnan(transl_c).any() or torch.isinf(transl_c).any():
            Log.info(f"[Metrosim] Skipping sample {vid} due to NaN/Inf in transl_c")
            return None

        gravity_vec = torch.tensor([0, 0, -1], dtype=torch.float32)  # (3), Metrosim is az
        R_c2gv = get_R_c2gv(T_w2c[..., :3, :3], axis_gravity_in_w=gravity_vec)  # (F, 3, 3)

        R_anz2ay = axis_angle_to_matrix(torch.tensor([1.0, 0.0, 0.0]) * -torch.pi / 2)  # (3, 3)
        T_w2ay = transform_mat(R_anz2ay, R_anz2ay.new([0, 0, 0]))  # (4, 4)

        # Image
        K_fullimg = K_fullimg.repeat(length, 1, 1)  # (F, 3, 3)

        cam_angvel = compute_cam_angvel(
            normed_T_w2c[:, :3, :3]
        )  # (F, 6)  slightly different from WHAM
        cam_tvel = compute_cam_tvel(normed_T_w2c[:, :3, 3])  # (F, 3)

        max_len = self.max_motion_frames
        if bbx_mask.shape[0] < max_len:
            bbx_mask = torch.cat(
                [bbx_mask, torch.zeros(max_len - bbx_mask.shape[0], dtype=torch.bool)], dim=0
            )
        return_data = {
            "meta": {
                "data_name": self.dataset_name,
                "dataset_id": "metrosim",
                "idx": idx,
                # "T_w2c": normed_T_w2c,
                "vid": vid,
                "config_dir": config_dir,
                "frame_start": int(start),
                "frame_end": int(end),
                # "mode": self.mode,
            },
            "length": length,
            "soma_params_c": soma_params_c,
            "soma_params_w": soma_params_w,
            "R_c2gv": R_c2gv,  # (F, 3, 3)
            "gravity_vec": gravity_vec,  # (3)
            "bbx_xys": bbx_xys.float(),  # (F, 3)
            "K_fullimg": K_fullimg,  # (F, 3, 3)
            "f_imgseq": f_imgseq,  # (F, D)
            "kp2d": kp2d.float(),  # (F, 77, 3)
            # "gtkp2d": gtkp2d.float(),  # (F, 77, 2)
            "cam_angvel": cam_angvel,  # (F, 6)
            "cam_tvel": cam_tvel,  # (F, 3),
            "noisy_cam_tvel": cam_tvel,  # (F, 3),
            "T_w2c": normed_T_w2c,  # (F, 4, 4),
            "static_gt": static_gt,
            "mask": {
                "valid": get_valid_mask(max_len, length),
                "humanoid": get_valid_mask(max_len, 0),
                "has_img_mask": get_valid_mask(max_len, length) & bbx_mask,
                "has_2d_mask": get_valid_mask(max_len, length),
                "has_cam_mask": get_valid_mask(max_len, length),
                "has_audio_mask": get_valid_mask(max_len, 0),
                "has_music_mask": get_valid_mask(max_len, 0),
                "valid_contact_mask": get_valid_mask(max_len, length),
                "2d_only": False,
                "vitpose": True,
                "bbx_xys": bbx_mask,
                "f_imgseq": True,
                "spv_incam_only": False,
                "invalid_contact": False,
            },
        }
        if self.mode in ["val", "test"]:
            return_data["T_w2ay"] = (T_w2ay,)  # (4, 4)

        # Batchable
        return_data["soma_params_w"] = repeat_to_max_len_dict(return_data["soma_params_w"], max_len)
        return_data["soma_params_c"] = repeat_to_max_len_dict(return_data["soma_params_c"], max_len)
        return_data["R_c2gv"] = repeat_to_max_len(return_data["R_c2gv"], max_len)
        return_data["bbx_xys"] = repeat_to_max_len(return_data["bbx_xys"], max_len)
        return_data["K_fullimg"] = repeat_to_max_len(return_data["K_fullimg"], max_len)
        return_data["f_imgseq"] = repeat_to_max_len(return_data["f_imgseq"], max_len)
        return_data["kp2d"] = repeat_to_max_len(return_data["kp2d"], max_len)
        # return_data["gtkp2d"] = repeat_to_max_len(return_data["gtkp2d"], max_len)
        return_data["cam_angvel"] = repeat_to_max_len(return_data["cam_angvel"], max_len)
        return_data["cam_tvel"] = repeat_to_max_len(return_data["cam_tvel"], max_len)
        return_data["noisy_cam_tvel"] = repeat_to_max_len(return_data["noisy_cam_tvel"], max_len)
        return_data["T_w2c"] = repeat_to_max_len(return_data["T_w2c"], max_len)
        return_data["static_gt"] = pad_to_max_len(return_data["static_gt"], max_len)

        return return_data
