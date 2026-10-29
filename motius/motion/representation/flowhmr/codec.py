"""FlowHMR's 491-D motion representation and SMPL-H reconstruction."""
import json
from pathlib import Path
from typing import Dict
import torch
from torch import Tensor
from motius.motion.body_models.flowhmr import SMPLMesh, SMPLSkeleton
from .geometry import rot6d_to_rotation_matrix, rotation_matrix_to_rot6d
from .motion import _global_rots_to_local_rots, _SMPLH_PARENTS


class FlowHMRMotionCodec(torch.nn.Module):
    COMPONENT_DIMS = {
            "smooth_root_vel_xz": 2,
            "smooth_root_pos_y": 1,
            "local_joints_positions": 52 * 3,   # 156
            "rotation_data": 52 * 6,             # 312
            "shapes": 16,
            "foot_contacts": 4,
        }
    LATENT_DIM = sum(COMPONENT_DIMS.values())
    DEFAULT_LOSS_WEIGHTS = {
            "smooth_root_vel_xz": 10.0,
            "smooth_root_pos_y": 10.0,
            "local_joints_positions": 10.0,
            "rotation_data": 10.0,
            "shapes": 1.0,
            "foot_contacts": 4.0,
            "fk_rotation": 5.0,
            "fk_consistency": 5.0,
        }

    def __init__(self, *, body_model_path, mean_std_path, motion_style="basic", fps=30.0,
                 statistics=None):
        super().__init__()
        if motion_style not in ("basic", "global", "mixed"):
            raise ValueError(f"Unsupported FlowHMR motion style: {motion_style}")
        self.motion_style = motion_style
        self.fps = fps
        self.latent_dim = self.LATENT_DIM
        self._component_split_sizes = list(self.COMPONENT_DIMS.values())
        self._component_names = list(self.COMPONENT_DIMS)
        self.body_model = SMPLSkeleton(model_path=str(body_model_path))
        self.mesh_model = SMPLMesh(model_path=str(body_model_path))
        self.load_mean_std(mean_std_path)
        if statistics:
            for name, value in statistics.items():
                if name not in self._buffers:
                    raise ValueError(f"Unknown FlowHMR statistic: {name}")
                if value.shape != self._buffers[name].shape:
                    raise ValueError(f"Incompatible FlowHMR statistic: {name}")
                self._buffers[name] = value.detach().clone()

    @classmethod
    def from_bundle(cls, bundle, body_model_path=None):
        args = bundle.config["train_pipeline_args"]
        body_model_path = body_model_path or bundle.body_model_path
        if not body_model_path:
            raise ValueError("FlowHMR pipelines/trainers require an explicit SMPL-H body_model_path")
        parameter = next(bundle.parameters())
        return cls(body_model_path=body_model_path, mean_std_path=bundle.mean_std_path,
                   motion_style=args.get("motion_style", "basic"), fps=args.get("fps", 30.0),
                   statistics=bundle.motion_statistics).to(device=parameter.device, dtype=parameter.dtype)

    def synchronize_statistics(self, bundle):
        """Use the current checkpoint buffers, including after a runner resume."""
        parameter = next(bundle.parameters())
        self.to(device=parameter.device, dtype=parameter.dtype)
        for name, value in bundle.motion_statistics.items():
            getattr(self, name).copy_(value)

    def load_mean_std(self, mean_std_path: str) -> None:
        with open(mean_std_path, "r") as f:
            stats = json.load(f)

        def _register_pair(name: str, key: str, shape_suffix: tuple = ()):
            mean = torch.FloatTensor(stats[key]["mean"])
            std = torch.FloatTensor(stats[key]["std"])
            if shape_suffix:
                mean = mean.reshape(*shape_suffix)
                std = std.reshape(*shape_suffix)
            std = std.clamp(min=1e-5)
            self.register_buffer(f"{name}_mean", mean[None, None])  # (1,1,...)
            self.register_buffer(f"{name}_std", std[None, None])

        _register_pair("smooth_root_vel", "smooth_root_vel")                         # (1,1,3)
        _register_pair("smooth_root_pos", "smooth_root_pos")                         # (1,1,3)
        _register_pair("local_joints_positions", "local_joints_positions", (52, 3))  # (1,1,52,3)
        _register_pair("global_rot_data", "global_rot_data", (52, 6))               # (1,1,52,6)
        _register_pair("local_rot_data", "local_rot_data", (52, 6))                 # (1,1,52,6)
        _register_pair("rep_shapes", "shapes")                                    # (1,1,16)
        _register_pair("rep_foot_contacts", "foot_contacts")                      # (1,1,4)

    def _ensure_motion_format(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        if "smooth_root_vel" in batch:
            return batch
        from .motion import _encode_motion_base_style
        return _encode_motion_base_style(batch, self.body_model, fps=self.fps, n_joints_pos=52)

    def encode_motion(self, batch: Dict[str, Tensor]) -> Tensor:
        batch = self._ensure_motion_format(batch)

        # 1. Root velocity XZ (2D)
        srv = batch["smooth_root_vel"]                         # (B,T,3)
        srv_xz = srv[..., [0, 2]]
        srv_xz_norm = (srv_xz - self.smooth_root_vel_mean[..., [0, 2]]) / \
                       self.smooth_root_vel_std[..., [0, 2]]

        # 2. Root position Y (1D)
        srp_y = batch["smooth_root_pos"][..., 1:2]            # (B,T,1)
        srp_y_norm = (srp_y - self.smooth_root_pos_mean[..., 1:2]) / \
                      self.smooth_root_pos_std[..., 1:2]

        # 3. Local joint positions (52×3=156)
        ljp = batch["local_joints_positions"]                  # (B,T,52,3)
        ljp_norm = (ljp - self.local_joints_positions_mean) / self.local_joints_positions_std
        B, T = ljp_norm.shape[:2]
        ljp_flat = ljp_norm.reshape(B, T, -1)

        if self.motion_style == "basic":
            rot = batch["local_rot_data"]                      # (B,T,52,6)
            rot_norm = (rot - self.local_rot_data_mean) / self.local_rot_data_std
        elif self.motion_style == "global":
            rot = batch["global_rot_data"]                     # (B,T,52,6)
            rot_norm = (rot - self.global_rot_data_mean) / self.global_rot_data_std
        elif self.motion_style == "mixed":
            rot_body = batch["global_rot_data"][:, :, :22]     # (B,T,22,6)
            rot_hand = batch["local_rot_data"][:, :, 22:]      # (B,T,30,6)
            rot_body_norm = (rot_body - self.global_rot_data_mean[:, :, :22]) / \
                             self.global_rot_data_std[:, :, :22]
            rot_hand_norm = (rot_hand - self.local_rot_data_mean[:, :, 22:]) / \
                             self.local_rot_data_std[:, :, 22:]
            rot_norm = torch.cat([rot_body_norm, rot_hand_norm], dim=2)
        rot_flat = rot_norm.reshape(B, T, -1)

        # 5. Shapes (16)
        shapes_norm = (batch["shapes"] - self.rep_shapes_mean) / self.rep_shapes_std

        # 6. Foot contacts (4)
        fc_norm = (batch["foot_contacts"] - self.rep_foot_contacts_mean) / \
                   self.rep_foot_contacts_std

        # 7. Concat: 2+1+156+312+16+4 = 491
        return torch.cat([srv_xz_norm, srp_y_norm, ljp_flat, rot_flat, shapes_norm, fc_norm], dim=-1)

    def _decode_latent_to_motion(self, latent: Tensor, fps: float = None) -> Dict[str, Tensor]:
        fps = fps or self.fps
        B, T, D = latent.shape

        # 1. Split by component
        parts = torch.split(latent, self._component_split_sizes, dim=-1)
        srv_xz_norm = parts[0]
        srp_y_norm = parts[1]
        ljp_flat = parts[2]
        rot_flat = parts[3]
        shapes_norm = parts[4]
        fc_norm = parts[5]

        # 2. Denormalize root
        srv_xz = srv_xz_norm * self.smooth_root_vel_std[..., [0, 2]] + \
                  self.smooth_root_vel_mean[..., [0, 2]]
        srp_y = srp_y_norm * self.smooth_root_pos_std[..., 1:2] + \
                 self.smooth_root_pos_mean[..., 1:2]

        # 3. Denormalize joint positions
        ljp = ljp_flat.reshape(B, T, 52, 3)
        ljp = ljp * self.local_joints_positions_std + self.local_joints_positions_mean

        # 4. Denormalize rotations → recover local SMPL rotations
        rot_data = rot_flat.reshape(B, T, 52, 6)
        device = latent.device

        if self.motion_style == "basic":
            local_rot_data = rot_data * self.local_rot_data_std + self.local_rot_data_mean
            root_rot6d = local_rot_data[:, :, 0]
            body_rot6d = local_rot_data[:, :, 1:]

        elif self.motion_style == "global":
            global_rot_data = rot_data * self.global_rot_data_std + self.global_rot_data_mean
            # global → local rotations
            global_rot_mats = rot6d_to_rotation_matrix(
                global_rot_data.reshape(B * T * 52, 6)
            ).reshape(B, T, 52, 3, 3)
            local_rot_mats = _global_rots_to_local_rots(
                global_rot_mats, _SMPLH_PARENTS.to(device)
            )
            root_rot6d = rotation_matrix_to_rot6d(local_rot_mats[:, :, 0])
            body_rot6d = rotation_matrix_to_rot6d(local_rot_mats[:, :, 1:])

        elif self.motion_style == "mixed":
            # body(0:22) global, hand(22:52) local
            rot_body = rot_data[:, :, :22]
            rot_hand = rot_data[:, :, 22:]
            rot_body = rot_body * self.global_rot_data_std[:, :, :22] + \
                        self.global_rot_data_mean[:, :, :22]
            rot_hand = rot_hand * self.local_rot_data_std[:, :, 22:] + \
                        self.local_rot_data_mean[:, :, 22:]

            # body global → local
            global_body_mats = rot6d_to_rotation_matrix(
                rot_body.reshape(B * T * 22, 6)
            ).reshape(B, T, 22, 3, 3)
            local_body_mats = _global_rots_to_local_rots(
                global_body_mats, _SMPLH_PARENTS[:22].to(device)
            )
            root_rot6d = rotation_matrix_to_rot6d(local_body_mats[:, :, 0])
            body_body_rot6d = rotation_matrix_to_rot6d(local_body_mats[:, :, 1:])
            # hand already local
            body_rot6d = torch.cat([body_body_rot6d, rot_hand], dim=2)

        shapes = shapes_norm * self.rep_shapes_std + self.rep_shapes_mean
        shapes = shapes.mean(dim=1, keepdim=True)

        # 6. Foot contacts
        foot_contacts = fc_norm * self.rep_foot_contacts_std + self.rep_foot_contacts_mean

        # 7. Recover smooth_root_pos from velocity
        smooth_root_pos = torch.zeros(B, T, 3, device=device, dtype=latent.dtype)
        if T > 1:
            smooth_root_pos[:, 1:, 0] = torch.cumsum(srv_xz[:, 1:, 0] / fps, dim=1)
            smooth_root_pos[:, 1:, 2] = torch.cumsum(srv_xz[:, 1:, 1] / fps, dim=1)
        smooth_root_pos[..., 1] = srp_y.squeeze(-1)

        # 8. Recover pelvis world position from local_joints_positions + smooth_root
        pelvis_local = ljp[:, :, 0, :]
        pelvis_world = pelvis_local.clone()
        pelvis_world[..., 0] += smooth_root_pos[..., 0]
        pelvis_world[..., 2] += smooth_root_pos[..., 2]
        # Y is absolute (no adjustment needed)

        # 9. Full rot6d stack
        rot6d = torch.cat([root_rot6d[:, :, None], body_rot6d], dim=2)

        return {
            "root_rot6d": root_rot6d,
            "body_rot6d": body_rot6d,
            "rot6d": rot6d,
            "local_joints_positions": ljp,
            "foot_contacts": foot_contacts,
            "smooth_root_pos": smooth_root_pos,
            "shapes": shapes,
            "pelvis_world": pelvis_world,
        }

    def _motion_to_smplh(self, motion_fields: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pelvis_world = motion_fields["pelvis_world"]    # (B, T, 3)
        shapes = motion_fields["shapes"]                # (B, 1, 16)
        rot6d = motion_fields["rot6d"]                  # (B, T, 52, 6)

        j_shaped = self.body_model.compute_j_shaped(shapes.squeeze(1))
        pelvis_rest = j_shaped[:, 0:1, :]

        trans = pelvis_world - pelvis_rest

        return {
            "rot6d": rot6d,
            "shapes": shapes,
            "trans": trans,
        }

    def decode_motion(self, latent: Tensor, fps: float = None, **kwargs) -> Dict[str, Tensor]:
        motion_fields = self._decode_latent_to_motion(latent, fps=fps)

        smplh_params = self._motion_to_smplh(motion_fields)

        return {
            "rot6d": smplh_params["rot6d"],
            "shapes": smplh_params["shapes"],
            "trans": smplh_params["trans"],
            "root_rot6d": motion_fields["root_rot6d"],
            "body_rot6d": motion_fields["body_rot6d"],
            "local_joints_positions": motion_fields["local_joints_positions"],
            "foot_contacts": motion_fields["foot_contacts"],
            "smooth_root_pos": motion_fields["smooth_root_pos"],
            "pelvis_world": motion_fields["pelvis_world"],
        }
