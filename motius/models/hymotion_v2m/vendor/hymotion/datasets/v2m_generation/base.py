from __future__ import annotations
import os
import shutil
from typing import Union
from glob import glob
import random
import time
import numpy as np
import cv2
import torch
from tqdm import tqdm
from ..geometry import angle_axis_to_rotation_matrix, rot6d_to_rotation_matrix, rotation_matrix_to_rot6d
from .geometry import compute_cam_angvel, get_R_c2gv, process_r_t
from ...evaluation.metrics import get_joints_from_smpl_params
from ...bodymodels.smpl_skeleton import SMPLSkeleton, SMPLX2SMPL


# =============================================================================
# Data Loading Utilities
# =============================================================================


def get_local_transl_vel(transl, global_orient_R, fps):
    """
    Compute translation velocity in local coordinate (SMPL-coord).

    Args:
        transl: (*, L, 3)
        global_orient_R: (*, L, 3, 3)
        fps: frames per second
    Returns:
        transl_vel: (*, L, 3)
    """
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]
    transl_vel = torch.cat([transl_vel, transl_vel[..., -1:, :]], dim=-2)  # last-padding
    transl_vel = transl_vel * fps

    # v_local = R^T @ v_global
    local_transl_vel = torch.einsum("...lij,...li->...lj", global_orient_R, transl_vel)
    return local_transl_vel


class MotionDataLoader:
    """Handles loading motion data from npz files."""

    @staticmethod
    def load_raw_motion(filename, start_frame, end_frame):
        """Load raw motion data and convert to rot6d format."""
        data = dict(np.load(filename))

        for key in ["poses", "trans", "betas"]:
            data[key] = torch.FloatTensor(data[key])
            if key != "betas":
                data[key] = data[key][start_frame:end_frame].clone()

        poses = data["poses"]
        if len(poses.shape) == 2:
            poses = poses.reshape(poses.shape[0], -1, 3)

        rotations = angle_axis_to_rotation_matrix(poses)
        rot6d = rotation_matrix_to_rot6d(rotations)
        transl = data["trans"]
        root_rotations = angle_axis_to_rotation_matrix(poses[:, 0])
        transl_vel = get_local_transl_vel(transl, root_rotations, fps=30)

        return {
            "rot6d": rot6d,
            "trans": transl,
            "trans_vel": transl_vel,
            "shapes": data["betas"][0][None].repeat(rot6d.shape[0], 1),
        }

    @staticmethod
    def load_rotation_translation(data_or_filename, start_frame, end_frame):
        """Load motion as separate root and body rotations."""
        if isinstance(data_or_filename, str):
            data = dict(np.load(data_or_filename))
        else:
            data = data_or_filename

        for key in ["poses", "trans", "betas"]:
            data[key] = torch.FloatTensor(data[key])
            if key != "betas":
                data[key] = data[key][start_frame:end_frame].clone()

        poses = data["poses"]
        if len(poses.shape) == 2:
            poses = poses.reshape(poses.shape[0], -1, 3)

        rotations = angle_axis_to_rotation_matrix(poses)
        transl = data["trans"]
        root_rotations = rotations[:, 0]
        body_rotations = rotations[:, 1:]
        transl_vel = get_local_transl_vel(transl, root_rotations, fps=30)

        return {
            "root_rotations": root_rotations,
            "body_rotations": body_rotations,
            "transl": transl,
            "transl_vel": transl_vel,
            "shapes": data["betas"][0][None].repeat(poses.shape[0], 1),
        }

    @staticmethod
    def load_joints_vertices(filename, start_frame=0, end_frame=None):
        """Load joints and vertices for evaluation datasets (RICH, EMDB, 3DPW)."""
        filename = glob(os.path.join(os.path.dirname(filename), "*_vertices.npz"))[0]
        assert os.path.exists(filename), f"filename {filename} does not exist"

        data = dict(np.load(filename))

        # Handle different key naming conventions
        key_mapping = {
            "global_orient": "poses_root",
            "body_pose": "poses_body",
            "transl": "trans",
        }
        for new_key, old_key in key_mapping.items():
            if new_key not in data and old_key in data:
                data[new_key] = data.pop(old_key)

        for key in ["global_orient", "body_pose", "transl", "joints", "vertices"]:
            data[key] = torch.FloatTensor(data[key])[start_frame:end_frame].clone()

        data["global_orient"] = data["global_orient"].reshape(data["global_orient"].shape[0], -1, 3)
        data["body_pose"] = data["body_pose"].reshape(data["body_pose"].shape[0], -1, 3)

        if data["body_pose"].shape[-2] == 23:
            data["body_pose"] = data["body_pose"][:, :21]

        # Add zero hand pose
        hand_pose = torch.zeros(data["body_pose"].shape[0], 30, 3)
        poses = torch.cat([data["global_orient"], data["body_pose"], hand_pose], dim=1)

        if len(poses.shape) == 2:
            poses = poses.reshape(poses.shape[0], -1, 3)

        rotations = angle_axis_to_rotation_matrix(poses)
        transl = data["transl"].reshape(data["transl"].shape[0], 3)
        root_rotations = rotations[:, 0]
        body_rotations = rotations[:, 1:]
        transl_vel = get_local_transl_vel(transl, root_rotations, fps=30)
        shapes = torch.zeros(poses.shape[0], 16)

        return {
            "root_rotations": root_rotations,
            "body_rotations": body_rotations,
            "transl": transl,
            "transl_vel": transl_vel,
            "shapes": shapes,
            "joints": data["joints"],
            "vertices": data["vertices"],
        }


class CameraDataLoader:
    """Handles loading camera data from npz files."""

    @staticmethod
    def load_camera(filename=None, data=None, start_frame=0, end_frame=None):
        """Load camera extrinsics and intrinsics."""
        if data is None and filename is not None:
            data = dict(np.load(filename))
        RT = torch.FloatTensor(data["RT"])

        # Handle both [N, 4, 4] and [N, 3, 4] formats
        if RT.shape[1] == 3:
            n = RT.shape[0]
            RT_homo = torch.zeros((n, 4, 4), dtype=RT.dtype)
            RT_homo[:, :3, :4] = RT
            RT_homo[:, 3, 3] = 1
            RT = RT_homo

        return {
            "RT": RT[start_frame:end_frame].clone(),
            "K": torch.FloatTensor(data["K"])[start_frame:end_frame].clone(),
            "movement_type": data.get("movement_type", "unset"),
        }


# =============================================================================
# Dataset Path Parsing
# =============================================================================


class DatasetPathParser:
    """Handles parsing and resolving dataset file paths."""

    def __init__(self, root_config, split="val"):
        self.root_config = root_config
        self.split = split

    def get_dirnames(self):
        """Get list of directory names from root configuration."""
        root = self.root_config
        dirnames = []
        already_sampled = False
        need_check = True

        if isinstance(root["root"], list):
            dirnames.extend(root["root"])
        elif os.path.isdir(root["root"]):
            dirnames = self._get_dirnames_from_folder(root)
        elif os.path.exists(root["root"]) and root["root"].endswith(".txt"):
            dirnames, already_sampled = self._get_dirnames_from_txt(root)
            need_check = False
        else:
            raise ValueError(f"The given dataset root {root} does not exist.")

        return dirnames, already_sampled, need_check

    def _get_dirnames_from_folder(self, root):
        """Get directory names from folder structure."""
        dirnames = []
        for folder in root["folders"]:
            folder_path = os.path.join(root["root"], folder)
            dirnames.extend([os.path.join(folder_path, x) for x in sorted(os.listdir(folder_path))])
        return dirnames

    def _get_dirnames_from_txt(self, root):
        """Get directory names from text file."""
        with open(root["root"], "r", encoding="utf-8") as f:
            lines = f.readlines()
        dirnames = [item.strip() for item in lines]

        already_sampled = False
        sample_step = root.get("sample_step", 1)

        if sample_step > 1:
            dirnames = [d for i, d in enumerate(dirnames) if i % sample_step == 0]
            already_sampled = True
        elif sample_step < 0:
            step = -sample_step
            dirnames = [d for i, d in enumerate(dirnames) if i % step != 0]
            already_sampled = True

        return dirnames, already_sampled

    def resolve_data_item(self, dirname_or_dict, need_check=True):
        """Resolve a single data item from dirname or dict."""
        if need_check and isinstance(dirname_or_dict, str) and not os.path.isdir(dirname_or_dict):
            return None

        if isinstance(dirname_or_dict, str):
            dirname = dirname_or_dict
        elif isinstance(dirname_or_dict, dict):
            dirname = dirname_or_dict["name"]
        else:
            raise NotImplementedError(f"Invalid dirname_or_dict: {dirname_or_dict}")

        root = self.root_config
        basename = "_".join(dirname.split("/")[-1].split("_")[:-1])
        last_dirname = os.path.basename(os.path.dirname(dirname))

        # Resolve motion name
        motion_name = self._resolve_motion_name(dirname, basename, last_dirname)

        # Resolve camera name
        camera_name = self._resolve_camera_name(dirname, last_dirname, need_check)
        if camera_name is None:
            return None

        # Resolve feature name
        feature_name = self._resolve_feature_name(dirname, last_dirname, need_check)
        if feature_name is None:
            return None

        # Resolve video name
        video_name = self._resolve_video_name(dirname, last_dirname)

        data_item = {
            "sequence_name": dirname.split("/")[-1],
            "motion_name": motion_name,
            "camera_name": camera_name,
            "feature_name": feature_name,
            "video_name": video_name,
        }

        # Add frame info if available
        if isinstance(dirname_or_dict, dict) and "start_frame" in dirname_or_dict:
            data_item["start_frame"] = int(dirname_or_dict["start_frame"])
            data_item["end_frame"] = int(dirname_or_dict["end_frame"])

        # 兼容一下keyframe模式
        if isinstance(dirname_or_dict, dict) and "keyframes" in dirname_or_dict:
            data_item["keyframes"] = dirname_or_dict["keyframes"]
        # 兼容interval drop模式
        if isinstance(dirname_or_dict, dict) and "drop_interval" in dirname_or_dict:
            data_item["drop_interval"] = dirname_or_dict["drop_interval"]

        if "do_post_ik" in dirname_or_dict:
            data_item["do_post_ik"] = dirname_or_dict["do_post_ik"]
        else:
            data_item["do_post_ik"] = True

        if "do_camera_fitting" in dirname_or_dict:
            data_item["do_camera_fitting"] = dirname_or_dict["do_camera_fitting"]
        else:
            data_item["do_camera_fitting"] = False

        # Add bbox if configured
        bbox_name = self._resolve_bbox_name(dirname, last_dirname, need_check)
        if bbox_name is not None:
            data_item["bbox_name"] = bbox_name
        elif "bbox_format" in root:
            return None  # Required but not found

        return data_item

    def _resolve_motion_name(self, dirname, basename, last_dirname):
        """Resolve motion file name."""
        motion_name = os.path.join(dirname, basename + ".npz")
        if self.split == "eval" and not os.path.exists(motion_name) and "EMDB" in dirname:
            motion_name = os.path.join(dirname, f"{last_dirname}_{basename}.npz")
        return motion_name

    def _resolve_camera_name(self, dirname, last_dirname, need_check):
        """Resolve camera file name."""
        root = self.root_config
        camera_format = root.get("camera_format", "{name}_camera.npz")
        camera_name = camera_format.format(
            name=dirname.split("/")[-1],
            last_dirname=last_dirname,
        )
        camera_name = os.path.join(dirname, camera_name)

        if need_check and not os.path.exists(camera_name):
            camera_name = os.path.join(dirname, dirname.split("/")[-1] + "_trimmed_camera.npz")
            if not os.path.exists(camera_name):
                raise AssertionError(f"camera_name {camera_name} does not exist")

        return camera_name

    def _resolve_feature_name(self, dirname, last_dirname, need_check):
        """Resolve feature file name."""
        root = self.root_config
        feature_format = root.get("feature_format", "{name}_sam3d_feat_v2.pt")
        feature_name = feature_format.format(
            name=dirname.split("/")[-1],
            last_dirname=last_dirname,
        )
        feature_name = os.path.join(dirname, feature_name)

        if need_check and not os.path.exists(feature_name):
            return None
        return feature_name

    def _resolve_video_name(self, dirname, last_dirname):
        """Resolve video file name."""
        root = self.root_config
        video_format = root.get("video_format", "{name}.mp4")
        video_name = video_format.format(
            name=dirname.split("/")[-1],
            last_dirname=last_dirname,
        )
        return os.path.join(dirname, video_name)

    def _resolve_bbox_name(self, dirname, last_dirname, need_check):
        """Resolve bbox file name."""
        root = self.root_config
        if "bbox_format" not in root:
            return None

        bbox_name = os.path.join(
            dirname, root["bbox_format"].format(name=dirname.split("/")[-1], last_dirname=last_dirname)
        )

        if need_check and not os.path.exists(bbox_name):
            print(f"bbox_name {bbox_name} does not exist")
            return None

        return bbox_name


# =============================================================================
# Base Dataset Classes
# =============================================================================


class BaseDatasetMixin:
    """Mixin providing common dataset utilities."""

    def _load_filenames_from_root(self, root, split):
        """Load filenames from a single root configuration."""
        parser = DatasetPathParser(root, split)
        dirnames, already_sampled, need_check = parser.get_dirnames()

        print(f"[{self.__class__.__name__}] {len(dirnames)} dirs fetched from {root['root']}")

        filenames = []
        for index, dirname_or_dict in enumerate(dirnames):
            if index % 10000 == 0:
                print(f"[{self.__class__.__name__}] {index} / {len(dirnames)} dirs processed")

            data_item = parser.resolve_data_item(dirname_or_dict, need_check)
            if data_item is not None:
                filenames.append(data_item)

        # Mark if already sampled to avoid double sampling
        if already_sampled:
            for item in filenames:
                item["_already_sampled"] = True

        return filenames

    def _apply_sampling(self, filenames, root):
        """Apply sampling step to filenames."""
        if filenames and filenames[0].get("_already_sampled"):
            return filenames

        sample_step = root.get("sample_step", 1)
        if sample_step > 1:
            return [f for i, f in enumerate(filenames) if i % sample_step == 0]
        elif sample_step < 0:
            step = -sample_step
            return [f for i, f in enumerate(filenames) if i % step != 0]
        return filenames

    def _load_all_filenames(self, roots, split):
        """Load all filenames from root configurations."""
        filenames = []

        for root in tqdm(roots, desc=f"fetching {split} data...", disable=True):
            filenames_tmp = self._load_filenames_from_root(root, split)
            filenames_tmp = self._apply_sampling(filenames_tmp, root)
            filenames.extend(filenames_tmp)
            print(f"[{self.__class__.__name__}] {len(filenames)} files loaded from {root['root']}")

        return filenames

    # get item
    def _get_frame_range(self, data, max_len, total_frames=-1, use_random_start=False, min_len=60):
        """Get start/end frame range from data item."""
        if "start_frame" in data:
            start_frame = data["start_frame"]
            end_frame = data["end_frame"]
            pad_clip_max = end_frame - start_frame
        else:
            # print(f"total_frames: {total_frames}, max_len: {max_len}, use_random_start: {use_random_start}")
            if use_random_start and total_frames > min_len:  # 如果数据的帧数小于min_len，则不进行随机采样
                start_frame = np.random.randint(0, total_frames - min_len)
                # 末尾不随机了，仅在开头进行随机，体现对不同的开始的朝向的兼容
                end_frame = min(total_frames, start_frame + max_len)
                pad_clip_max = max_len
            else:
                start_frame = 0
                end_frame = None if max_len == -1 else max_len
                pad_clip_max = end_frame
            # print(f"start_frame: {start_frame}, end_frame: {end_frame}, pad_clip_max: {pad_clip_max}")
        return start_frame, end_frame, pad_clip_max

    def _load_motion_for_eval(self, filename, start_frame=0, end_frame=None):
        motion = MotionDataLoader.load_joints_vertices(filename, start_frame, end_frame)
        # Convert smplx to smpl(h) if needed
        if motion["vertices"].shape[-2] != 6890:
            assert hasattr(self, "smplx2smpl"), "smplx2smpl must be initialized"
            motion["joints"] = self.smplx2smpl.smplx2smpl_joints(motion["joints"])
            motion["vertices"] = self.smplx2smpl.smplx2smpl_verts(motion["vertices"])
        return motion

    def _load_motion_from_data(self, data, start_frame=0, end_frame=None):
        return MotionDataLoader.load_rotation_translation(data, start_frame, end_frame)

    def _data_sanity_check(self, inputs, target, inputs_keys=None, target_keys=None):
        """Verify data consistency."""
        if inputs_keys is None:
            inputs_keys = ["feature", "camera_R", "camera_T", "bbox_info"]
        if target_keys is None:
            target_keys = ["root_rot6d", "body_rot6d", "transl_vel", "trans", "shapes", "end_effector_vel"]

        shape_0 = []
        for k in inputs_keys:
            val = inputs[k]
            shape_0.append(val.shape[0] if isinstance(val, torch.Tensor) else val)

        for k in target_keys:
            val = target[k]
            shape_0.append(val.shape[0] if isinstance(val, torch.Tensor) else val)

        assert all(s == shape_0[0] for s in shape_0), "Length inconsistency in inputs and target data"

    @staticmethod
    def padding_or_clip(data, max_len, round_frames, keys=None):
        """Pad or clip data to specified length."""
        if keys is None:
            keys = ["rot6d", "trans", "shapes", "trans_vel"]

        current_length = max_len
        for key in keys:
            if key not in data:
                continue

            length = data[key].shape[0]
            length = length // round_frames * round_frames

            if max_len is None:
                current_length = length
            elif length > max_len:
                data[key] = data[key][:max_len]
            else:
                current_length = length
                padding = torch.zeros(max_len - length, *data[key].shape[1:]) + data[key][-1:]
                data[key] = torch.cat([data[key][:length], padding], dim=0)

        data["length"] = current_length
        return data


class FeatureLoaderMixin:
    """Mixin for loading features and bounding boxes."""

    def load_feature(self, feature_name_or_feature: Union[str, torch.Tensor], start_frame, end_frame, load_hand=False):
        """Load visual features from file."""
        if isinstance(feature_name_or_feature, str):
            data = torch.load(feature_name_or_feature, map_location="cpu", weights_only=True)
        else:
            data = feature_name_or_feature
        data = data[start_frame:end_frame]

        # Extract body features (first 1024 dims)
        if not load_hand:
            feature = data[:, :1024]
        else:
            feature = data

        # Handle NaN values: replace NaN with 0, keep original length
        feature_length = feature.shape[0]
        feature[feature.isnan()] = 0

        return {"feature": feature, "length": feature_length}

    def load_bbox(self, bbox_name_or_bbox: Union[str, torch.Tensor], start_frame, end_frame):
        """Load bounding box data."""
        if isinstance(bbox_name_or_bbox, str):
            data = dict(np.load(bbox_name_or_bbox))
        else:
            data = bbox_name_or_bbox
        bbox = torch.FloatTensor(data["bbox"][start_frame:end_frame])[..., :4]

        kp2d = (
            torch.FloatTensor(data["kp2d"][start_frame:end_frame])
            if "kp2d" in data
            else torch.zeros(bbox.shape[0], 52, 2)
        )

        item = {
            "bbox": bbox,
            "keypoints2d": kp2d,
            "start_frame": int(data["start_end"][0]) - start_frame,
            "end_frame": int(data["start_end"][1]) - start_frame,
        }
        item["bbox_center"] = (item["bbox"][:, :2] + item["bbox"][:, 2:4]) / 2
        item["bbox_scale"] = (item["bbox"][:, 2:4] - item["bbox"][:, :2]).max(dim=-1, keepdim=True).values
        return item

    def bbox_info_from_bbox(self, bbox, K):
        # Normalize bbox center
        bbox_center_pixel = torch.cat([bbox["bbox_center"], torch.ones_like(bbox["bbox_center"][:, :1])], dim=-1)
        bbox_center_pixel = torch.inverse(K) @ bbox_center_pixel.reshape(-1, 3, 1)
        bbox_center_pixel = bbox_center_pixel.squeeze(-1)[:, :2]
        bbox_scale_pixel = bbox["bbox_scale"] * 2 / (K[:, 0, 0] + K[:, 1, 1]).unsqueeze(-1)
        return torch.cat([bbox_center_pixel, bbox_scale_pixel], dim=-1)

    def load_sapiens_bbox(self, bbox_name, start_frame, end_frame):
        """Load Sapiens format bounding box data."""
        COCO17_IN_BODY25 = [0, 16, 15, 18, 17, 5, 2, 6, 3, 7, 4, 12, 9, 13, 10, 14, 11]

        def coco17tobody25(points2d):
            kpts = np.zeros((points2d.shape[0], 25, 3))
            kpts[:, COCO17_IN_BODY25, :2] = points2d[:, :, :2]
            kpts[:, COCO17_IN_BODY25, 2:3] = points2d[:, :, 2:3]
            kpts[:, 8, :2] = kpts[:, [9, 12], :2].mean(axis=1)
            kpts[:, 8, 2] = kpts[:, [9, 12], 2].min(axis=1)
            kpts[:, 1, :2] = kpts[:, [2, 5], :2].mean(axis=1)
            kpts[:, 1, 2] = kpts[:, [2, 5], 2].min(axis=1)
            return kpts

        def coco23tobody25(points2d):
            kpts = coco17tobody25(points2d[:, :17, :])
            kpts[:, [19, 20, 21, 22, 23, 24], :] = points2d[:, 17:23, :]
            return kpts

        def coco133tobody25(points2d):
            kpts = coco23tobody25(points2d[:, :23, :])
            return kpts

        data = dict(np.load(bbox_name))
        bbox = torch.FloatTensor(data["bbox"][start_frame:end_frame])

        if "keypoints3d" in data:
            k2d = data["keypoints3d"]
            if len(k2d.shape) == 2:
                k2d = k2d.reshape(-1, 133, 3)
            k2d = k2d[start_frame:end_frame]
            keypoints2d_body25 = coco133tobody25(k2d)
            keypoints3d = torch.FloatTensor(keypoints2d_body25)
            bbox = bbox[: keypoints3d.shape[0]]
        else:
            keypoints3d = torch.zeros(bbox.shape[0], 52, 3)

        item = {
            "bbox": bbox,
            "keypoints2d": keypoints3d,
            "start_frame": 0,
            "end_frame": bbox.shape[0],
        }
        item["bbox_center"] = (item["bbox"][:, :2] + item["bbox"][:, 2:4]) / 2
        item["bbox_scale"] = (item["bbox"][:, 2:4] - item["bbox"][:, :2]).max(dim=-1, keepdim=True).values
        return item


class AugmentationMixin:
    """Mixin for data augmentation."""

    # These attributes are expected to be defined in the class that uses this mixin
    augmentation_type: str
    augment_n: int
    split: str

    def augment_input_features(self, orig_inputs):
        """Apply augmentation to input features."""
        # 训练时期直接进行增强
        if self.split == "train" and self.augmentation_type == "original":
            return self._augment_original_mode(orig_inputs)

        if self.split == "train" and self.augmentation_type == "train_full0109":
            return self._augment_train_full0109(orig_inputs)
        if self.split == "train" and self.augmentation_type == "train_full0119":
            return self._augment_train_full0119(orig_inputs)

        return orig_inputs
        # Evaluation mode: only keep start and end frames
        if self.augmentation_type == "eval_0107_keyframe_start_end":
            return self._augment_start_end_only(orig_inputs)

        # Original mode or non-training split
        if self.augmentation_type == "original" or self.split != "train":
            return self._augment_original_mode(orig_inputs)

        # Training augmentation
        return self._augment_training_mode(orig_inputs)

    def _augment_start_end_only(self, orig_inputs):
        """Keep only start and end frame features."""
        augmented = orig_inputs.copy()
        augmented["feature"] = torch.zeros_like(orig_inputs["feature"])
        augmented["feature"][0] = orig_inputs["feature"][0]
        augmented["feature"][-1] = orig_inputs["feature"][-1]
        return augmented

    def _augment_original_mode(self, orig_inputs):
        """Original augmentation mode with 10% CFG dropout."""
        if self.split == "train" and random.random() < 0.1:
            augmented = orig_inputs.copy()
            augmented["feature"] = torch.zeros_like(orig_inputs["feature"])
            return augmented
        return orig_inputs

    def _augment_train_full0109(self, orig_inputs):
        """
        - [0,   0.5]: original
        - [0.5, 0.7]: drop random frames
        - [0.7, 0.9]: keep random keyframes
        - [0.9, 1.0]: drop all
        """
        prob = random.random()
        if prob <= 0.5:
            return orig_inputs
        augmented = orig_inputs.copy()
        if prob > 0.5 and prob <= 0.7:
            # mask random intervals
            length = orig_inputs["length"]
            # 最多抹掉一半的区间
            try:
                interval_length = random.randint(1, length // 2)
                interval_start = random.randint(0, length - interval_length)
            except Exception as e:
                print(f"length: {length}")
                print(f"Error: {e}")
                raise e
            augmented["feature"][interval_start : interval_start + interval_length] = 0
            return augmented
        elif prob > 0.7 and prob <= 0.9:
            augmented["feature"] = torch.zeros_like(orig_inputs["feature"])
            return self._augment_startend_and_random(orig_inputs, augmented)
        elif prob > 0.9:
            augmented["feature"] = torch.zeros_like(orig_inputs["feature"])
            return augmented
        else:
            raise NotImplementedError(f"Invalid probability: {prob}")

    def _augment_train_full0119(self, orig_inputs):
        """
        - [0,   0.7]: original
        - [0.7, 0.8]: drop random frames
        - [0.8, 0.9]: keep random keyframes
        - [0.9, 1.0]: drop all
        """
        prob = random.random()
        if prob <= 0.7:
            return orig_inputs
        augmented = orig_inputs.copy()
        if prob > 0.7 and prob <= 0.8:
            # mask random intervals
            length = orig_inputs["length"]
            # 最多抹掉一半的区间
            try:
                interval_length = random.randint(1, length // 2)
                interval_start = random.randint(0, length - interval_length)
            except Exception as e:
                print(f"length: {length}")
                print(f"Error: {e}")
                raise e
            augmented["feature"][interval_start : interval_start + interval_length] = 0
            return augmented
        elif prob > 0.8 and prob <= 0.9:
            augmented["feature"] = torch.zeros_like(orig_inputs["feature"])
            return self._augment_startend_and_random(orig_inputs, augmented)
        elif prob > 0.9:
            augmented["feature"] = torch.zeros_like(orig_inputs["feature"])
            return augmented
        else:
            raise NotImplementedError(f"Invalid probability: {prob}")

    def _augment_startend_and_random(self, orig_inputs, augmented):
        """Augmentation with start/end frames plus random keyframes."""
        sample_frames = max(orig_inputs["length"] // 30 - 1, 0)

        if sample_frames == 0:
            keyframes = [0, orig_inputs["length"] - 1]
        else:
            if sample_frames > 1:
                sample_frames = random.randint(1, sample_frames)
            keyframes = sorted(random.sample(range(1, orig_inputs["length"] - 1), sample_frames))
            keyframes = keyframes + [0, orig_inputs["length"] - 1]

        augmented["feature"][keyframes] = orig_inputs["feature"][keyframes]
        return augmented


# =============================================================================
# ExampleDataset
# =============================================================================


class MotionCameraTransform:
    @staticmethod
    def _compute_wv_transform(camera_R0):
        """Compute world-to-WV coordinate transformation."""
        axis_z_in_c = torch.tensor([0, 0, 1], dtype=torch.float32)
        axis_z_in_w = camera_R0.t() @ axis_z_in_c
        axis_up_in_w = torch.tensor([0, 1, 0], dtype=torch.float32)

        axis_newx_in_w = torch.cross(axis_up_in_w, axis_z_in_w, dim=-1)
        axis_newx_in_w = axis_newx_in_w / axis_newx_in_w.norm(dim=-1, keepdim=True)

        axis_newz_in_w = torch.cross(axis_newx_in_w, axis_up_in_w, dim=-1)
        axis_newz_in_w = axis_newz_in_w / axis_newz_in_w.norm(dim=-1, keepdim=True)

        relative_transform = torch.stack([axis_newx_in_w, axis_up_in_w, axis_newz_in_w], dim=-1).t()

        # Verify transform correctness
        # CoordinateTransformer._verify_transform(relative_transform, axis_newx_in_w, axis_newz_in_w)

        return relative_transform

    def _process_motion_to_wv(self, motion, camera_RT, smpl_skeleton, split):
        """Process motion data to WV coordinate system."""
        camera_R0 = camera_RT[0, :3, :3]
        relative_transform = self._compute_wv_transform(camera_R0)

        # Get j_shaped
        if split == "eval" and "joints" in motion:
            j_shaped = motion["joints"][:, :1] - motion["transl"][:, None, :]
        else:
            j_shaped = smpl_skeleton.compute_j_shaped(motion["shapes"][:1])

        # Transform root rotation and translation
        root_rotation, transl = process_r_t(
            relative_transform, motion["root_rotations"], motion["transl"], j_shaped[:, 0]
        )
        transl0 = transl[:1].clone()
        transl = transl - transl0

        return root_rotation, transl, relative_transform, transl0

    @staticmethod
    def transform_camera_to_wv(camera_RT, relative_transform, transl0):
        camera_RT = camera_RT.clone()
        trans = torch.eye(4)
        trans[:3, :3] = relative_transform
        trans[:3, 3] = -transl0
        camera_RT = camera_RT @ torch.inverse(trans)
        return camera_RT

    @staticmethod
    def _compute_camera_transforms(camera_RT):
        """Compute camera coordinate transformations."""
        camera_R0 = camera_RT[0, :3, :3]
        R_to_first_frame = camera_RT[:, :3, :3] @ camera_R0.t()[None]

        # Camera center coordinates
        T_camera_wv = torch.einsum("tij,tj->ti", camera_RT[:, :3, :3].transpose(1, 2), -camera_RT[:, :3, -1])

        # Convert to WV coordinate system
        center_velocity = T_camera_wv[1:] - T_camera_wv[:-1]
        center_velocity = torch.cat([center_velocity, center_velocity[-1:]], dim=0)

        return R_to_first_frame, center_velocity

    @staticmethod
    def _verify_transform(transform, axis_x, axis_z):
        """Verify coordinate transform is correct."""
        target_x = torch.tensor([1, 0, 0], dtype=torch.float32)
        target_z = torch.tensor([0, 0, 1], dtype=torch.float32)

        assert torch.allclose(
            transform @ axis_x.reshape(3, 1), target_x.reshape(3, 1), atol=1e-4
        ), "axis_newx_in_w is not correct"
        assert torch.allclose(
            transform @ axis_z.reshape(3, 1), target_z.reshape(3, 1), atol=1e-4
        ), "axis_newz_in_w is not correct"

    def _calculate_end_effector_vel(self, target, fps=30):
        """Calculate end effector velocities."""
        joint_ids = [7, 10, 8, 11, 20, 21]  # L_Ankle, L_foot, R_Ankle, R_foot, L_wrist, R_wrist

        smpl_params = {
            "rot6d": torch.cat([target["root_rot6d"][:, None], target["body_rot6d"]], dim=1).unsqueeze(0),
            "trans": target["trans"].unsqueeze(0),
            "shapes": target["shapes"].unsqueeze(0).mean(dim=-2, keepdim=True),
        }

        end_joints = get_joints_from_smpl_params(self.smpl_skeleton_cpu, smpl_params)["global_joints"].squeeze()
        end_joints = end_joints[..., joint_ids, :]

        end_joints_vel = end_joints[1:] - end_joints[:-1]
        end_joints_vel = torch.cat([end_joints_vel, end_joints_vel[-1:]], dim=0)
        end_joints_vel = end_joints_vel * fps

        return end_joints_vel

    # @staticmethod
    # def _build_coord_transform_dict(camera_RT, relative_transform, R_to_first, T_to_first):
    #     """Build coordinate transformation dictionary."""
    #     return {
    #         "world_to_wv": relative_transform[None].repeat(camera_RT.shape[0], 1, 1),
    #         "world_to_cami": camera_RT,
    #         "cam0_to_cami": torch.cat([R_to_first, T_to_first[..., None]], dim=2),
    #     }


class ExampleDataset(BaseDatasetMixin, FeatureLoaderMixin, AugmentationMixin, MotionCameraTransform):
    """Basic dataset for loading motion data with visual features."""

    def __init__(
        self, roots, augmentation_type="original", augment_n=0, max_len=360, round_frames=4, split="val", **kwargs
    ):
        self.augmentation_type = augmentation_type
        self.augment_n = augment_n
        self.max_len = max_len
        self.round_frames = round_frames
        self.split = split
        self.filenames = self._load_all_filenames(roots, self.split)

    def make_dummy_feature(self, target, camera):
        """Create dummy features from camera-projected rotations (for testing)."""
        root_rotation = rot6d_to_rotation_matrix(target["rot6d"][:, 0])
        RT = camera["RT"]
        RT_padding = torch.cat([RT, RT[-1:].repeat(self.max_len - RT.shape[0], 1, 1)], dim=0)
        root_rotation_camera = RT_padding[:, :3, :3] @ root_rotation

        root_rot6d_camera = rotation_matrix_to_rot6d(root_rotation_camera)
        rot6d_camera = torch.cat(
            [
                root_rot6d_camera[:, None],
                target["rot6d"][:, 1:],
            ],
            dim=1,
        )
        rot6d_camera = rot6d_camera.reshape(rot6d_camera.shape[0], -1)

        if self.split == "train":
            rot6d_camera = rot6d_camera + torch.randn_like(rot6d_camera) * 0.1

        return {"feature": rot6d_camera}

    def __getitem__(self, index):
        data = self.filenames[index]
        start_frame = data.get("start_frame", 0)
        end_frame = data.get("end_frame", self.max_len)

        camera = load_camera_from_npz(data["camera_name"], start_frame, end_frame)

        if self.split == "infer":
            motion = {
                "rot6d": torch.zeros(camera["RT"].shape[0], 52, 6),
                "trans_vel": torch.zeros(camera["RT"].shape[0], 3),
                "trans": torch.zeros(camera["RT"].shape[0], 3),
                "shapes": torch.zeros(camera["RT"].shape[0], 16),
            }
        else:
            motion = load_raw_motion_from_npz(data["motion_name"], start_frame, end_frame)

        target = {
            "rot6d": motion["rot6d"],
            "trans_vel": motion["trans_vel"],
            "trans": motion["trans"],
            "shapes": motion["shapes"],
        }

        # Add camera info
        gravity_vec = torch.tensor([0, -1, 0], dtype=torch.float32)
        R_c2gv = get_R_c2gv(camera["RT"][:, :3, :3], gravity_vec)
        cam_angvel = compute_cam_angvel(camera["RT"][:, :3, :3])
        target["camera_R"] = camera["RT"][:, :3, :3]
        target["R_c2gv"] = R_c2gv
        target["cam_angvel"] = cam_angvel

        target = self.padding_or_clip(
            target,
            self.max_len,
            round_frames=4,
            keys=["rot6d", "trans_vel", "trans", "shapes", "camera_R", "R_c2gv", "cam_angvel"],
        )

        inputs = self.load_feature(data["feature_name"], start_frame, end_frame)
        inputs = self.padding_or_clip(inputs, self.max_len, round_frames=1, keys=["feature"])
        inputs = self.augment_input_features(inputs)

        if "length" in inputs:
            target["length"] = inputs["length"]

        return {
            "target": target,
            "inputs": inputs,
            "index": index,
            "length": target["length"],
            "meta": {
                "sequence_name": data["sequence_name"],
                "feature_name": data["feature_name"],
                "motion_name": data["motion_name"],
                "camera_name": data["camera_name"],
            },
        }


# =============================================================================
# UnifiedMotionDataset
# =============================================================================


class UnifiedMotionDataset(ExampleDataset):
    """Extended dataset with WV coordinate system and end effector computation."""

    def __init__(self, timer=False, load_feature=True, **kwargs):
        super().__init__(**kwargs)
        self.smpl_skeleton_cpu = SMPLSkeleton()
        self.smplx2smpl = SMPLX2SMPL()
        self.timer = timer
        self.flag_load_feature = load_feature
        self.start_time = None

    def tic(self):
        """Start timer."""
        self.start_time = time.time()

    def toc(self, text=""):
        """Stop timer and print elapsed time."""
        if not self.timer or self.start_time is None:
            return 0
        print(f"[{self.__class__.__name__}] {text} cost: {time.time() - self.start_time} seconds")

    def _data_sanity_check(self, inputs, target, inputs_keys=None, target_keys=None):
        """Verify data consistency."""
        if inputs_keys is None:
            inputs_keys = ["feature", "camera_R", "camera_T", "bbox_info"]
        if target_keys is None:
            target_keys = ["root_rot6d", "body_rot6d", "transl_vel", "trans", "shapes", "end_effector_vel"]

        shape_0 = []
        for k in inputs_keys:
            val = inputs[k]
            shape_0.append(val.shape[0] if isinstance(val, torch.Tensor) else val)

        for k in target_keys:
            val = target[k]
            shape_0.append(val.shape[0] if isinstance(val, torch.Tensor) else val)

        assert all(s == shape_0[0] for s in shape_0), "Length inconsistency in inputs and target data"

    def _build_target_dict(self, motion, root_rotation, transl):
        """Build target dictionary from motion data."""
        target = {
            "root_rot6d": rotation_matrix_to_rot6d(root_rotation),
            "body_rot6d": rotation_matrix_to_rot6d(motion["body_rotations"]),
            "transl_vel": motion["transl_vel"],
            "trans": transl,
            "shapes": motion["shapes"],
        }
        target["end_effector_vel"] = self._calculate_end_effector_vel(target)

        if self.split == "eval" and "joints" in motion:
            target["joints"] = motion["joints"]
        if self.split == "eval" and "vertices" in motion:
            target["vertices"] = motion["vertices"]

        return target

    def _build_inputs_dict(self, data, camera, R_to_first_frame, T_to_prev_frame, start_frame, end_frame, pad_clip_max):
        """Build inputs dictionary with features and camera info."""
        inputs = self.load_feature(data["feature_name"], start_frame, end_frame)
        self.toc("load_feature")

        # Load bounding box
        if data["bbox_name"].endswith("_sapiens.npz"):
            bbox = self.load_sapiens_bbox(data["bbox_name"], start_frame, end_frame)
        else:
            bbox = self.load_bbox(data["bbox_name"], start_frame, end_frame)
        self.toc("load_bbox")

        # Apply feature masking based on bbox frame range
        if bbox["start_frame"] > 0:
            inputs["feature"][: bbox["start_frame"]] = 0
        if bbox["end_frame"] + 1 < inputs["feature"].shape[0]:
            inputs["feature"][bbox["end_frame"] + 1 :] = 0

        # Add camera info
        inputs["camera_R"] = R_to_first_frame.reshape(R_to_first_frame.shape[0], -1)
        inputs["camera_T"] = T_to_prev_frame * 30  # Use velocity representation
        inputs["camera_RT"] = torch.cat([inputs["camera_R"], inputs["camera_T"]], dim=-1)

        # Normalize bbox center
        K = camera["K"]
        bbox_center_pixel = torch.cat([bbox["bbox_center"], torch.ones_like(bbox["bbox_center"][:, :1])], dim=-1)
        bbox_center_pixel = torch.inverse(K) @ bbox_center_pixel.reshape(-1, 3, 1)
        bbox_center_pixel = bbox_center_pixel.squeeze(-1)[:, :2]
        bbox_scale_pixel = bbox["bbox_scale"] * 2 / (K[:, 0, 0] + K[:, 1, 1]).unsqueeze(-1)
        inputs["bbox_info"] = torch.cat([bbox_center_pixel, bbox_scale_pixel], dim=-1)

        # Padding and augmentation
        inputs = self.padding_or_clip(
            inputs, pad_clip_max, round_frames=1, keys=["feature", "camera_R", "camera_T", "bbox_info"]
        )
        inputs = self.augment_input_features(inputs)
        self.toc("augment_input_features")

        # Organize features into dict
        feature = inputs.pop("feature")
        inputs["feature"] = {
            "feature": feature,
            "camera_R": inputs["camera_R"],
            "camera_T": inputs["camera_T"],
            "bbox_info": inputs["bbox_info"],
        }

        return inputs, bbox, K

    def __getitem__(self, index):
        data = self.filenames[index]
        self.tic()
        if "start_frame" in data:
            start_frame = data["start_frame"]
            end_frame = data["end_frame"]
            pad_clip_max = end_frame - start_frame
        else:
            start_frame = 0
            end_frame = None if self.max_len == -1 else self.max_len
            pad_clip_max = end_frame

        start_frame, end_frame, pad_clip_max = self._get_frame_range(data, self.max_len)

        # Load camera
        camera = self._load_camera(data["camera_name"], start_frame, end_frame)
        if self.split == "infer":
            end_frame = camera["RT"].shape[0]
        self.toc("load_camera_from_npz")

        camera_is_static = torch.tensor(camera["movement_type"] == "static")

        # Load motion
        if "koala" not in data["motion_name"]:
            motion = self._load_motion(data["motion_name"], start_frame, end_frame)
        else:
            motion = {
                "root_rotations": torch.zeros(camera["RT"].shape[0], 3, 3),
                "body_rotations": torch.zeros(camera["RT"].shape[0], 51, 3, 3),
                "transl": torch.zeros(camera["RT"].shape[0], 3),
                "transl_vel": torch.zeros(camera["RT"].shape[0], 3),
                "shapes": torch.zeros(camera["RT"].shape[0], 16),
            }
        self.toc("load_rotation_translation_from_npz")

        # Process motion to WV coordinates
        root_rotation, transl, relative_transform = self._process_motion_to_wv(
            motion, camera["RT"], self.smpl_skeleton_cpu
        )

        # Compute camera transforms
        R_to_first_frame, T_to_first_frame, T_to_prev_frame = CoordinateTransformer.compute_camera_transforms(
            camera["RT"], relative_transform
        )

        coord_transform = CoordinateTransformer.build_coord_transform_dict(
            camera["RT"], relative_transform, R_to_first_frame, T_to_first_frame
        )
        coord_transform = self.padding_or_clip(
            coord_transform, pad_clip_max, round_frames=1, keys=["world_to_wv", "world_to_cami", "cam0_to_cami"]
        )
        self.toc("record coordinate transformations")

        # Build target
        target = self._build_target_dict(motion, root_rotation, transl)
        self.toc("process_r_t")

        target = self.padding_or_clip(
            target,
            pad_clip_max,
            round_frames=1,
            keys=["root_rot6d", "body_rot6d", "transl_vel", "trans", "shapes", "end_effector_vel"],
        )
        self.toc("padding_or_clip")

        # Build inputs
        if self.flag_load_feature:
            inputs, bbox, K = self._build_inputs_dict(
                data, camera, R_to_first_frame, T_to_prev_frame, start_frame, end_frame, pad_clip_max
            )

            if "length" in inputs:
                target["length"] = inputs["length"]
            if "keypoints3d" in bbox:
                target["keypoints3d"] = bbox["keypoints3d"]
            if self.split == "infer" or self.split == "eval":
                target["K"] = K
                target["cam_RT"] = camera["RT"]
                # target["kp2d"] = motion["kp2d"]
        else:
            inputs = {}
            bbox = {"start_frame": 0, "end_frame": end_frame}

        # Sanity check
        self._data_sanity_check(inputs["feature"], target)

        ret = {
            "target": target,
            "inputs": inputs,
            "coord_transform": coord_transform,
            "feature_start_frame": bbox["start_frame"],
            "feature_end_frame": bbox["end_frame"] + 1,
            "index": index,
            "camera_is_static": camera_is_static,
            "length": target["length"],
            "meta": {
                "sequence_name": data["sequence_name"],
                "feature_name": data["feature_name"],
                "motion_name": data["motion_name"],
                "camera_name": data["camera_name"],
                "video_name": data["video_name"],
            },
        }

        if self.split != "train":
            ret["meta"]["camera_RT"] = camera["RT"]

        return ret


class Eval3DDataset(BaseDatasetMixin, MotionCameraTransform, FeatureLoaderMixin):
    # 这个数据集用来实现包含3D GT的数据集的评估
    # 命令行参数里控制是否需要分段
    def __init__(
        self, roots, segment_infer=False, segment_len=300, segment_overlap=0, load_hand=False, is_demo_camera=False
    ):
        # 这个评估的时候不需要max_len
        self.split = "eval"
        self.filenames = self._load_all_filenames(roots, self.split)
        if segment_infer:
            self.filenames = self._apply_segment_inference(self.filenames, segment_len, segment_overlap)
        self.smplx2smpl = SMPLX2SMPL()
        self.smpl_skeleton_cpu = SMPLSkeleton()
        self.load_hand = load_hand
        self.is_demo_camera = is_demo_camera

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def _apply_segment_inference(filenames, segment_len=300, segment_overlap=0, min_frames=1):
        """Segment long sequences into multiple segments."""
        step = segment_len - segment_overlap

        assert segment_overlap >= 0, f"segment_overlap must be >= 0, got {segment_overlap}"
        assert step >= 0, f"segment_len ({segment_len}) must be > segment_overlap ({segment_overlap})"

        segmented = []
        for data_item in filenames:
            cam_npz = np.load(data_item["camera_name"])
            total_len = cam_npz["RT"].shape[0]

            start = 0
            while start < total_len:
                end = min(start + segment_len, total_len)
                seg_item = data_item.copy()
                seg_item["start_frame"] = int(start)
                seg_item["end_frame"] = int(end)
                seg_item["sequence_name"] = f"{data_item['sequence_name']}_s{start:06d}_e{end:06d}"
                if end - start < min_frames:
                    pass
                else:
                    segmented.append(seg_item)

                if end >= total_len:
                    break
                start += step

        return segmented

    def log_camera_RT(self, RT):
        # 计算相机中心位置 (camera center in world coordinates)
        # C = -R^T @ t
        camera_centers = -torch.einsum("tij,tj->ti", RT[:, :3, :3].transpose(1, 2), RT[:, :3, 3])

        # 计算相机Z轴朝向 (camera Z-axis direction in world coordinates)
        # Z轴方向是R^T的第三列 (或者R的第三行)
        camera_z_directions = RT[:, :3, :3].transpose(1, 2)[:, :, 2]  # R^T @ [0,0,1]

        print(f"[{self.__class__.__name__}]  First frame camera center: {camera_centers[0]}")
        print(f"[{self.__class__.__name__}]  Last frame camera center: {camera_centers[-1]}")

        # 计算相机中心位置范围
        center_min = camera_centers.min(dim=0)[0]
        center_max = camera_centers.max(dim=0)[0]
        center_range = center_max - center_min
        print(
            f"[{self.__class__.__name__}]  Camera center range: min={center_min}, max={center_max}, range={center_range}"
        )

        print(f"[{self.__class__.__name__}]  First frame camera Z-axis direction: {camera_z_directions[0]}")
        print(f"[{self.__class__.__name__}]  Last frame camera Z-axis direction: {camera_z_directions[-1]}")

        # 打印整段序列的位移的长度
        if len(camera_centers) >= 2:
            total_displacement_vector = camera_centers[-1] - camera_centers[0]
            total_displacement_length = torch.norm(total_displacement_vector)
            print(f"[{self.__class__.__name__}]  Total displacement length (last - first): {total_displacement_length:.6f}")
        else:
            print(f"[{self.__class__.__name__}]  Not enough frames ({len(camera_centers)}) to compute total displacement length")

    def _check_camera_RT(self, camera, displacement_threshold=0.5):
        """
        检查相机RT中的位移突变，如果某一帧的位移突变超过阈值，
        则将该帧及之后的相机中心都基于前一帧进行修正。

        Args:
            camera: 相机字典，包含 "RT" 键
            displacement_threshold: 位移突变的阈值，默认0.5

        Returns:
            camera: 修正后的相机字典
        """
        RT = camera["RT"].clone()  # 克隆以避免修改原始数据
        num_frames = RT.shape[0]

        if num_frames < 2:
            return camera

        # 计算所有帧的相机中心: C = -R^T * T
        camera_centers = -torch.einsum("tij,tj->ti", RT[:, :3, :3].transpose(1, 2), RT[:, :3, 3])

        # 计算帧间位移
        frame_displacements = torch.norm(camera_centers[1:] - camera_centers[:-1], dim=1)

        # 检测位移突变并修正
        corrected = False
        for i in range(1, num_frames):
            displacement = frame_displacements[i - 1]

            if displacement > displacement_threshold:
                print(f"[{self.__class__.__name__}] Detected displacement jump at frame {i}: {displacement:.6f} > {displacement_threshold}")
                corrected = True

                # 从该帧开始，将相机中心修正为基于前一帧的连续位移
                # 计算位移差（突变量）
                jump_offset = camera_centers[i] - camera_centers[i - 1]
                # 期望的位移应该是较小的值，这里我们将其设为0（保持前一帧的位置）
                # 或者可以使用前几帧的平均位移来估计

                # 将从第i帧开始的所有相机中心减去这个突变偏移
                camera_centers[i:] = camera_centers[i:] - jump_offset

                # 更新RT中的T: T_new = -R * C_new
                for j in range(i, num_frames):
                    T_new = -torch.einsum("ij,j->i", RT[j, :3, :3], camera_centers[j])
                    RT[j, :3, 3] = T_new

                # 重新计算帧间位移以检测后续的突变
                if i < num_frames - 1:
                    frame_displacements[i:] = torch.norm(
                        camera_centers[i + 1:] - camera_centers[i:-1], dim=1
                    )

        if corrected:
            camera["RT"] = RT
            print(f"[{self.__class__.__name__}] Camera RT corrected for displacement jumps")

        return camera

    def __getitem__(self, index):
        data = self.filenames[index]
        # 推理的时候使用整段数据
        start_frame, end_frame, pad_clip_max = self._get_frame_range(data, max_len=-1)
        # Load camera
        camera = CameraDataLoader.load_camera(
            filename=data["camera_name"], start_frame=start_frame, end_frame=end_frame
        )
        # 在加载相机后调用
        camera = self._check_camera_RT(camera, displacement_threshold=0.5)
        if True:
            # 打印相机信息用于调试
            RT = camera["RT"]  # world to camera transform
            # self.log_camera_RT(RT)
        camera_is_static = torch.tensor(camera["movement_type"] == "static")

        motion = self._load_motion_for_eval(data["motion_name"], start_frame, end_frame)
        # Process motion to WV coordinates
        # ATTN：在输入的是SLAM的相机的时候，这里不应该这么操作
        root_rotation, transl, relative_transform, transl0 = self._process_motion_to_wv(
            motion, camera["RT"], self.smpl_skeleton_cpu, self.split
        )
        # 把相机转到WV系中去
        camera_RT = self.transform_camera_to_wv(camera["RT"], relative_transform, transl0)
        if self.is_demo_camera:
            # demo相机要把相机挪到原点去
            centers_old = -torch.einsum("tij,tj->ti", camera_RT[:, :3, :3].transpose(1, 2), camera_RT[:, :3, 3])
            centers_new = centers_old - centers_old[:1]
            centers_new[:, 2] += -2
            T_new = -torch.einsum("tij,tj->ti", camera_RT[:, :3, :3], centers_new)
            camera_RT[:, :3, 3] = T_new

        self.log_camera_RT(camera_RT)
        R_to_first_frame, center_velocity = self._compute_camera_transforms(camera_RT)

        # 读入特征
        inputs = self.load_feature(data["feature_name"], start_frame, end_frame, load_hand=self.load_hand)

        # Load bounding box
        if data["bbox_name"].endswith("_sapiens.npz"):
            bbox = self.load_sapiens_bbox(data["bbox_name"], start_frame, end_frame)
        else:
            bbox = self.load_bbox(data["bbox_name"], start_frame, end_frame)
        bbox["bbox_info"] = self.bbox_info_from_bbox(bbox, camera["K"])
        feature = inputs["feature"]

        inputs_dict = {
            "feature": {
                "feature": feature,
                "camera_R": R_to_first_frame.reshape(R_to_first_frame.shape[0], -1),
                "camera_T": center_velocity * 30,
                # "camera_T": center_velocity * 0.,
                "bbox_value": bbox["bbox"],
                "bbox_info": bbox["bbox_info"],
                "keypoints2d": bbox["keypoints2d"],
            },
            "length": inputs["length"],
        }

        target = {
            "root_rot6d": rotation_matrix_to_rot6d(root_rotation),
            "body_rot6d": rotation_matrix_to_rot6d(motion["body_rotations"]),
            "transl_vel": motion["transl_vel"],
            "trans": transl,
            "shapes": motion["shapes"],
            "joints3d": motion["joints"],
            "vertices": motion["vertices"],
        }
        target["end_effector_vel"] = self._calculate_end_effector_vel(target)

        ret = {
            "target": target,
            "inputs": inputs_dict,
            "feature_start_frame": bbox["start_frame"],
            "feature_end_frame": bbox["end_frame"] + 1,
            "index": index,
            "camera_is_static": camera_is_static,
            "length": inputs["length"],
            "meta": {
                "start_frame": start_frame,
                "sequence_name": data["sequence_name"],
                "feature_name": data["feature_name"],
                "motion_name": data["motion_name"],
                "camera_name": data["camera_name"],
                "video_name": data["video_name"],
                "camera_origin_K": camera["K"],
                "camera_origin_RT": camera["RT"],
                "camera_wv_RT": camera_RT,
            },
        }
        return ret


class Eval2DDataset(BaseDatasetMixin, MotionCameraTransform, FeatureLoaderMixin):
    # 这个用来实现纯2D数据集的评估
    def __init__(self, roots, segment_infer=False, segment_len=300, segment_overlap=0, load_hand=False):
        # 这个评估的时候不需要max_len
        self.split = "eval"
        self.filenames = self._load_all_filenames(roots, self.split)
        if segment_infer:
            self.filenames = Eval3DDataset._apply_segment_inference(
                self.filenames, segment_len, segment_overlap, min_frames=100
            )
        self.smplx2smpl = SMPLX2SMPL()
        self.smpl_skeleton_cpu = SMPLSkeleton()
        self.load_hand = load_hand

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        data = self.filenames[index]
        # 推理的时候使用整段数据
        # 相机帧数和视频帧数总是对应的
        start_frame, end_frame, pad_clip_max = self._get_frame_range(data, max_len=-1)
        # Load bounding box
        if data["bbox_name"].endswith("_sapiens.npz"):
            bbox = self.load_sapiens_bbox(data["bbox_name"], start_frame, end_frame)
        else:
            bbox = self.load_bbox(data["bbox_name"], start_frame, end_frame)
        print(f"bbox: {bbox['bbox_center'].shape}")
        # 检查外面的start_frame和end_frame与bbox的start_frame和end_frame是否一致
        if bbox["start_frame"] != 0:
            print(f"bbox['start_frame']: {bbox['start_frame']}, start_frame: {start_frame}")
            start_frame = start_frame + bbox["start_frame"]
        if (end_frame is None and bbox["end_frame"] is not None) or \
            (end_frame is not None and bbox["end_frame"] < end_frame): # 只有bbox 更小的时候，才使用
            print(f"bbox['end_frame']: {bbox['end_frame']}, end_frame: {end_frame}")
            end_frame = start_frame + bbox["end_frame"]
        # 把之前的数据都裁剪一下
        # Load camera
        print(f"[{self.__class__.__name__}] start_frame: {start_frame}, end_frame: {end_frame}")
        camera = CameraDataLoader.load_camera(
            filename=data["camera_name"], start_frame=start_frame, end_frame=end_frame
        )
        camera_is_static = torch.tensor(camera["movement_type"] == "static")

        # make empty motion
        num_frames = camera["RT"].shape[0]
        motion = {
            "root_rotations": torch.eye(3)[None].repeat(num_frames, 1, 1),
            "body_rotations": torch.eye(3)[None].repeat(num_frames, 51, 1, 1),
            "joints": torch.zeros(num_frames, 24, 3),
            "vertices": torch.zeros(num_frames, 6890, 3),
            "transl": torch.zeros(num_frames, 3),
            "transl_vel": torch.zeros(num_frames, 3),
            "shapes": torch.zeros(num_frames, 16),
        }
        # Process motion to WV coordinates
        root_rotation, transl, relative_transform, transl0 = self._process_motion_to_wv(
            motion, camera["RT"], self.smpl_skeleton_cpu, self.split
        )
        # 把相机转到WV系中去
        camera_RT = self.transform_camera_to_wv(camera["RT"], relative_transform, transl0)
        R_to_first_frame, center_velocity = self._compute_camera_transforms(camera_RT)

        # 读入特征
        inputs = self.load_feature(data["feature_name"], start_frame, end_frame, load_hand=self.load_hand)
        # TODO: 重新采样features

        bbox["bbox_info"] = self.bbox_info_from_bbox(bbox, camera["K"])
        feature = inputs["feature"]

        inputs_dict = {
            "feature": {
                "feature": feature,
                "camera_R": R_to_first_frame.reshape(R_to_first_frame.shape[0], -1),
                "camera_T": center_velocity * 30,
                "bbox_value": bbox["bbox"],
                "bbox_info": bbox["bbox_info"],
                "keypoints2d": bbox["keypoints2d"],
            },
            "length": inputs["length"],
        }

        target = {
            "root_rot6d": rotation_matrix_to_rot6d(root_rotation),
            "body_rot6d": rotation_matrix_to_rot6d(motion["body_rotations"]),
            "transl_vel": motion["transl_vel"],
            "trans": transl,
            "shapes": motion["shapes"],
            "joints3d": motion["joints"],
            "vertices": motion["vertices"],
        }
        target["end_effector_vel"] = self._calculate_end_effector_vel(target)

        self._data_sanity_check(inputs_dict["feature"], target)

        ret = {
            "target": target,
            "inputs": inputs_dict,
            "feature_start_frame": bbox["start_frame"],
            "feature_end_frame": bbox["end_frame"] + 1,
            "index": index,
            "camera_is_static": camera_is_static,
            "length": inputs["length"],
            "meta": {
                "start_frame": start_frame,
                "sequence_name": data["sequence_name"],
                "feature_name": data["feature_name"],
                "motion_name": data["motion_name"],
                "camera_name": data["camera_name"],
                "video_name": data["video_name"],
                "camera_origin_K": camera["K"],
                "camera_origin_RT": camera["RT"],
                "camera_wv_RT": camera_RT,
                "do_post_ik": data["do_post_ik"],
                "do_camera_fitting": data["do_camera_fitting"],
            },
        }
        return ret


class DemoDatasetWo2D(BaseDatasetMixin, MotionCameraTransform, FeatureLoaderMixin):
    """
    Simplified dataset for demo inference pipeline.
    Directly accepts file paths without complex directory structure parsing.
    """

    def __init__(
        self,
        video_path,
        camera_path,
        feature_path,
        bbox_path,
        sequence_name=None,
        segment_infer=False,
        segment_len=300,
        segment_overlap=0,
        load_hand=False,
    ):
        """
        Args:
            video_path: Path to the video file
            camera_path: Path to the camera npz file
            feature_path: Path to the SAM3D feature pt file
            bbox_path: Path to the bbox npz file
            sequence_name: Name of the sequence (default: basename of video)
            segment_infer: Whether to segment the sequence for inference
            segment_len: Length of each segment
            segment_overlap: Overlap between segments
            load_hand: Whether to load hand features
        """
        self.split = "eval"
        self.load_hand = load_hand
        self.smplx2smpl = SMPLX2SMPL()
        self.smpl_skeleton_cpu = SMPLSkeleton()

        # Store file paths
        if sequence_name is None:
            sequence_name = os.path.basename(video_path).replace(".mp4", "")

        # Create a single data item
        data_item = {
            "sequence_name": sequence_name,
            "video_name": video_path,
            "camera_name": camera_path,
            "feature_name": feature_path,
            "bbox_name": bbox_path,
            "motion_name": "",  # No GT motion for demo
        }

        # Check if files exist
        for key, path in [
            ("camera_name", camera_path),
            ("feature_name", feature_path),
            ("bbox_name", bbox_path),
        ]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"{key}: {path} does not exist")

        self.filenames = [data_item]

        # Apply segmentation if requested
        if segment_infer:
            self.filenames = Eval3DDataset._apply_segment_inference(
                self.filenames, segment_len, segment_overlap, min_frames=100
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        data = self.filenames[index]

        # Get frame range
        start_frame, end_frame, pad_clip_max = self._get_frame_range(data, max_len=-1)

        # Load camera
        camera = CameraDataLoader.load_camera(
            filename=data["camera_name"], start_frame=start_frame, end_frame=end_frame
        )
        camera_is_static = torch.tensor(camera["movement_type"] == "static")

        # Create empty motion (no GT for demo)
        num_frames = camera["RT"].shape[0]
        motion = {
            "root_rotations": torch.eye(3)[None].repeat(num_frames, 1, 1),
            "body_rotations": torch.eye(3)[None].repeat(num_frames, 51, 1, 1),
            "joints": torch.zeros(num_frames, 24, 3),
            "vertices": torch.zeros(num_frames, 6890, 3),
            "transl": torch.zeros(num_frames, 3),
            "transl_vel": torch.zeros(num_frames, 3),
            "shapes": torch.zeros(num_frames, 16),
        }

        # Process motion to WV coordinates
        root_rotation, transl, relative_transform, transl0 = self._process_motion_to_wv(
            motion, camera["RT"], self.smpl_skeleton_cpu, self.split
        )

        # Transform camera to WV coordinates
        camera_RT = self.transform_camera_to_wv(camera["RT"], relative_transform, transl0)
        R_to_first_frame, center_velocity = self._compute_camera_transforms(camera_RT)

        # Load features
        inputs = self.load_feature(data["feature_name"], start_frame, end_frame, load_hand=self.load_hand)

        # Load bounding box

        bbox = self.load_bbox(data["bbox_name"], start_frame, end_frame)
        bbox["bbox_info"] = self.bbox_info_from_bbox(bbox, camera["K"])
        feature = inputs["feature"]

        # Assert all tensors have the same length
        # This ensures data consistency is maintained during data generation
        assert (
            feature.shape[0] == R_to_first_frame.shape[0]
        ), f"Feature length ({feature.shape[0]}) != Camera R length ({R_to_first_frame.shape[0]})"
        assert (
            feature.shape[0] == center_velocity.shape[0]
        ), f"Feature length ({feature.shape[0]}) != Camera T length ({center_velocity.shape[0]})"
        assert (
            feature.shape[0] == bbox["bbox_info"].shape[0]
        ), f"Feature length ({feature.shape[0]}) != BBox info length ({bbox['bbox_info'].shape[0]})"
        assert (
            feature.shape[0] == bbox["keypoints2d"].shape[0]
        ), f"Feature length ({feature.shape[0]}) != Keypoints2D length ({bbox['keypoints2d'].shape[0]})"

        # Build inputs dictionary
        inputs_dict = {
            "feature": {
                "feature": feature,
                "camera_R": R_to_first_frame.reshape(R_to_first_frame.shape[0], -1),
                "camera_T": center_velocity * 30,
                "bbox_info": bbox["bbox_info"],
                "keypoints2d": bbox["keypoints2d"],
            },
            "length": inputs["length"],
        }

        # Build target dictionary
        target = {
            "root_rot6d": rotation_matrix_to_rot6d(root_rotation),
            "body_rot6d": rotation_matrix_to_rot6d(motion["body_rotations"]),
            "transl_vel": motion["transl_vel"],
            "trans": transl,
            "shapes": motion["shapes"],
            "joints3d": motion["joints"],
            "vertices": motion["vertices"],
        }
        target["end_effector_vel"] = self._calculate_end_effector_vel(target)

        # Sanity check
        self._data_sanity_check(inputs_dict["feature"], target)

        # Build return dictionary
        ret = {
            "target": target,
            "inputs": inputs_dict,
            "feature_start_frame": bbox["start_frame"],
            "feature_end_frame": bbox["end_frame"] + 1,
            "index": index,
            "camera_is_static": camera_is_static,
            "length": inputs["length"],
            "meta": {
                "start_frame": start_frame,
                "sequence_name": data["sequence_name"],
                "feature_name": data["feature_name"],
                "motion_name": data.get("motion_name", ""),
                "camera_name": data["camera_name"],
                "video_name": data["video_name"],
                "camera_origin_K": camera["K"],
                "camera_origin_RT": camera["RT"],
                "camera_wv_RT": camera_RT,
            },
        }
        return ret


class Eval2DDatasetKeyframe(Eval2DDataset):
    # FIXME: 这个数据集偷懒了，应该重新实现一个getitem，支持从图片中进行读取的
    def __getitem__(self, index):
        ret = super().__getitem__(index)
        # 对parent取出来的数据，只保留指定的关键帧
        if "keyframes" not in self.filenames[index] and "drop_interval" not in self.filenames[index]:
            return ret
        if "keyframes" in self.filenames[index]:
            keyframes = self.filenames[index]["keyframes"]
            feature = torch.zeros_like(ret["inputs"]["feature"]["feature"])
            for srcframe, targetframe in keyframes:
                feature[targetframe] = ret["inputs"]["feature"]["feature"][srcframe]
            ret["inputs"]["feature"]["feature"] = feature
            # 取目标frame的最大值作为实际的长度
            ret["length"] = max([k[1] for k in keyframes]) + 1
        if "drop_interval" in self.filenames[index]:
            drop_interval = self.filenames[index]["drop_interval"]
            ret["inputs"]["feature"]["feature"][drop_interval[0] : drop_interval[1]] = 0
        return ret


# class DemoDatasetWo2D:
#     # 这个dataset不需要读入sapiens 关键点
#     # 直接进行推理，不能进行后处理优化
#     pass


class DemoDatasetW2D:
    # 这个dataset需要读入sapiens2D关键点
    # 可以进行后处理优化
    pass


# =============================================================================
# Test Functions
# =============================================================================


def log_tensor_or_other(data):
    """Log tensor shapes and other data types."""
    for key in data:
        if isinstance(data[key], torch.Tensor):
            print(f"{key}: {data[key].shape} {data[key].dtype}")
        elif isinstance(data[key], dict):
            log_tensor_or_other(data[key])
        else:
            print(f"{key}: {data[key]}")


def log_dict_shapes(d):
    """Log shapes of all tensors in a dictionary."""
    assert isinstance(d, dict)
    for k in d.keys():
        if isinstance(d[k], torch.Tensor):
            print(f"inputs {k}: ", d[k].shape)
        elif isinstance(d[k], dict):
            log_dict_shapes(d[k])
        else:
            print(f"inputs {k}: ", d[k])


def test_example_dataset():
    dataset = ExampleDataset(
        roots=[
            {
                "root": "/apdcephfs_cq10/share_1467498/datasets/motion_data/BlenderMotion/251114",
                "folders": ["Academic/HumanML3D-CMU"],
                "feature_format": "{name}_sam3d_feat.pt",
            }
        ],
        max_len=360,
        round_frames=4,
    )
    print(len(dataset))
    for i in range(0, len(dataset), 100):
        data = dataset[i]
        if data["inputs"]["feature"].isnan().any():
            print(data["meta"]["feature_name"])
            breakpoint()


def test_unified_motion_dataset():
    from hymotion.utils.loaders import load_object, read_yaml

    cfg = read_yaml("configs/v2m_generation/data/original_wv_bedlam_150h.yml")
    cfg = read_yaml("configs/v2m_generation/data/original_wv_2000h.yml")
    dataset = load_object(cfg["val_dataset"], cfg["val_dataset_args"])
    print(len(dataset))

    start_time = time.time()
    index_list = torch.randint(0, len(dataset), (12,)).tolist()
    for i in index_list:
        data = dataset[i]
        log_tensor_or_other(data["inputs"])
        log_tensor_or_other(data["target"])
    end_time = time.time()
    print(f"time cost: {end_time - start_time} seconds")

    for num_work in [0, 1, 2, 4, 8, 16]:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=32, shuffle=False, num_workers=num_work, drop_last=True
        )
        get_data_time_list = []
        end_time = time.time()
        for batch_idx, batch in enumerate(dataloader):
            get_data_time = time.time() - end_time
            get_data_time_list.append(get_data_time)

            for key in batch["inputs"]:
                if isinstance(batch["inputs"][key], torch.Tensor):
                    print(f"{key}: {batch['inputs'][key].shape} {batch['inputs'][key].dtype}")
                else:
                    print(f"{key}: {batch['inputs'][key]}")
            for key in batch["target"]:
                if isinstance(batch["target"][key], torch.Tensor):
                    print(f"{key}: {batch['target'][key].shape} {batch['target'][key].dtype}")
                else:
                    print(f"{key}: {batch['target'][key]}")

            time.sleep(0.5)
            print(f"batch_idx: {batch_idx}, get_data_time: {get_data_time:.2f}s")
            end_time = time.time()
            if batch_idx == 16:
                break
        print(f"num_workers: {num_work}, mean get_data_time: {sum(get_data_time_list) / len(get_data_time_list):.2f}s")
    print(f"num_workers: 0, time cost: {end_time - start_time} seconds")

    for num_workers in [0, 4, 8, 16]:
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=16, shuffle=False, num_workers=num_workers, drop_last=True
        )
        start_time = time.time()
        for batch in dataloader:
            print(batch.keys())
        end_time = time.time()
        print(f"num_workers: {num_workers}, time cost: {end_time - start_time} seconds")


def calculate_mean_std(debug=False):
    from hymotion.utils.loaders import load_object, read_yaml
    import json

    cfg = read_yaml("configs/v2m_generation/data/original_wv_1200h.yml")
    if debug:
        dataset = load_object(cfg["val_dataset"], cfg["val_dataset_args"], load_feature=False)
    else:
        dataset = load_object(cfg["train_dataset"], cfg["train_dataset_args"], load_feature=False)
    print(len(dataset))

    collection_dict = {"root_rot6d": [], "body_rot6d": [], "transl_vel": [], "shapes": [], "end_effector_vel": []}
    step = 10
    for i in tqdm(range(0, len(dataset), step), desc="calculating mean and std"):
        data = dataset[i]
        length = data["length"]
        for key in collection_dict:
            collection_dict[key].append(data["target"][key][:length])

    mean_std_dict = {}
    for key in collection_dict:
        value = torch.cat(collection_dict[key], dim=0)
        mean_std_dict[key] = {
            "mean": value.mean(dim=0).tolist(),
            "std": value.std(dim=0).tolist(),
            "min": value.min(dim=0)[0].tolist(),
            "max": value.max(dim=0)[0].tolist(),
        }

    with open(f"assets/v2m_wv_mean_std_1200h_step{step}.json", "w") as f:
        json.dump(mean_std_dict, f, indent=4)


def plot_keypoints_to_image(keypoints2d, frame, col=(0, 0, 255)):
    for i in range(keypoints2d.shape[0]):
        x, y = keypoints2d[i, 0], keypoints2d[i, 1]
        cv2.circle(frame, (int(x), int(y)), 5, col, -1)
    return frame


def check_by_2D_keypoints(data, video_name):
    keypoints2d = data["inputs"]["feature"]["keypoints2d"]
    print(f"keypoints2d shape: {keypoints2d.shape}")
    print(f"video_name: {video_name}")

    debug_dir = os.path.basename(video_name).replace(".mp4", "")
    debug_dir = os.path.join("debug", debug_dir + "_2d_keypoints")

    cap = cv2.VideoCapture(video_name)
    for index in range(keypoints2d.shape[0]):
        keypoints2d_frame = keypoints2d[index]
        ret, frame = cap.read()
        if not ret:
            break
        frame = plot_keypoints_to_image(keypoints2d_frame, frame)
        outname = f"{debug_dir}/{index:06d}.jpg"
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, frame)
    cap.release()
    # 使用ffmpeg合成视频
    cmd = (
        f"ffmpeg -loglevel error -y -r 30 -i {debug_dir}/%06d.jpg -c:v libx265 -q:v 5 -pix_fmt yuv420p {debug_dir}.mp4"
    )
    os.system(cmd)
    print(f"video saved to {debug_dir}.mp4")
    shutil.rmtree(debug_dir)


def _check_by_reproj(video_name, keypoints3d, K, RT, keypoints2d=None, postfix="_reproj", start_frame=0):
    debug_dir = os.path.basename(video_name).replace(".mp4", "")
    debug_dir = os.path.join("debug", "debug3d", debug_dir + postfix + f"_{start_frame:06d}")

    assert len(K.shape) == 3 and K.shape[1] == 3 and K.shape[2] == 3, f"K shape: {K.shape}"
    assert len(RT.shape) == 3 and RT.shape[1] == 4 and RT.shape[2] == 4, f"RT shape: {RT.shape}"

    keypoints3d_camera = RT[:, :3, :3] @ keypoints3d.transpose(1, 2) + RT[:, :3, 3:4]
    keypoints3d_K = K @ keypoints3d_camera
    keypoints3d_2d = keypoints3d_K[:, :2] / keypoints3d_K[:, 2:3]
    keypoints3d_2d = keypoints3d_2d.transpose(1, 2)
    print(f"keypoints3d_2d shape: {keypoints3d_2d.shape}")

    cap = cv2.VideoCapture(video_name)
    if start_frame > 0:
        for i in range(start_frame):
            cap.read()
    for index in range(keypoints3d.shape[0]):
        keypoints3d_frame = keypoints3d_2d[index]
        ret, frame = cap.read()
        if not ret:
            print(f"index: {index}, video_name: {video_name} is not read")
            break
        frame = plot_keypoints_to_image(keypoints3d_frame[:, :2], frame)
        if keypoints2d is not None:
            keypoints2d_frame = keypoints2d[index]
            frame = plot_keypoints_to_image(keypoints2d_frame[:, :2], frame, col=(0, 255, 0))
        frame = cv2.resize(frame, None, fx=0.5, fy=0.5)
        outname = f"{debug_dir}/{index:06d}.jpg"
        os.makedirs(os.path.dirname(outname), exist_ok=True)
        cv2.imwrite(outname, frame)
    cap.release()
    # 使用ffmpeg合成视频
    cmd = (
        f"ffmpeg -loglevel error -y -r 30 -i {debug_dir}/%06d.jpg -c:v libx265 -q:v 5 -pix_fmt yuv420p {debug_dir}.mp4"
    )
    os.system(cmd)
    print(f"video saved to {debug_dir}.mp4")
    shutil.rmtree(debug_dir)


def _check_by_matplotlib(
    video_name,
    keypoints3d,
    K,
    RT,
    prefix="debug/",
    postfix="_matplotlib",
    frame_step=10,
    start_frame=0,
    use_camera_position=False,
):
    """
    使用matplotlib可视化相机参数和3D关键点。

    Args:
        video_name: 视频名称，用于生成输出文件名
        keypoints3d: 3D关键点，shape为 (N, num_joints, 3)
        K: 相机内参矩阵，shape为 (N, 3, 3)
        RT: 相机外参矩阵，shape为 (N, 4, 4)
        postfix: 输出文件名后缀
        frame_step: 每隔多少帧画一帧，默认为10
    """
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    from mpl_toolkits.mplot3d import Axes3D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # SMPL skeleton连接定义
    SMPL_KINTREE = [
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 4],
        [2, 5],
        [3, 6],
        [4, 7],
        [5, 8],
        [6, 9],
        [7, 10],
        [8, 11],
        [9, 12],
        [9, 13],
        [9, 14],
        [12, 15],
        [13, 16],
        [14, 17],
        [16, 18],
        [17, 19],
        [18, 20],
        [19, 21],
        [20, 22],
        [21, 23],
    ]

    debug_dir = os.path.basename(video_name).replace(".mp4", "")
    debug_dir = os.path.join(prefix, debug_dir)
    os.makedirs(debug_dir, exist_ok=True)

    if isinstance(keypoints3d, torch.Tensor):
        keypoints3d = keypoints3d.cpu().numpy()
    if isinstance(K, torch.Tensor):
        K = K.cpu().numpy()
    if isinstance(RT, torch.Tensor):
        RT = RT.cpu().numpy()

    assert len(K.shape) == 3 and K.shape[1] == 3 and K.shape[2] == 3, f"K shape: {K.shape}"
    assert len(RT.shape) == 3 and RT.shape[1] == 4 and RT.shape[2] == 4, f"RT shape: {RT.shape}"

    num_frames = keypoints3d.shape[0]
    num_joints = keypoints3d.shape[1]

    # 从RT矩阵中提取相机位置（RT是world-to-camera变换）
    # 相机在世界坐标系中的位置: C = -R^T @ t
    camera_positions = []
    camera_orientations = []
    for i in range(num_frames):
        R = RT[i, :3, :3]
        t = RT[i, :3, 3]
        # 相机位置
        cam_pos = -R.T @ t
        camera_positions.append(cam_pos)
        # 相机朝向（Z轴方向，即光轴方向）
        cam_orient = R.T @ np.array([0, 0, 1])
        camera_orientations.append(cam_orient)
    camera_positions = np.array(camera_positions)
    camera_orientations = np.array(camera_orientations)

    def swap_yz(point):
        """交换Y和Z坐标，用于Y-up显示"""
        return np.array([point[0], point[2], point[1]])

    def draw_camera_pyramid(ax, position, R, scale=0.3, color="blue", alpha=0.3):
        """
        绘制相机四棱锥（Y-up坐标系）。

        Args:
            ax: matplotlib 3D轴
            position: 相机位置 (3,)
            R: 相机旋转矩阵的转置 (world-to-camera的逆)
            scale: 四棱锥的大小
            color: 颜色
            alpha: 透明度
        """
        # 四棱锥顶点（在相机坐标系中）
        # 顶点在相机中心，底面在相机前方
        apex = np.array([0, 0, 0])
        # 底面四个角点（相机看向+Z方向）
        half_w = scale * 0.6
        half_h = scale * 0.4
        depth = scale
        corners = np.array(
            [
                [-half_w, -half_h, depth],
                [half_w, -half_h, depth],
                [half_w, half_h, depth],
                [-half_w, half_h, depth],
            ]
        )

        # 转换到世界坐标系
        apex_world = R @ apex + position
        corners_world = (R @ corners.T).T + position

        # 交换Y和Z用于Y-up显示
        apex_world_yup = swap_yz(apex_world)
        corners_world_yup = np.array([swap_yz(c) for c in corners_world])

        # 绘制四棱锥的四个侧面
        faces = []
        for i in range(4):
            j = (i + 1) % 4
            face = [apex_world_yup, corners_world_yup[i], corners_world_yup[j]]
            faces.append(face)
        # 底面
        faces.append(corners_world_yup.tolist())

        poly = Poly3DCollection(faces, alpha=alpha, facecolor=color, edgecolor="darkblue", linewidth=0.5)
        ax.add_collection3d(poly)

        return apex_world, corners_world

    def draw_skeleton(ax, joints, kintree, color="red", linewidth=1.5, marker_size=20):
        """
        绘制骨骼连线（Y-up坐标系，交换Y和Z）。

        Args:
            ax: matplotlib 3D轴
            joints: 关节点位置 (num_joints, 3)
            kintree: 骨骼连接列表
            color: 颜色
            linewidth: 线宽
            marker_size: 关节点大小
        """
        # 绘制关节点 (X, Z, Y) -> matplotlib的 (X, Y, Z) 对应显示为 Y-up
        ax.scatter(joints[:, 0], joints[:, 2], joints[:, 1], c=color, s=marker_size, marker="o")

        # 绘制骨骼连线
        for bone in kintree:
            if bone[0] < joints.shape[0] and bone[1] < joints.shape[0]:
                start = joints[bone[0]]
                end = joints[bone[1]]
                ax.plot([start[0], end[0]], [start[2], end[2]], [start[1], end[1]], color=color, linewidth=linewidth)

    # 计算场景范围
    all_points = keypoints3d.reshape(-1, 3)
    if use_camera_position:
        all_points = np.concatenate([all_points, camera_positions], axis=0)
    center = np.mean(all_points, axis=0)
    max_range = np.max(np.abs(all_points - center)) * 1.2

    # 为每个采样帧生成图片
    frame_indices = list(range(0, num_frames, frame_step))
    print(f"Total frames: {num_frames}, visualizing {len(frame_indices)} frames (every {frame_step} frames)")

    # 创建一个大图，同时展示所有帧的数据
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")

    # 使用颜色映射来区分不同帧
    cmap = plt.get_cmap("viridis")
    colors = [cmap(i / len(frame_indices)) for i in range(len(frame_indices))]

    for idx, frame_idx in enumerate(tqdm(frame_indices)):
        joints = keypoints3d[frame_idx]
        R = RT[frame_idx, :3, :3]
        cam_pos = camera_positions[frame_idx]
        R_inv = R.T  # world-to-camera的逆

        # 绘制骨骼
        color = colors[idx]
        draw_skeleton(ax, joints, SMPL_KINTREE, color=color, linewidth=1.0, marker_size=10)

        # 绘制相机
        draw_camera_pyramid(ax, cam_pos, R_inv, scale=0.2, color=color, alpha=0.2)

    # 设置轴范围 (Y-up: 交换Y和Z)
    ax.set_xlim(center[0] - max_range, center[0] + max_range)
    ax.set_ylim(center[2] - max_range, center[2] + max_range)  # 原Z -> 显示Y
    ax.set_zlim(center[1] - max_range, center[1] + max_range)  # type: ignore  # 原Y -> 显示Z

    ax.set_xlabel("X")
    ax.set_ylabel("Z")  # 原Z轴现在是水平的Y轴
    ax.set_zlabel("Y")  # type: ignore  # 原Y轴现在是垂直的Z轴
    ax.set_title(
        f"Camera and 3D Keypoints Visualization\n(every {frame_step} frames, total {len(frame_indices)} frames)"
    )

    # 设置视角：从正面平视 (elev=0水平, azim=-90从正面看)
    ax.view_init(elev=0, azim=-90)

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=num_frames - 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label("Frame Index")

    # 保存3D场景图
    output_path = os.path.join(debug_dir, f"3d_scene{postfix}_{start_frame:06d}.png")
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"3D scene saved to {output_path}")
    print(f"Total {len(frame_indices)} frames visualized in one scene")


def check_by_reproj(dataset, data, video_name, start_frame=0, use_camera_position=False):
    # 这里要使用3D关键点重投影
    debug_dir = os.path.basename(video_name).replace(".mp4", "")
    debug_dir = os.path.join("debug", debug_dir + "_reproj")
    use_origin_joints = False

    if use_origin_joints:
        joints3d = data["target"]["joints3d"]
    else:
        rot6d = torch.cat([data["target"]["root_rot6d"][:, None], data["target"]["body_rot6d"]], dim=1)
        params = {
            "rot6d": rot6d,
            "trans": data["target"]["trans"],
            "shapes": data["target"]["shapes"],
        }
        joints3d = dataset.smpl_skeleton_cpu(params)["keypoints3d"]

    K = data["meta"]["camera_origin_K"]
    camera_origin_RT = data["meta"]["camera_origin_RT"]
    camera_wv_RT = data["meta"]["camera_wv_RT"]
    # 打印两个相机的相机中心的开始帧和最后一帧的位置
    print(f"Camera center positions for {os.path.basename(video_name)}:")

    # 原始相机中心位置
    origin_camera_center_start = -camera_origin_RT[0, :3, :3].T @ camera_origin_RT[0, :3, 3]
    origin_camera_center_end = -camera_origin_RT[-1, :3, :3].T @ camera_origin_RT[-1, :3, 3]

    # 计算整段序列中相机中心的Y轴最小最大值
    origin_camera_centers = -torch.einsum(
        "tij,tj->ti", camera_origin_RT[:, :3, :3].transpose(1, 2), camera_origin_RT[:, :3, 3]
    )

    origin_y_min, origin_y_max = origin_camera_centers[:, 1].min(), origin_camera_centers[:, 1].max()
    origin_y_range = origin_y_max - origin_y_min

    # 原始相机轴朝向 (R^T的列向量是相机轴在世界坐标系中的朝向)
    origin_R_start = camera_origin_RT[0, :3, :3].T  # R^T
    origin_R_end = camera_origin_RT[-1, :3, :3].T
    origin_x_start, origin_y_start, origin_z_start = origin_R_start[:, 0], origin_R_start[:, 1], origin_R_start[:, 2]
    origin_x_end, origin_y_end, origin_z_end = origin_R_end[:, 0], origin_R_end[:, 1], origin_R_end[:, 2]
    print(f"  Original camera - Start frame: {origin_camera_center_start}")
    print(f"  Original camera - Start frame X-axis: {origin_x_start}")
    print(f"  Original camera - Start frame Y-axis: {origin_y_start}")
    print(f"  Original camera - Start frame Z-axis: {origin_z_start}")
    print(f"  Original camera - End frame: {origin_camera_center_end}")
    print(f"  Original camera - End frame X-axis: {origin_x_end}")
    print(f"  Original camera - End frame Y-axis: {origin_y_end}")
    print(f"  Original camera - End frame Z-axis: {origin_z_end}")
    print(f"  Original camera - Y range: [{origin_y_min:.4f}, {origin_y_max:.4f}], diff: {origin_y_range:.4f}")
    displacement_vec = origin_camera_center_end - origin_camera_center_start
    displacement_norm = np.linalg.norm(displacement_vec)
    print(f"  Original camera - Displacement norm (end - start): {displacement_norm:.6f}")

    # WV坐标系相机中心位置
    wv_camera_center_start = -camera_wv_RT[0, :3, :3].T @ camera_wv_RT[0, :3, 3]
    wv_camera_center_end = -camera_wv_RT[-1, :3, :3].T @ camera_wv_RT[-1, :3, 3]

    wv_camera_centers = -torch.einsum("tij,tj->ti", camera_wv_RT[:, :3, :3].transpose(1, 2), camera_wv_RT[:, :3, 3])
    wv_y_min, wv_y_max = wv_camera_centers[:, 1].min(), wv_camera_centers[:, 1].max()
    wv_y_range = wv_y_max - wv_y_min

    # WV相机轴朝向
    wv_R_start = camera_wv_RT[0, :3, :3].T  # R^T
    wv_R_end = camera_wv_RT[-1, :3, :3].T
    wv_x_start, wv_y_start, wv_z_start = wv_R_start[:, 0], wv_R_start[:, 1], wv_R_start[:, 2]
    wv_x_end, wv_y_end, wv_z_end = wv_R_end[:, 0], wv_R_end[:, 1], wv_R_end[:, 2]

    print(f"  WV camera - Start frame: {wv_camera_center_start}")
    print(f"  WV camera - Start frame X-axis: {wv_x_start}")
    print(f"  WV camera - Start frame Y-axis: {wv_y_start}")
    print(f"  WV camera - Start frame Z-axis: {wv_z_start}")
    print(f"  WV camera - End frame: {wv_camera_center_end}")
    print(f"  WV camera - End frame X-axis: {wv_x_end}")
    print(f"  WV camera - End frame Y-axis: {wv_y_end}")
    print(f"  WV camera - End frame Z-axis: {wv_z_end}")
    print(f"  WV camera - Y range: [{wv_y_min:.4f}, {wv_y_max:.4f}], diff: {wv_y_range:.4f}")
    displacement_vec = wv_camera_center_end - wv_camera_center_start
    displacement_norm = np.linalg.norm(displacement_vec)
    print(f"  WV camera - Displacement norm (end - start): {displacement_norm:.6f}")
    if use_origin_joints:
        RT = data["meta"]["camera_origin_RT"]
    else:
        RT = data["meta"]["camera_wv_RT"]
    # _check_by_reproj(video_name, joints3d, K, RT, start_frame=start_frame)
    # _check_by_matplotlib(video_name, joints3d, K, RT, start_frame=start_frame, use_camera_position=use_camera_position)
    if "24_outdoor_long_walk" in video_name:
        breakpoint()


def test_eval3d_dataset():
    from hymotion.utils.loaders import load_object, read_yaml

    # cfg = read_yaml("configs/v2m_generation/data/original_wv_eval_rich.yml")
    cfg = read_yaml("configs/v2m_generation/data/original_wv_eval_emdb2_gtcamera.yml")
    cfg = read_yaml("configs/v2m_generation/data/original_wv_eval_emdb2_vipecamera.yml")
    cfg["val_dataset_args"]["segment_infer"] = False
    cfg["val_dataset_args"]["segment_len"] = 300
    dataset = load_object(cfg["val_dataset"], cfg["val_dataset_args"])
    print(len(dataset))
    for i in range(0, len(dataset), 1):
        data = dataset[i]
        video_name = data["meta"]["video_name"]
        assert os.path.exists(video_name), f"video_name {video_name} does not exist"
        joints3d = data["target"]["joints3d"]
        # 计算手臂长度
        # SMPL skeleton中的关键点索引：
        # 左肩: 16, 左肘: 18, 左腕: 20
        # 右肩: 17, 右肘: 19, 右腕: 21
        left_shoulder = joints3d[:, 16]  # (T, 3)
        left_elbow = joints3d[:, 18]  # (T, 3)
        left_wrist = joints3d[:, 20]  # (T, 3)

        right_shoulder = joints3d[:, 17]  # (T, 3)
        right_elbow = joints3d[:, 19]  # (T, 3)
        right_wrist = joints3d[:, 21]  # (T, 3)

        # 计算上臂长度（肩到肘）
        left_upper_arm_length = torch.norm(left_elbow - left_shoulder, dim=-1)  # (T,)
        right_upper_arm_length = torch.norm(right_elbow - right_shoulder, dim=-1)  # (T,)

        # 计算前臂长度（肘到腕）
        left_forearm_length = torch.norm(left_wrist - left_elbow, dim=-1)  # (T,)
        right_forearm_length = torch.norm(right_wrist - right_elbow, dim=-1)  # (T,)

        # 计算整个手臂长度（肩到腕）
        left_arm_length = torch.norm(left_wrist - left_shoulder, dim=-1)  # (T,)
        right_arm_length = torch.norm(right_wrist - right_shoulder, dim=-1)  # (T,)

        print(f"Video: {os.path.basename(video_name)}")
        print(f"Left upper arm length: mean={left_upper_arm_length.mean():.4f}, std={left_upper_arm_length.std():.4f}")
        print(
            f"Right upper arm length: mean={right_upper_arm_length.mean():.4f}, std={right_upper_arm_length.std():.4f}"
        )
        print(f"Left forearm length: mean={left_forearm_length.mean():.4f}, std={left_forearm_length.std():.4f}")
        print(f"Right forearm length: mean={right_forearm_length.mean():.4f}, std={right_forearm_length.std():.4f}")
        print(f"Left full arm length: mean={left_arm_length.mean():.4f}, std={left_arm_length.std():.4f}")
        print(f"Right full arm length: mean={right_arm_length.mean():.4f}, std={right_arm_length.std():.4f}")

        # 检查并可视化相机位置
        import matplotlib.pyplot as plt

        # 获取相机RT矩阵（优先使用WV坐标系下的相机）
        if "camera_wv_RT" in data["meta"]:
            RT = data["meta"]["camera_wv_RT"]
        elif "camera_origin_RT" in data["meta"]:
            RT = data["meta"]["camera_origin_RT"]
        else:
            RT = data["meta"].get("camera_RT", None)

        if RT is not None:
            # 计算相机中心位置 (camera center in world coordinates)
            # C = -R^T @ t
            camera_centers = -torch.einsum("tij,tj->ti", RT[:, :3, :3].transpose(1, 2), RT[:, :3, 3])
            camera_centers_np = camera_centers.cpu().numpy()

            num_frames = camera_centers_np.shape[0]
            fps = 30
            time_axis = np.arange(num_frames) / fps  # 时间轴（秒）

            # 创建输出目录
            output_dir = "debug/debug_emdb_camera"
            os.makedirs(output_dir, exist_ok=True)

            # 生成输出文件名
            sequence_name = data["meta"]["sequence_name"]
            output_path = os.path.join(output_dir, f"{sequence_name}_camera_centers.png")

            # 创建图表
            fig, axes = plt.subplots(3, 1, figsize=(12, 10))
            fig.suptitle(f"Camera Center Position - {sequence_name}", fontsize=14, fontweight='bold')

            # 绘制X坐标
            axes[0].plot(time_axis, camera_centers_np[:, 0], 'r-', linewidth=1.5, label='X')
            axes[0].set_ylabel('X Position', fontsize=12)
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()
            axes[0].set_title(f'X Coordinate (min={camera_centers_np[:, 0].min():.4f}, max={camera_centers_np[:, 0].max():.4f}, range={camera_centers_np[:, 0].max() - camera_centers_np[:, 0].min():.4f})')

            # 绘制Y坐标
            axes[1].plot(time_axis, camera_centers_np[:, 1], 'g-', linewidth=1.5, label='Y')
            axes[1].set_ylabel('Y Position', fontsize=12)
            axes[1].grid(True, alpha=0.3)
            axes[1].legend()
            axes[1].set_title(f'Y Coordinate (min={camera_centers_np[:, 1].min():.4f}, max={camera_centers_np[:, 1].max():.4f}, range={camera_centers_np[:, 1].max() - camera_centers_np[:, 1].min():.4f})')

            # 绘制Z坐标
            axes[2].plot(time_axis, camera_centers_np[:, 2], 'b-', linewidth=1.5, label='Z')
            axes[2].set_xlabel('Time (seconds)', fontsize=12)
            axes[2].set_ylabel('Z Position', fontsize=12)
            axes[2].grid(True, alpha=0.3)
            axes[2].legend()
            axes[2].set_title(f'Z Coordinate (min={camera_centers_np[:, 2].min():.4f}, max={camera_centers_np[:, 2].max():.4f}, range={camera_centers_np[:, 2].max() - camera_centers_np[:, 2].min():.4f})')

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Camera center visualization saved to {output_path}")

            # 打印统计信息
            print(f"\nCamera Center Statistics for {sequence_name}:")
            print(f"  Total frames: {num_frames}")
            print(f"  Duration: {num_frames / fps:.2f} seconds")
            print(f"  X: min={camera_centers_np[:, 0].min():.6f}, max={camera_centers_np[:, 0].max():.6f}, mean={camera_centers_np[:, 0].mean():.6f}, std={camera_centers_np[:, 0].std():.6f}")
            print(f"  Y: min={camera_centers_np[:, 1].min():.6f}, max={camera_centers_np[:, 1].max():.6f}, mean={camera_centers_np[:, 1].mean():.6f}, std={camera_centers_np[:, 1].std():.6f}")
            print(f"  Z: min={camera_centers_np[:, 2].min():.6f}, max={camera_centers_np[:, 2].max():.6f}, mean={camera_centers_np[:, 2].mean():.6f}, std={camera_centers_np[:, 2].std():.6f}")

            # 计算总位移
            if num_frames >= 2:
                total_displacement = camera_centers_np[-1] - camera_centers_np[0]
                total_distance = np.linalg.norm(total_displacement)
                print(f"  Total displacement: [{total_displacement[0]:.6f}, {total_displacement[1]:.6f}, {total_displacement[2]:.6f}]")
                print(f"  Total distance: {total_distance:.6f}")

        # check_by_2D_keypoints(data, video_name)
        # check_by_reproj(dataset, data, video_name, start_frame=data["meta"]["start_frame"])


def test_eval3d_camera():
    from hymotion.utils.loaders import load_object, read_yaml

    # cfg = read_yaml("configs/v2m_generation/data/original_wv_eval_rich.yml")
    cfg_gt = read_yaml("configs/v2m_generation/data/original_wv_eval_emdb2_gtcamera.yml")
    cfg_vipe = read_yaml("configs/v2m_generation/data/original_wv_eval_emdb2_vipecamera.yml")
    cfg_gt["val_dataset_args"]["segment_infer"] = False
    cfg_vipe["val_dataset_args"]["segment_infer"] = False
    dataset_gt = load_object(cfg_gt["val_dataset"], cfg_gt["val_dataset_args"])
    dataset_vipe = load_object(cfg_vipe["val_dataset"], cfg_vipe["val_dataset_args"])
    print(len(dataset_gt))
    for i in range(0, len(dataset_gt), 1):
        data_gt = dataset_gt[i]
        data_vipe = dataset_vipe[i]
        video_name = data_gt["meta"]["video_name"]
        assert os.path.exists(video_name), f"video_name {video_name} does not exist"

        # 检查并可视化相机位置
        import matplotlib.pyplot as plt

        def get_camera_centers(data):
            """获取相机中心位置"""
            # 获取相机RT矩阵（优先使用WV坐标系下的相机）
            if "camera_wv_RT" in data["meta"]:
                RT = data["meta"]["camera_wv_RT"]
            elif "camera_origin_RT" in data["meta"]:
                RT = data["meta"]["camera_origin_RT"]
            else:
                RT = data["meta"].get("camera_RT", None)

            if RT is None:
                return None, None

            # 计算相机中心位置 (camera center in world coordinates)
            # C = -R^T @ t
            camera_centers = -torch.einsum("tij,tj->ti", RT[:, :3, :3].transpose(1, 2), RT[:, :3, 3])
            camera_centers_np = camera_centers.cpu().numpy()

            # 将第一帧对齐到0
            camera_centers_np = camera_centers_np - camera_centers_np[0:1]

            # 计算相机轴方向 (camera axes directions in world coordinates)
            # R^T的列向量就是相机轴在世界坐标系中的方向
            R = RT[:, :3, :3]  # world-to-camera旋转矩阵
            R_T = R.transpose(1, 2)  # camera-to-world旋转矩阵
            # R_T的列向量分别是相机X、Y、Z轴在世界坐标系中的方向
            camera_axes = R_T.cpu().numpy()  # (T, 3, 3)，每一列是一个轴的方向向量

            return camera_centers_np, camera_axes

        # 获取GT和VIPE相机的中心位置和轴方向
        centers_gt, axes_gt = get_camera_centers(data_gt)
        centers_vipe, axes_vipe = get_camera_centers(data_vipe)

        if centers_gt is not None and centers_vipe is not None:
            num_frames = min(centers_gt.shape[0], centers_vipe.shape[0])
            centers_gt = centers_gt[:num_frames]
            centers_vipe = centers_vipe[:num_frames]
            axes_gt = axes_gt[:num_frames]
            axes_vipe = axes_vipe[:num_frames]

            fps = 30
            time_axis = np.arange(num_frames) / fps  # 时间轴（秒）

            # 创建输出目录
            output_dir = "debug/debug_emdb_camera"
            os.makedirs(output_dir, exist_ok=True)

            # 生成输出文件名
            sequence_name = data_gt["meta"]["sequence_name"]
            output_path = os.path.join(output_dir, f"{sequence_name}_camera_centers.png")

            # 计算所有位置数据的全局最小值和最大值，用于统一y轴范围
            all_center_data = np.concatenate([centers_gt, centers_vipe], axis=0)
            center_global_min = all_center_data.min()
            center_global_max = all_center_data.max()
            padding_center = (center_global_max - center_global_min) * 0.1
            center_y_min = center_global_min - padding_center
            center_y_max = center_global_max + padding_center

            # 相机轴方向是单位向量，y轴范围固定为-1到1
            axes_y_min = -1.0
            axes_y_max = 1.0

            # 创建图表：3行4列
            fig, axes = plt.subplots(3, 4, figsize=(20, 10))
            fig.suptitle(f"Camera Position & Orientation Comparison (GT vs VIPE) - {sequence_name}", fontsize=14, fontweight='bold')

            # 列标题和配置
            col_configs = [
                {
                    'name': 'Center Position',
                    'data_gt': centers_gt,
                    'data_vipe': centers_vipe,
                    'y_min': center_y_min,
                    'y_max': center_y_max,
                    'ylabel_prefix': '',
                    'unit': 'm',
                    'is_position': True
                },
                {
                    'name': 'Camera X-axis',
                    'data_gt': axes_gt[:, :, 0],  # X轴方向
                    'data_vipe': axes_vipe[:, :, 0],
                    'y_min': axes_y_min,
                    'y_max': axes_y_max,
                    'ylabel_prefix': 'X-axis',
                    'unit': '',
                    'is_position': False
                },
                {
                    'name': 'Camera Y-axis',
                    'data_gt': axes_gt[:, :, 1],  # Y轴方向
                    'data_vipe': axes_vipe[:, :, 1],
                    'y_min': axes_y_min,
                    'y_max': axes_y_max,
                    'ylabel_prefix': 'Y-axis',
                    'unit': '',
                    'is_position': False
                },
                {
                    'name': 'Camera Z-axis',
                    'data_gt': axes_gt[:, :, 2],  # Z轴方向
                    'data_vipe': axes_vipe[:, :, 2],
                    'y_min': axes_y_min,
                    'y_max': axes_y_max,
                    'ylabel_prefix': 'Z-axis',
                    'unit': '',
                    'is_position': False
                }
            ]

            # 行标签（X、Y、Z分量）
            row_labels = ['X', 'Y', 'Z']

            # 使用for循环绘制所有子图
            for col_idx, col_config in enumerate(col_configs):
                for row_idx in range(3):
                    ax = axes[row_idx, col_idx]

                    if col_config['is_position']:
                        # 相机中心位置：只绘制GT和VIPE的对应分量
                        ax.plot(time_axis, col_config['data_gt'][:, row_idx], 'r-', linewidth=1.5, label='GT', alpha=0.7)
                        ax.plot(time_axis, col_config['data_vipe'][:, row_idx], 'b--', linewidth=1.5, label='VIPE', alpha=0.7)
                        gt_range = col_config['data_gt'][:, row_idx].max() - col_config['data_gt'][:, row_idx].min()
                        vipe_range = col_config['data_vipe'][:, row_idx].max() - col_config['data_vipe'][:, row_idx].min()
                        ax.set_title(f'{row_labels[row_idx]} - GT range: {gt_range:.4f}{col_config["unit"]}, VIPE range: {vipe_range:.4f}{col_config["unit"]}')
                    else:
                        # 相机轴方向：只绘制GT和VIPE的对应分量（使用row_idx）
                        ax.plot(time_axis, col_config['data_gt'][:, row_idx], 'r-', linewidth=1.5, label='GT', alpha=0.7)
                        ax.plot(time_axis, col_config['data_vipe'][:, row_idx], 'b--', linewidth=1.5, label='VIPE', alpha=0.7)
                        gt_range = col_config['data_gt'][:, row_idx].max() - col_config['data_gt'][:, row_idx].min()
                        vipe_range = col_config['data_vipe'][:, row_idx].max() - col_config['data_vipe'][:, row_idx].min()
                        ax.set_title(f'{col_config["name"]} {row_labels[row_idx]} - GT range: {gt_range:.4f}, VIPE range: {vipe_range:.4f}')

                    # 设置y轴标签
                    if col_config['is_position']:
                        ax.set_ylabel(f'{row_labels[row_idx]} Position ({col_config["unit"]})', fontsize=12)
                    else:
                        ax.set_ylabel(f'{col_config["ylabel_prefix"]} {row_labels[row_idx]}', fontsize=12)

                    # 设置y轴范围
                    ax.set_ylim(col_config['y_min'], col_config['y_max'])
                    ax.grid(True, alpha=0.3)

                    # 只在第一行设置图例
                    if row_idx == 0:
                        ax.legend()

                    # 只在最后一行设置x轴标签
                    if row_idx == 2:
                        ax.set_xlabel('Time (seconds)', fontsize=12)

            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            plt.close()

            print(f"Camera center visualization saved to {output_path}")

            # 打印统计信息
            print(f"\nCamera Center Statistics for {sequence_name}:")
            print(f"  Total frames: {num_frames}")
            print(f"  Duration: {num_frames / fps:.2f} seconds")

            print(f"\n  GT Camera:")
            print(f"    X: min={centers_gt[:, 0].min():.6f}, max={centers_gt[:, 0].max():.6f}, mean={centers_gt[:, 0].mean():.6f}, std={centers_gt[:, 0].std():.6f}")
            print(f"    Y: min={centers_gt[:, 1].min():.6f}, max={centers_gt[:, 1].max():.6f}, mean={centers_gt[:, 1].mean():.6f}, std={centers_gt[:, 1].std():.6f}")
            print(f"    Z: min={centers_gt[:, 2].min():.6f}, max={centers_gt[:, 2].max():.6f}, mean={centers_gt[:, 2].mean():.6f}, std={centers_gt[:, 2].std():.6f}")

            print(f"\n  VIPE Camera:")
            print(f"    X: min={centers_vipe[:, 0].min():.6f}, max={centers_vipe[:, 0].max():.6f}, mean={centers_vipe[:, 0].mean():.6f}, std={centers_vipe[:, 0].std():.6f}")
            print(f"    Y: min={centers_vipe[:, 1].min():.6f}, max={centers_vipe[:, 1].max():.6f}, mean={centers_vipe[:, 1].mean():.6f}, std={centers_vipe[:, 1].std():.6f}")
            print(f"    Z: min={centers_vipe[:, 2].min():.6f}, max={centers_vipe[:, 2].max():.6f}, mean={centers_vipe[:, 2].mean():.6f}, std={centers_vipe[:, 2].std():.6f}")

            # 计算总位移
            if num_frames >= 2:
                total_displacement_gt = centers_gt[-1] - centers_gt[0]
                total_distance_gt = np.linalg.norm(total_displacement_gt)
                total_displacement_vipe = centers_vipe[-1] - centers_vipe[0]
                total_distance_vipe = np.linalg.norm(total_displacement_vipe)

                print(f"\n  GT Total displacement: [{total_displacement_gt[0]:.6f}, {total_displacement_gt[1]:.6f}, {total_displacement_gt[2]:.6f}], distance: {total_distance_gt:.6f}")
                print(f"  VIPE Total displacement: [{total_displacement_vipe[0]:.6f}, {total_displacement_vipe[1]:.6f}, {total_displacement_vipe[2]:.6f}], distance: {total_distance_vipe:.6f}")

        # check_by_2D_keypoints(data, video_name)
        # check_by_reproj(dataset, data, video_name, start_frame=data["meta"]["start_frame"])

def test_eval2d_dataset():
    from hymotion.utils.loaders import load_object, read_yaml

    # cfg = read_yaml("configs/v2m_generation/data/original_wv_eval_rich.yml")
    cfg = read_yaml("configs/v2m_generation/data/original_wv_eval_koala_all.yml")
    cfg["val_dataset_args"]["segment_infer"] = True
    cfg["val_dataset_args"]["segment_len"] = 300
    dataset = load_object(cfg["val_dataset"], cfg["val_dataset_args"])
    print(len(dataset))
    for i in range(0, len(dataset), 1):
        data = dataset[i]
        video_name = data["meta"]["video_name"]
        assert os.path.exists(video_name), f"video_name {video_name} does not exist"
        # _check_by_matplotlib(
        #     video_name,
        #     data["target"]["joints3d"],
        #     data["meta"]["camera_origin_K"],
        #     data["meta"]["camera_origin_RT"],
        #     start_frame=data["meta"]["start_frame"],
        #     use_camera_position=True,
        # )
        check_by_reproj(dataset, data, video_name, start_frame=data["meta"]["start_frame"], use_camera_position=True)
        breakpoint()


def unit_test():
    # test_eval3d_dataset()
    test_eval3d_camera()
    # test_eval2d_dataset()
    # test_unified_motion_dataset()


if __name__ == "__main__":
    # python -m hymotion.datasets.v2m_generation.base
    unit_test()
