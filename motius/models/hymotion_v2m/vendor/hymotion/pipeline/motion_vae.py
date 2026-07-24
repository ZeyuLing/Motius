from __future__ import annotations
import json
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ..bodymodels.smpl_skeleton import JOINTS_WEIGHTS_SMPLH_JOINTS, SMPLMesh, SMPLSkeleton
from ..datasets.geometry import angle_axis_to_rotation_matrix, rot6d_to_rotation_matrix, rotation_matrix_to_rot6d
from ..utils.loaders import load_object
from ..utils.motion_augmentation import rotation_around_y_axis, swap_left_right
from ..utils.visualize2d import visualize_skeleton
from .metric import compute_jitter


def get_local_trans(transl, global_orient_R):
    local_trans = torch.einsum("...lij,...li->...lj", global_orient_R, transl)
    return local_trans


def read_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_local_transl_vel(transl, global_orient_R, fps):
    """
    transl velocity is in local coordinate (or, SMPL-coord)
    Args:
        transl: (*, L, 3)
        global_orient: (*, L, 3, 3)
    Returns:
        transl_vel: (*, L, 3)
    """
    transl_vel = transl[..., 1:, :] - transl[..., :-1, :]  # (B, L-1, 3)
    transl_vel = torch.cat([torch.zeros_like(transl_vel[..., :1, :]), transl_vel], dim=-2)  # (B, L, 3)  last-padding
    transl_vel = transl_vel * fps

    # v_local = R^T @ v_global
    local_transl_vel = torch.einsum("...lij,...li->...lj", global_orient_R, transl_vel)
    return local_transl_vel


smpl_skeleton_cpu = SMPLSkeleton()


def interpolate_to_frame(tensor, target_frame):
    # tensor: (seqlen, ...)
    if tensor.shape[0] == target_frame:
        return tensor
    origin_shape = tensor.shape
    if len(origin_shape) == 3:
        tensor = tensor.reshape(origin_shape[0], -1)
    interp = torch.nn.functional.interpolate(
        tensor.transpose(0, 1)[None], size=target_frame, mode="linear", align_corners=False
    )[0].T
    if len(origin_shape) == 3:
        interp = interp.reshape(target_frame, *origin_shape[1:])
    return interp


def load_raw_motion_from_npz(
    filename,
    min_len=15,
    max_len=64,
    split="train",
    motion_rep="abstrans",
    aug_mirror=False,
    aug_rotation=False,
    aug_start=False,
    aug_speed=False,
    padding2seconds=False,
):
    data = dict(np.load(filename))

    if aug_mirror:
        flag_mirror = np.random.rand() < 0.5
        if flag_mirror:
            data = swap_left_right(data)

    if aug_rotation:
        j_shaped = smpl_skeleton_cpu.compute_j_shaped(torch.FloatTensor(data["betas"][0].reshape(1, -1)))
        rotation_angle = np.random.rand() * 2 * np.pi
        data = rotation_around_y_axis(data, rotation_angle, j_shaped)

    if aug_start:
        flag_aug_start = np.random.rand() < 0.5
        if flag_aug_start:
            start_frame = np.random.randint(0, data["poses"].shape[0] - min_len)
            data["poses"] = data["poses"][start_frame:]
            data["trans"] = data["trans"][start_frame:]

    for key in ["poses", "trans", "betas"]:
        data[key] = torch.FloatTensor(data[key])

    poses = data["poses"]
    if len(poses.shape) == 2:
        poses = poses.reshape(poses.shape[0], -1, 3)
    if "hand" not in motion_rep and "full" not in motion_rep:
        poses = poses[:, :22]  # 只保留躯干部分
    rotations = angle_axis_to_rotation_matrix(poses)
    rot6d = rotation_matrix_to_rot6d(rotations)
    transl = data["trans"]
    if split == "train" and rot6d.shape[1] > max_len:
        start = np.random.randint(0, rot6d.shape[1] - min_len)
        end = start + max_len
        rot6d = rot6d[start:end]
        transl = transl[start:end]
        # 第一帧的高度不是在0
        # transl = transl - transl[:1]
    if aug_speed:
        target_frame = int(rot6d.shape[0] * np.random.uniform(0.8, 1.2))
        rot6d = interpolate_to_frame(rot6d, target_frame)
        rotations = rot6d_to_rotation_matrix(rot6d)
        rot6d = rotation_matrix_to_rot6d(rotations)
        transl = interpolate_to_frame(transl, target_frame)

    if padding2seconds:
        PADDING_FRAMES = 60
        if rot6d.shape[0] % PADDING_FRAMES != 0:
            padding_frames = PADDING_FRAMES - rot6d.shape[0] % PADDING_FRAMES
            start_rot_delta = (rot6d[0] - rot6d[1]).abs().mean()
            end_rot_delta = (rot6d[-1] - rot6d[-2]).abs().mean()
            if start_rot_delta > end_rot_delta:
                # 在末尾拼接
                padding_rot6d = rot6d[-1:].repeat(padding_frames, 1, 1)
                rot6d = torch.cat([rot6d, padding_rot6d], dim=0)
                padding_transl = transl[-1:].repeat(padding_frames, 1)
                transl = torch.cat([transl, padding_transl], dim=0)
            else:
                # 在开头拼接
                padding_rot6d = rot6d[0:1].repeat(padding_frames, 1, 1)
                rot6d = torch.cat([padding_rot6d, rot6d], dim=0)
                padding_transl = transl[0:1].repeat(padding_frames, 1)
                transl = torch.cat([padding_transl, transl], dim=0)

    data["rot6d"] = rot6d
    data["trans"] = transl
    data["shapes"] = data["betas"][0]
    return data


def length_to_mask(lengths, max_len):
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


class ExampleDataset:
    def __init__(
        self,
        roots,
        min_len=15,
        max_len=64,
        motion_dir="motions",
        max_item=None,
        n_fold=-1,
        split="train",
        text_dir=None,
        text_ext=".pt",
        text_max_len=128,
        skip_mirror=False,
        round_frames=4,
        local_rank=None,
        global_size=None,
        append_mirror=True,
        ext=".npz",
        always_max_length=False,
        body_only=False,
        motion_rep="abstrans",
        aug_mirror=False,
        aug_rotation=False,
        aug_speed=False,
        aug_start=False,
        padding2seconds=False,
    ):
        self.roots = roots
        self.max_len = max_len
        self.always_max_length = always_max_length
        self.body_only = body_only
        self.motion_rep = motion_rep
        self.text_max_len = text_max_len
        self.min_len = min_len
        self.round_frames = round_frames
        self.filenames = []
        self.aug_mirror = aug_mirror
        self.aug_rotation = aug_rotation
        self.aug_speed = aug_speed
        self.aug_start = aug_start
        self.padding2seconds = padding2seconds
        for cfg in self.roots:
            root = cfg["root"]
            ignore_dir = cfg.get("ignore_dir", [])
            if "checklist" in cfg:
                checklist = cfg["checklist"]
                if isinstance(checklist, str):
                    with open(os.path.join(root, checklist), "r") as f:
                        checklist = json.load(f)
                if "valid" in checklist:
                    validlist = checklist["valid"]
                    if "subdir" in cfg:
                        validlist = [item for item in validlist if os.path.dirname(item["filename"]) in cfg["subdir"]]
                    filenames = []
                    for iindex, item in enumerate(validlist):
                        if iindex % 10000 == 0:
                            print(
                                f"[{self.__class__.__name__}] {iindex} / {len(validlist)} files loaded => {len(filenames)} files loaded"
                            )
                        dir_base = item["filename"].replace(".npz", ext)
                        dirname = os.path.dirname(dir_base)
                        filename = os.path.join(root, motion_dir, dir_base)
                        if not os.path.exists(filename):
                            print(f"{filename} not found")
                            continue
                        filenames.append(
                            {
                                "filename": filename,
                                "dirname": dirname,
                            }
                        )
                        if text_dir is not None:
                            text_filename = os.path.join(root, text_dir, item["filename"].replace(".npz", text_ext))
                            if not os.path.exists(text_filename):
                                print(f"{text_filename} not found")
                                breakpoint()
                            filenames[-1]["text_filename"] = text_filename
                        if append_mirror:
                            filename_m = os.path.join(root, motion_dir, "M_" + dir_base)
                            if not os.path.exists(filename_m):
                                print(f"{filename_m} not found")
                                continue
                            filenames.append(
                                {
                                    "filename": filename_m,
                                    "dirname": "M_" + dirname,
                                }
                            )
                            if text_dir is not None:
                                text_filename = os.path.join(
                                    root, text_dir, "M_" + item["filename"].replace(".npz", text_ext)
                                )
                                if not os.path.exists(text_filename):
                                    print(f"{text_filename} not found")
                                    breakpoint()
                                filenames[-1]["text_filename"] = text_filename

                    print(f"[{self.__class__.__name__}] {len(filenames)} / {len(validlist)} files loaded")
                elif "file_list" in checklist:
                    filenames = []
                    for item in checklist["file_list"]:
                        filenames.append(
                            {
                                "filename": os.path.join(root, motion_dir, item + ext),
                                "dirname": os.path.dirname(item),
                            }
                        )
                        if text_dir is not None:
                            text_filename = os.path.join(root, text_dir, item + text_ext)
                            if not os.path.exists(text_filename):
                                print(f"{text_filename} not found")
                                breakpoint()
                            filenames[-1]["text_filename"] = text_filename
                else:
                    raise ValueError(f"Invalid checklist: {checklist}")
                self.filenames.extend(filenames)
            elif "namelist" in cfg:
                namelist = cfg["namelist"]
                with open(os.path.join(root, namelist), "r") as f:
                    namelist = f.readlines()
                namelist = [item.strip() for item in namelist]
                filenames = []
                for item in namelist:
                    filenames.append(
                        {
                            "filename": os.path.join(root, motion_dir, item.replace(".fbx", "") + ext),
                            "dirname": os.path.dirname(item.split(".")[0]),
                        }
                    )
                self.filenames.extend(filenames)
                print(f"[{self.__class__.__name__}] {len(filenames)} / {len(namelist)} files loaded")
            else:
                assert os.path.exists(os.path.join(root, motion_dir)), f"{root} is not a valid path"
                dirnames = sorted(os.listdir(os.path.join(root, motion_dir)))
                for dirname in dirnames:
                    if dirname in ignore_dir:
                        continue
                    if skip_mirror and "M_" in dirname:
                        continue
                    filenames = sorted(os.listdir(os.path.join(root, motion_dir, dirname)))
                    filenames = [
                        {
                            "filename": os.path.join(root, motion_dir, dirname, filename),
                            "dirname": dirname,
                        }
                        for filename in filenames
                    ]
                    self.filenames.extend(filenames)
        print(f"[{self.__class__.__name__}] {len(self.filenames)} files loaded")

        self.split = split
        if n_fold > 0:
            if split == "train":
                self.filenames = [self.filenames[i] for i in range(len(self.filenames)) if i % n_fold != 0]
            else:
                self.filenames = [self.filenames[i] for i in range(len(self.filenames)) if i % n_fold == 0]
            print(f"[{self.__class__.__name__}] get {len(self.filenames)} from n_fold={n_fold} and split ={split}")

        # # Apply distributed sampling if local_rank and global_size are provided
        # if local_rank is not None and global_size is not None:
        #     # Distribute data across ranks
        #     self.filenames = self.filenames[local_rank::global_size]
        #     print(f'[{self.__class__.__name__}] Rank {local_rank}/{global_size}: processing {len(self.filenames)} files')

        self.max_item = max_item
        self.train_iterations = len(self.filenames)

    def __len__(self):
        if self.max_item is not None:
            return self.max_item
        return self.train_iterations

    def set_train_iterations(self, train_iterations):
        self.train_iterations = train_iterations

    @staticmethod
    def padding_or_clip(data, max_len, round_frames):
        current_length = max_len
        for key in ["rot6d", "trans"]:
            length = data[key].shape[0]
            # ATTN: 这里的长度要进行截断，保证是N的倍数
            length = length // round_frames * round_frames
            if length > max_len:
                data[key] = data[key][:max_len]
            else:
                current_length = length
                data[key] = torch.cat(
                    [data[key][:length], torch.zeros(max_len - length, *data[key].shape[1:]) + data[key][-1:]], dim=0
                )
        data["length"] = current_length
        return data

    def _padding_or_truncate_tensor(self, tensor, target_length):
        if tensor.shape[0] < target_length:
            tensor = torch.cat([tensor, torch.zeros(target_length - tensor.shape[0], *tensor.shape[1:]) + tensor[-1:]])
        elif tensor.shape[0] > target_length:
            tensor = tensor[:target_length]
        return tensor

    @staticmethod
    def data_from_npz(filename, min_len, max_len, round_frames):
        data = load_raw_motion_from_npz(filename, min_len, max_len, round_frames)
        ret = {
            "rot6d": data["rot6d"],
            "shapes": data["shapes"],
            "trans": data["trans"],
            "data_meta": {
                "input_filename": filename,
            },
        }
        ret = ExampleDataset.padding_or_clip(ret, max_len, round_frames)
        return ret

    def load_all_data(self, local_rank=0, global_size=1):
        data_all = []
        for index in tqdm(
            range(local_rank, len(self), global_size),
            desc=f"[{self.__class__.__name__}] Loading all data",
            disable=global_size > 1,
        ):
            data = self[index]
            data["input_filename"] = data["data_meta"]["input_filename"]
            data_all.append(data)
        return data_all

    def load_text_data(self, filename):
        # >>> data['result'][0]['text_embedding'].keys()
        # dict_keys(['text_vec_raw', 'text_ctxt_raw', 'text_ctxt_raw_length'])
        # >>> data['result'][0]['text_embedding']['text_ctxt_raw'].shape
        # torch.Size([1, 9, 4096])
        # >>> data['result'][0]['text_embedding']['text_ctxt_raw_length'].shape
        # torch.Size([1])
        # >>> data['result'][0]['text_embedding']['text_ctxt_raw_length']
        # tensor([9]
        # 默认只load first text
        load_first_only = True
        text_data = torch.load(filename)
        if load_first_only:
            text_data = text_data["result"][0]
        else:
            breakpoint()
        ret = {
            "caption": text_data["caption"],
            "text": text_data["caption"],
            "text_vec_raw": text_data["text_embedding"]["text_vec_raw"][0],
            "text_ctxt_raw": text_data["text_embedding"]["text_ctxt_raw"][0],
            "text_ctxt_raw_length": text_data["text_embedding"]["text_ctxt_raw_length"],
        }
        ret["text_ctxt_raw"] = self._padding_or_truncate_tensor(ret["text_ctxt_raw"], self.text_max_len)
        return ret

    def __getitem__(self, index):
        assert len(self.filenames) > 0, f"{self.filenames}"
        filename = self.filenames[index % len(self.filenames)]["filename"]
        dirname = self.filenames[index % len(self.filenames)]["dirname"]
        data = load_raw_motion_from_npz(
            filename,
            self.min_len,
            self.max_len,
            self.split,
            motion_rep=self.motion_rep,
            aug_mirror=self.aug_mirror,
            aug_rotation=self.aug_rotation,
            aug_speed=self.aug_speed,
            aug_start=self.aug_start,
            padding2seconds=self.padding2seconds,
        )
        ret = {
            "rot6d": data["rot6d"],
            "shapes": data["shapes"],
            "trans": data["trans"],
            "index": index,
            "data_meta": {"dirname": dirname, "input_filename": filename, "motion_fps": 30},
        }
        ret["true_length"] = min(data["rot6d"].shape[0], self.max_len)
        ret = self.padding_or_clip(ret, self.max_len, self.round_frames)
        if self.always_max_length:
            ret["length"] = self.max_len
        else:
            ret["length"] = ret["true_length"]

        # 加载文本
        if "text_filename" in self.filenames[index % len(self.filenames)]:
            text_filename = self.filenames[index % len(self.filenames)]["text_filename"]
            text_data = self.load_text_data(text_filename)
            ret.update(text_data)
        #
        return ret


class DebugRot6dDataset(ExampleDataset):
    def __init__(self, mean_std_dir=None, overparameter=False, **kwargs):
        super().__init__(**kwargs)
        self.overparameter = overparameter
        if mean_std_dir is not None:
            print(f"[{self.__class__.__name__}] load mean and std from {mean_std_dir}")
            mean, std = self.load_mean_std(mean_std_dir)
            self.mean = mean
            self.std = std
        else:
            self.mean = None
            self.std = None

    @staticmethod
    def load_mean_std(mean_std_dir):
        mean_std = read_json(mean_std_dir)
        rot6d_mean = torch.FloatTensor(mean_std["rot6d"]["mean"])[:22, :]
        rot6d_std = torch.FloatTensor(mean_std["rot6d"]["std"])[:22, :]
        rot6d_mean = rot6d_mean
        rot6d_std = rot6d_std
        trans_mean = torch.FloatTensor(mean_std["trans"]["mean"])[None]
        trans_std = torch.FloatTensor(mean_std["trans"]["std"])[None]
        mean = torch.cat(
            [
                rot6d_mean.reshape(1, -1),
                trans_mean.reshape(1, -1),
            ],
            dim=-1,
        )
        std = torch.cat(
            [
                rot6d_std.reshape(1, -1),
                trans_std.reshape(1, -1),
            ],
            dim=-1,
        )
        assert (std > 1e-3).all(), f"std is too small: {std.min()}"
        return mean, std

    def __getitem__(self, index):
        filename = self.filenames[index % len(self.filenames)]["filename"]
        dirname = self.filenames[index % len(self.filenames)]["dirname"]
        motion = np.load(filename)
        poses = torch.FloatTensor(motion["poses"])
        poses = poses.reshape(poses.shape[0], -1, 3)
        rotations = angle_axis_to_rotation_matrix(poses)
        rot6d = rotation_matrix_to_rot6d(rotations)
        transl = torch.FloatTensor(motion["trans"])
        motion = torch.cat(
            [
                rot6d[:, :22].reshape(rot6d.shape[0], -1),
                transl,
            ],
            dim=-1,
        )
        # 有一些超出max len的
        true_length = min(motion.shape[0], self.max_len)
        motion = self._padding_or_truncate_tensor(motion, self.max_len)
        if self.mean is not None:
            motion = (motion - self.mean) / self.std
        ret = {
            "motion": motion,
            "text": "",
            "index": index,
            "true_length": true_length,
            "data_meta": {"input_filename": filename, "motion_fps": 30},
        }
        if self.always_max_length:
            ret["length"] = self.max_len
        else:
            ret["length"] = true_length
        if "text_filename" in self.filenames[index % len(self.filenames)]:
            text_filename = self.filenames[index % len(self.filenames)]["text_filename"]
            text_data = self.load_text_data(text_filename)
            ret.update(text_data)
        return ret


class VAEDataset(ExampleDataset):
    def __init__(self, mean_std_dir=None, **kwargs):
        super().__init__(**kwargs)

    def _padding_or_truncate_tensor(self, tensor, target_length):
        if tensor.shape[0] < target_length:
            tensor = torch.cat([tensor, torch.zeros(target_length - tensor.shape[0], *tensor.shape[1:]) + tensor[-1:]])
        elif tensor.shape[0] > target_length:
            tensor = tensor[:target_length]
        return tensor

    def __getitem__(self, index):
        filename = self.filenames[index % len(self.filenames)]["filename"]
        dirname = self.filenames[index % len(self.filenames)]["dirname"]
        motion = torch.FloatTensor(np.load(filename))
        motion = self._padding_or_truncate_tensor(motion, self.max_len)
        return {
            "motion": motion,
            "text": "",
            "index": index,
            "length": self.max_len,
            "data_meta": {"input_filename": filename, "motion_fps": 30},
        }


class ExampleO6DPDataset(ExampleDataset):
    def __init__(
        self,
        mean_std_dir=None,
        disable_vel=False,
        disablerifke=False,
        disablefoot=False,
        disable_trans=False,
        disable_all_rot=False,
        disable_all_body=False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.disable_vel = disable_vel
        self.disablerifke = disablerifke
        self.disablefoot = disablefoot
        self.disable_trans = disable_trans
        self.disable_all_rot = disable_all_rot
        self.disable_all_body = disable_all_body
        assert mean_std_dir is not None, f"mean_std_dir is None"
        self.mean = torch.from_numpy(np.load(os.path.join(mean_std_dir, "Mean.npy"))).float()
        self.std = torch.from_numpy(np.load(os.path.join(mean_std_dir, "Std.npy"))).float()
        self.mean = self._reformulate_motion_dimension(self.mean.unsqueeze(0)).squeeze(0)
        self.std = self._reformulate_motion_dimension(self.std.unsqueeze(0)).squeeze(0)

    def _reformulate_motion_dimension(self, motion):
        L, D = motion.shape
        dev, dtype = motion.device, motion.dtype

        base_dim = 10  # 4(ry/xz/y) + 6(root o6d)
        full_o6d_dim = 51 * 6
        full_rifke_dim = 52 * 3
        full_rifke_vel_dim = 52 * 3
        foot_dim = 4

        if self.disable_all_rot:
            return motion[:, :4]
        if self.disable_all_body:
            return motion[:, :10]

        def z(cols: int):
            return torch.zeros(L, cols, dtype=dtype, device=dev)

        if D == 632:
            o6d_start = base_dim
            rifke_start = base_dim + full_o6d_dim
            rifke_vel_start = rifke_start + full_rifke_dim
            foot_start = rifke_vel_start + full_rifke_vel_dim
            append_list = []
            if self.disable_trans:
                append_list.append(motion[:, 4:base_dim])  # 10
            else:
                append_list.append(motion[:, :base_dim])  # 10

            append_list.append(motion[:, o6d_start : o6d_start + 21 * 6])  # 21*6
            if not self.disablerifke:
                append_list.append(motion[:, rifke_start : rifke_start + 22 * 3])  # 22*3
            if not self.disable_vel:
                append_list.append(motion[:, rifke_vel_start : rifke_vel_start + 22 * 3])  # 22*3
            if not self.disablefoot:
                append_list.append(motion[:, foot_start : foot_start + foot_dim])  # 4
            return torch.cat(append_list, dim=1)
        elif D == 272:
            if self.disablefoot and self.disable_vel and self.disablerifke and self.disable_trans:
                return motion[:, 4 : 272 - 4 - 22 * 3 - 22 * 3]
            if self.disablefoot and self.disable_vel and self.disablerifke:
                return motion[:, : 272 - 4 - 22 * 3 - 22 * 3]
            if self.disablefoot and self.disable_vel:
                return motion[:, : 272 - 4 - 22 * 3]
            if self.disablefoot:
                return motion[:, : 272 - 4]
            return motion
        else:
            print(motion.shape)
            raise ValueError(f"Motion shape is {motion.shape}, expected 272 or 632")

    def _padding_or_truncate_tensor(self, tensor, target_length):
        if tensor.shape[0] < target_length:
            tensor = torch.cat([tensor, torch.zeros(target_length - tensor.shape[0], *tensor.shape[1:]) + tensor[-1:]])
        elif tensor.shape[0] > target_length:
            tensor = tensor[:target_length]
        return tensor

    def __getitem__(self, index):
        filename = self.filenames[index % len(self.filenames)]["filename"]
        dirname = self.filenames[index % len(self.filenames)]["dirname"]
        motion = torch.from_numpy(np.load(filename))
        motion = self._reformulate_motion_dimension(motion)
        std_zero = self.std < 1e-3
        std = torch.where(std_zero, torch.ones_like(self.std), self.std)
        motion = (motion - self.mean) / std
        true_length = motion.shape[0]
        motion = self._padding_or_truncate_tensor(motion, self.max_len)
        ret = {
            "motion": motion,
            "text": "",
            "index": index,
            "true_length": true_length,
            "data_meta": {"input_filename": filename, "motion_fps": 30},
        }
        if self.always_max_length:
            ret["length"] = self.max_len
        else:
            ret["length"] = true_length
        if "text_filename" in self.filenames[index % len(self.filenames)]:
            text_filename = self.filenames[index % len(self.filenames)]["text_filename"]
            text_data = self.load_text_data(text_filename)
            ret.update(text_data)
        return ret


def compute_metrics(keypoints3d_gt, transl_gt, keypoints3d_gen, transl_gen):
    k3d_center = keypoints3d_gt - transl_gt[..., None, :]
    k3d_gen_center = keypoints3d_gen - transl_gen[..., None, :]
    dist_wotrans = torch.norm(k3d_center - k3d_gen_center, dim=-1).mean()
    dist = torch.norm(keypoints3d_gt - keypoints3d_gen, dim=-1).mean()
    vel_center = k3d_center[..., 1:, :, :] - k3d_center[..., :-1, :, :]
    vel_gen_center = k3d_gen_center[..., 1:, :, :] - k3d_gen_center[..., :-1, :, :]
    vel_wotrans = torch.norm(vel_center - vel_gen_center, dim=-1).mean()
    acc_center = vel_center[..., 1:, :, :] - vel_center[..., :-1, :, :]
    acc_gen_center = vel_gen_center[..., 1:, :, :] - vel_gen_center[..., :-1, :, :]
    acc_wotrans = torch.norm(acc_center - acc_gen_center, dim=-1).mean()
    # 完整的
    vel = keypoints3d_gt[..., 1:, :, :] - keypoints3d_gt[..., :-1, :, :]
    vel_gen = keypoints3d_gen[..., 1:, :, :] - keypoints3d_gen[..., :-1, :, :]
    vel_error = torch.norm(vel - vel_gen, dim=-1).mean()
    acc = vel[..., 1:, :, :] - vel[..., :-1, :, :]
    acc_gen = vel_gen[..., 1:, :, :] - vel_gen[..., :-1, :, :]
    acc_error = torch.norm(acc - acc_gen, dim=-1).mean()

    return {
        "dist_body": dist_wotrans,
        "vel_body": vel_wotrans,
        "acc_body": acc_wotrans,
        "dist_global": dist,
        "vel_global": vel_error,
        "acc_global": acc_error,
    }


def rotation_to_angle(rotations):
    """将旋转矩阵转换为角度

    Args:
        rotations (torch.Tensor): 旋转矩阵，形状为 (frames, joints, 3, 3)
    """
    traces = rotations[..., 0, 0] + rotations[..., 1, 1] + rotations[..., 2, 2]
    return torch.acos(torch.clamp((traces - 1) / 2, -1.0, 1.0)) * 180 / torch.pi


class AbsoluateMotionVAEPipeline(nn.Module):
    def __init__(
        self,
        network_module,
        network_module_args,
        loss_weight={},
        loss_vertices_type="smooth_l1",
        vertices_loss_start_iteration=-1,
        max_trans_vel=0.1,
        end_effect_index=[7, 8, 10, 11],
        end_effect_threshold=0.005,
        fps=30,
        mean_std=None,
        clip_max_length=True,
        mask_loss=True,
        motion_representation="abstrans",
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.fps = fps
        self.loss_vertices_type = loss_vertices_type
        self.clip_max_length = clip_max_length
        self.mask_loss = mask_loss
        self.motion_rep = motion_representation
        self.network = load_object(network_module, network_module_args)
        self.body_model = SMPLSkeleton()
        self.mesh_model = SMPLMesh()
        self.load_mean_std(mean_std)
        self.vertices_loss_start_iteration = vertices_loss_start_iteration
        self.max_trans_vel = max_trans_vel
        print(f"[{self.__class__.__name__}] clip the max trans vel to {self.max_trans_vel * 3}")
        self.end_effect_index = end_effect_index
        self.end_effect_threshold = end_effect_threshold
        print(f"[{self.__class__.__name__}] set end_effect_index to {self.end_effect_index}")
        print(f"[{self.__class__.__name__}] set end_effect_threshold to {self.end_effect_threshold}")
        print(f"[{self.__class__.__name__}] set vertices_loss_start_iteration to {self.vertices_loss_start_iteration}")
        if mean_std is not None:
            self.use_mean_std = True
        else:
            self.use_mean_std = False
            self.rot6d_mean = torch.zeros(1)
            self.rot6d_std = torch.ones(1)
            self.trans_mean = torch.zeros(1)
            self.trans_std = torch.ones(1)
            self.shapes_mean = torch.zeros(1)
            self.shapes_std = torch.ones(1)
            self.mean_std = None
        # 增加一个计算均值方差的；传入到网络的时候进行归一化操作
        self.epoch = 0
        self.global_iteration = 0

    def load_mean_std(self, mean_std):
        with open(mean_std, "r") as f:
            mean_std = json.load(f)
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

    def _encode_abstrans(self, batch):
        # 这个函数只考虑躯干的rot6d
        rot6d = batch["rot6d"][:, :, :22]
        rot6d = rot6d - self.rot6d_mean[:, :, :22]
        trans = batch["trans"]
        trans = trans - self.trans_mean
        shapes = batch["shapes"]
        shapes = shapes - self.shapes_mean
        bs, seq_len = rot6d.shape[:2]
        rot6d = rot6d.reshape(bs, seq_len, -1)
        shapes = shapes[:, None].repeat(1, seq_len, 1)
        return {
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }

    def _encode_reltrans(self, batch):
        rot6d = batch["rot6d"][:, :, :22]
        rot6d = rot6d - self.rot6d_mean[:, :, :22]
        trans = batch["trans"]
        trans = trans - self.trans_mean
        trans_vel = (trans[:, 1:] - trans[:, :-1]) * self.fps
        trans_vel = torch.cat([trans_vel, trans_vel[:, [-1]]], dim=1)
        trans = trans_vel
        shapes = batch["shapes"]
        shapes = shapes - self.shapes_mean
        bs, seq_len = rot6d.shape[:2]
        rot6d = rot6d.reshape(bs, seq_len, -1)
        shapes = shapes[:, None].repeat(1, seq_len, 1)
        return {
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }

    def _encode_truncreltrans(self, batch, max_trans_vel=0.1):
        rot6d = batch["rot6d"][:, :, :22]
        rot6d = rot6d - self.rot6d_mean[:, :, :22]
        trans = batch["trans"]
        trans = trans - self.trans_mean
        trans_vel = (trans[:, 1:] - trans[:, :-1]) / max_trans_vel
        trans_vel = torch.clamp(trans_vel, -3.0, 3.0)
        trans_vel = torch.cat([trans_vel, trans_vel[:, [-1]]], dim=1)
        trans = trans_vel
        shapes = batch["shapes"]
        shapes = shapes - self.shapes_mean
        bs, seq_len = rot6d.shape[:2]
        rot6d = rot6d.reshape(bs, seq_len, -1)
        shapes = shapes[:, None].repeat(1, seq_len, 1)
        return {
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }

    def _encode_truncreltrans_hand(self, batch, max_trans_vel=0.1):
        rot6d = batch["rot6d"][:, :, :22]
        rot6d_hand = batch["rot6d"][:, :, 22:]
        rot6d = rot6d - self.rot6d_mean[:, :, :22]
        rot6d_hand = rot6d_hand - self.rot6d_mean[:, :, 22:]
        trans = batch["trans"]
        trans = trans - self.trans_mean
        trans_vel = (trans[:, 1:] - trans[:, :-1]) / max_trans_vel
        trans_vel = torch.clamp(trans_vel, -3.0, 3.0)
        trans_vel = torch.cat([trans_vel, trans_vel[:, [-1]]], dim=1)
        trans = trans_vel
        shapes = batch["shapes"]
        shapes = shapes - self.shapes_mean
        bs, seq_len = rot6d.shape[:2]
        rot6d = rot6d.reshape(bs, seq_len, -1)
        rot6d_hand = rot6d_hand.reshape(bs, seq_len, -1)
        shapes = shapes[:, None].repeat(1, seq_len, 1)
        return {
            "rot6d": rot6d,
            "rot6d_hand": rot6d_hand,
            "trans": trans,
            "shapes": shapes,
        }

    def _encode_relbodytrans(self, batch):
        rot6d = batch["rot6d"][:, :, :22]
        rot6d = rot6d - self.rot6d_mean[:, :, :22]
        trans = batch["trans"]
        trans = trans - self.trans_mean
        root_rotations = rot6d_to_rotation_matrix(rot6d[:, :, 0])
        trans_vel = get_local_transl_vel(trans, root_rotations, self.fps)
        trans = trans_vel
        shapes = batch["shapes"]
        shapes = shapes - self.shapes_mean
        bs, seq_len = rot6d.shape[:2]
        rot6d = rot6d.reshape(bs, seq_len, -1)
        shapes = shapes[:, None].repeat(1, seq_len, 1)
        return {
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }

    def _encode_rot6d(self, batch):
        rot6d = batch["rot6d"]
        rot6d = rot6d - self.rot6d_mean[:, :, :22]
        bs, seq_len = rot6d.shape[:2]
        rot6d = rot6d.reshape(bs, seq_len, -1)
        return {
            "rot6d": rot6d,
        }

    def encode_motion(self, batch, length=None):
        if self.motion_rep == "abstrans":
            return self._encode_abstrans(batch)
        elif self.motion_rep == "reltrans":
            return self._encode_reltrans(batch)
        elif self.motion_rep == "truncreltrans":
            return self._encode_truncreltrans(batch, self.max_trans_vel)
        elif self.motion_rep == "truncreltrans_hand":
            return self._encode_truncreltrans_hand(batch, self.max_trans_vel)
        elif self.motion_rep == "relbodytrans":
            return self._encode_relbodytrans(batch)
        elif self.motion_rep == "rot6d_hand_reltrans_endeffect":
            return self._encode_truncreltrans_hand(batch, self.max_trans_vel)
        elif self.motion_rep == "full":
            return self._encode_truncreltrans_hand(batch, self.max_trans_vel)
        elif self.motion_rep == "rot6d":
            return self._encode_rot6d(batch)
        else:
            raise ValueError(f"Unsupported motion representation: {self.motion_rep}")

    def _decode_abstrans(self, latent):
        rot6d = latent["rot6d"]
        trans = latent["trans"]
        shapes = latent["shapes"]
        bs, seq_len = rot6d.shape[:2]
        rot6d = rot6d.reshape(bs, seq_len, -1, 6)
        rot6d_hand = torch.zeros(bs, seq_len, 30, 6, device=rot6d.device, dtype=rot6d.dtype)
        rot6d = torch.cat([rot6d, rot6d_hand], dim=-2)
        rot6d = rot6d + self.rot6d_mean
        trans = trans.reshape(bs, seq_len, 3)
        trans = trans + self.trans_mean
        shapes = shapes.mean(dim=1)
        shapes = shapes + self.shapes_mean
        return {
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }

    def _decode_reltrans(self, latent):
        rot6d = latent["rot6d"]
        trans = latent["trans"]
        shapes = latent["shapes"]
        bs, seq_len = rot6d.shape[:2]
        rot6d = rot6d.reshape(bs, seq_len, -1, 6)
        rot6d_hand = torch.zeros(bs, seq_len, 30, 6, device=rot6d.device, dtype=rot6d.dtype)
        rot6d = torch.cat([rot6d, rot6d_hand], dim=-2)
        rot6d = rot6d + self.rot6d_mean
        #
        trans = torch.cumsum(trans, dim=1) / self.fps
        trans = trans + self.trans_mean
        shapes = shapes + self.shapes_mean
        shapes = shapes.mean(dim=1)
        return {
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }

    def _decode_truncreltrans(self, latent, max_trans_vel=0.1, use_shapes=False):
        rot6d_latent = latent["rot6d"]
        bs, seq_len = rot6d_latent.shape[:2]
        rot6d = rot6d_latent.reshape(bs, seq_len, -1, 6)
        if "rot6d_hand" in latent:
            rot6d_hand_latent = latent["rot6d_hand"]
            rot6d_hand = rot6d_hand_latent.reshape(bs, seq_len, -1, 6)
        else:
            rot6d_hand_latent = torch.zeros(bs, seq_len, 180, device=rot6d.device, dtype=rot6d.dtype)
            rot6d_hand = torch.zeros(bs, seq_len, 30, 6, device=rot6d.device, dtype=rot6d.dtype)
        rot6d = torch.cat([rot6d, rot6d_hand], dim=-2)
        rot6d = rot6d + self.rot6d_mean
        #
        trans_vel_latent = latent["trans"]
        trans_vel = trans_vel_latent * max_trans_vel
        trans = torch.cumsum(trans_vel, dim=1)
        trans = trans + self.trans_mean
        if "shapes" in latent:
            shapes = latent["shapes"]
        else:
            shapes = torch.zeros(rot6d.shape[0], rot6d.shape[1], 16, device=rot6d.device, dtype=rot6d.dtype)
        shapes = shapes + self.shapes_mean
        # 不要取均值
        # shapes = shapes.mean(dim=1)
        return {
            "rot6d_latent": rot6d_latent,
            "rot6d_hand_latent": rot6d_hand_latent,
            "shapes_latent": shapes,
            "trans_vel_latent": trans_vel_latent,
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }

    def _decode_rot6d_hand_reltrans_endeffect(self, latent, max_trans_vel):
        rot6d_latent = latent["rot6d"]
        bs, seq_len = rot6d_latent.shape[:2]
        rot6d = rot6d_latent.reshape(bs, seq_len, -1, 6)
        if "rot6d_hand" in latent:
            rot6d_hand_latent = latent["rot6d_hand"]
            rot6d_hand = rot6d_hand_latent.reshape(bs, seq_len, -1, 6)
        else:
            rot6d_hand_latent = torch.zeros(bs, seq_len, 180, device=rot6d.device, dtype=rot6d.dtype)
            rot6d_hand = torch.zeros(bs, seq_len, 30, 6, device=rot6d.device, dtype=rot6d.dtype)
        rot6d = torch.cat([rot6d, rot6d_hand], dim=-2)
        rot6d = rot6d + self.rot6d_mean
        #
        trans_vel_latent = latent["trans"]
        trans_vel = trans_vel_latent * max_trans_vel
        # end effect
        if "end_effect_vel" in latent:  # 网络输出的情况
            endeffect_vel_latent = latent["end_effect_vel"].reshape(bs, seq_len, -1, 3)
            endeffect_vel = endeffect_vel_latent * max_trans_vel
            endeffect_stationary = latent["end_effect_stationary"]
            # 网络输出的情况，需要更新一下
            # 训练的时候似乎不应该更新
            # 不然就会全靠这个东西来纠正了；似乎不太合理
            # 这个是一个纯local，面向最终输出的东西
        else:
            endeffect_vel_latent = None
            endeffect_vel = None
            endeffect_stationary = None

        trans = torch.cumsum(trans_vel, dim=1)
        trans = trans + self.trans_mean
        shapes = torch.zeros(rot6d.shape[0], rot6d.shape[1], 16, device=rot6d.device, dtype=rot6d.dtype)
        shapes = shapes + self.shapes_mean
        shapes = shapes.mean(dim=1)

        ret = {
            "rot6d_latent": rot6d_latent,
            "rot6d_hand_latent": rot6d_hand_latent,
            "trans_vel_latent": trans_vel_latent,
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }
        if "end_effect_vel" in latent:
            ret["endeffect_vel"] = endeffect_vel
            ret["endeffect_stationary"] = endeffect_stationary
        return ret

    def _decode_relbodytrans(self, latent):
        rot6d = latent["rot6d"]
        trans_vel_local = latent["trans"]
        shapes = latent["shapes"]
        bs, seq_len = rot6d.shape[:2]
        rot6d = rot6d.reshape(bs, seq_len, -1, 6)
        rot6d_hand = torch.zeros(bs, seq_len, 30, 6, device=rot6d.device, dtype=rot6d.dtype)
        rot6d = torch.cat([rot6d, rot6d_hand], dim=-2)
        rot6d = rot6d + self.rot6d_mean
        #
        rotations = rot6d_to_rotation_matrix(rot6d)
        rotations_root = rotations[:, :, 0]
        trans = MotionVAEPipeline.rollout_local_transl_vel(trans_vel_local / self.fps, rotations_root)
        trans = trans + self.trans_mean
        shapes = shapes + self.shapes_mean
        shapes = shapes.mean(dim=1)
        return {
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }

    def _decode_rot6d(self, latent):
        rot6d_latent = latent["rot6d"]
        rot6d = rot6d_latent.reshape(rot6d_latent.shape[0], rot6d_latent.shape[1], -1, 6)
        rot6d_hand = torch.zeros(rot6d.shape[0], rot6d.shape[1], 30, 6, device=rot6d.device, dtype=rot6d.dtype)
        rot6d = torch.cat([rot6d, rot6d_hand], dim=-2)
        rot6d = rot6d + self.rot6d_mean
        #
        trans = torch.zeros(rot6d.shape[0], rot6d.shape[1], 3, device=rot6d.device, dtype=rot6d.dtype)
        shapes = torch.zeros(rot6d.shape[0], rot6d.shape[1], 16, device=rot6d.device, dtype=rot6d.dtype)
        trans = trans + self.trans_mean
        shapes = shapes + self.shapes_mean
        shapes = shapes.mean(dim=1)
        return {
            "rot6d_latent": rot6d_latent,
            "rot6d": rot6d,
            "trans": trans,
            "shapes": shapes,
        }

    def decode_motion(self, latent):
        if self.motion_rep == "abstrans":
            return self._decode_abstrans(latent)
        elif self.motion_rep == "reltrans":
            return self._decode_reltrans(latent)
        elif self.motion_rep == "truncreltrans" or self.motion_rep == "truncreltrans_hand":
            # 复用truncreltrans的decode函数
            return self._decode_truncreltrans(latent, self.max_trans_vel)
        elif self.motion_rep == "full":
            return self._decode_truncreltrans(latent, self.max_trans_vel, use_shapes=True)
        elif self.motion_rep == "rot6d_hand_reltrans_endeffect":
            return self._decode_rot6d_hand_reltrans_endeffect(latent, self.max_trans_vel)
        elif self.motion_rep == "relbodytrans":
            return self._decode_relbodytrans(latent)
        elif self.motion_rep == "rot6d":
            return self._decode_rot6d(latent)
        else:
            raise ValueError(f"Unsupported motion representation: {self.motion_rep}")

    def calculate_end_effect(self, keypoints3d):
        # 这里使用的是global的keypoints3d；必须要有trans
        end_effect = keypoints3d[:, :, self.end_effect_index, :]
        end_effect_vel = end_effect[:, 1:] - end_effect[:, :-1]
        # 使用repeat padding；补上最后一帧的速度
        end_effect_vel_padding = torch.cat([end_effect_vel, end_effect_vel[:, [-1]]], dim=1)
        end_effect_vel_norm = torch.norm(end_effect_vel_padding, dim=-1)
        flag_static = end_effect_vel_norm < self.end_effect_threshold
        return {
            "end_effect_vel": end_effect_vel_padding,
            "end_effect_stationary": flag_static,
        }

    def forward_body(self, latent, forward_vertices=True):
        rot6d = latent["rot6d"]
        transl = latent["trans"]
        shapes = latent["shapes"]

        if rot6d.shape[-2] == 22:
            rot6d_hand = torch.zeros((rot6d.shape[0], rot6d.shape[1], 30, 6), device=rot6d.device, dtype=rot6d.dtype)
            rot6d_hand[..., 0] = 1.0
            rot6d_hand[..., 3] = 1.0
            rot6d = torch.cat([rot6d, rot6d_hand], dim=-2)

        if len(shapes.shape) == 2:
            # 增加一个时间维度
            shapes = shapes[:, None]

        if shapes.shape[1] == 1:
            shapes = shapes.repeat(1, rot6d.shape[1], 1)

        rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
        transl_flat = transl.reshape(transl.shape[0] * transl.shape[1], 3)
        shapes_flat = shapes.reshape(shapes.shape[0] * shapes.shape[1], -1)

        params = {
            "rot6d": rot6d_flat,
            "trans": transl_flat,
            "shapes": shapes_flat,
        }

        out_keypoints = self.body_model(params)
        ret = {}

        params_woshapes = params.copy()
        params_woshapes["shapes"] = torch.zeros_like(params_woshapes["shapes"])
        params_woshapes_wotrans = params_woshapes.copy()
        params_woshapes_wotrans["trans"] = torch.zeros_like(params_woshapes_wotrans["trans"])
        out_keypoints_woshapes_wotrans = self.body_model(params_woshapes_wotrans)["keypoints3d"]
        out_keypoints_woshapes_wotrans = out_keypoints_woshapes_wotrans.reshape(rot6d.shape[0], rot6d.shape[1], -1, 3)

        rotations = rot6d_to_rotation_matrix(rot6d)

        ret["params_decode"] = {
            "rot6d": rot6d,
            "rotations": rotations,
            "trans": transl,
            "shapes": shapes,
            "keypoints3d": out_keypoints["keypoints3d"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3),
            "keypoints3d_woshapes_wotrans": out_keypoints_woshapes_wotrans,
        }
        ret["params_decode"].update(self.calculate_end_effect(ret["params_decode"]["keypoints3d"]))
        if (
            forward_vertices
            and not self.training
            or (self.vertices_loss_start_iteration >= 0 and self.global_iteration >= self.vertices_loss_start_iteration)
        ):
            out = self.mesh_model(params)
            ret["params_decode"]["vertices_wotrans"] = out["vertices_wotrans"].reshape(
                rot6d.shape[0], rot6d.shape[1], -1, 3
            )
            ret["params_decode"]["vertices"] = out["vertices"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3)
        return ret

    def compute_loss(self, output, batch, length):
        length_mask = length_to_mask(length, batch["rot6d"].shape[1])
        length_mask_item = max(length_mask.sum(), 1)
        length_mask_downsample = length_to_mask(
            length // self.network.downsample, batch["rot6d"].shape[1] // self.network.downsample
        )
        length_mask_downsample_item = max(length_mask_downsample.sum(), 1)

        # KL Loss: 在特征空间计算
        loss_dict = {}
        metric_dict = {}
        per_sample_loss = {}
        loss_kl = output["q_z"].kl().mean(dim=-1)
        assert loss_kl.shape[1] == batch["rot6d"].shape[1] // self.network.downsample, "You use wrong length mask"
        per_sample_loss["loss_kl"] = (loss_kl.detach() * length_mask_downsample).sum(dim=-1) / (
            length_mask_downsample.sum(dim=-1)
        )
        loss_kl = (loss_kl * length_mask_downsample).sum() / length_mask_downsample_item
        loss_dict["loss_kl"] = loss_kl

        # Loss on latent space
        if self.motion_rep == "truncreltrans":
            output_decode_keys = ["rot6d_latent", "trans_vel_latent"]
            params_decode_keys = ["trans", "vertices_wotrans"]
        elif self.motion_rep == "truncreltrans_hand":
            output_decode_keys = ["rot6d_latent", "rot6d_hand_latent", "trans_vel_latent"]
            params_decode_keys = ["trans", "vertices_wotrans"]
        elif self.motion_rep == "full":
            output_decode_keys = ["rot6d_latent", "rot6d_hand_latent", "trans_vel_latent", "shapes_latent"]
            params_decode_keys = ["trans", "vertices_wotrans"]
        elif self.motion_rep == "rot6d":
            output_decode_keys = ["rot6d_latent"]
            params_decode_keys = ["rot6d", "vertices_wotrans"]
        elif self.motion_rep == "rot6d_hand_reltrans_endeffect":
            output_decode_keys = ["rot6d_latent", "rot6d_hand_latent", "trans_vel_latent"]
            params_decode_keys = ["trans", "vertices_wotrans"]
            # batch['params_decode']['end_effect_vel'].shape
            # torch.Size([8, 360, 4, 3])
            # batch['params_decode']['flag_static'].shape
            # torch.Size([8, 360, 4])
            loss_end_effect_vel = (
                torch.nn.functional.mse_loss(
                    output["output_decode"]["endeffect_vel"],
                    batch["params_decode"]["end_effect_vel"],
                    reduction="none",
                )
                .mean(dim=-1)
                .mean(dim=-1)
            )
            target_endeffect_stationary = batch["params_decode"]["end_effect_stationary"]
            target_true = target_endeffect_stationary == True
            target_false = target_endeffect_stationary == False
            stationary_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                output["output_decode"]["endeffect_stationary"],
                target_endeffect_stationary.float(),
                reduction="none",
            )
            true_mask = target_true & length_mask[..., None]
            false_mask = target_false & length_mask[..., None]
            true_sum = max(true_mask.sum(), 1)
            false_sum = max(false_mask.sum(), 1)
            stationary_loss_true = (stationary_loss * true_mask).sum() / true_sum
            stationary_loss_false = (stationary_loss * false_mask).sum() / false_sum
            loss_dict["loss_end_effect_stationary_true"] = stationary_loss_true
            loss_dict["loss_end_effect_stationary_false"] = stationary_loss_false
            loss_dict["loss_end_effect_vel"] = (loss_end_effect_vel * length_mask).sum() / length_mask_item
            with torch.no_grad():
                pred_true = output["output_decode"]["endeffect_stationary"] > 0.5
                pred_false = output["output_decode"]["endeffect_stationary"] < 0.5
                pred_endeffect_stationary_true_acc = (pred_true == target_true).float()
                pred_endeffect_stationary_true_acc = (pred_endeffect_stationary_true_acc * true_mask).sum() / true_sum
                pred_endeffect_stationary_false_acc = (pred_false == target_false).float()
                pred_endeffect_stationary_false_acc = (
                    pred_endeffect_stationary_false_acc * false_mask
                ).sum() / false_sum
                metric_dict["endeffect_static_acc_true"] = pred_endeffect_stationary_true_acc
                metric_dict["endeffect_static_acc_false"] = pred_endeffect_stationary_false_acc
        else:
            raise ValueError(f"Unsupported motion representation: {self.motion_rep}")

        for key in output_decode_keys:
            # 这里进行latent space的loss计算
            # latent里的变量 一定都是 (B, T, C) 的形状
            loss_key = torch.nn.functional.mse_loss(
                output["output_decode"][key], batch["output_decode"][key], reduction="none"
            ).mean(dim=-1)
            per_sample_loss["loss_" + key] = (loss_key.detach() * length_mask).sum(dim=-1) / (
                length_mask.sum(dim=-1) + 1e-8
            )
            loss_key = (loss_key * length_mask).sum() / length_mask_item
            loss_dict["loss_" + key] = loss_key
        #
        if "vertices_wotrans" in batch["params_decode"]:
            loss_vertices_wotrans = (
                torch.nn.functional.mse_loss(
                    output["params_decode"]["vertices_wotrans"],
                    batch["params_decode"]["vertices_wotrans"],
                    reduction="none",
                )
                .mean(dim=-1)
                .mean(dim=-1)
            )
            loss_trans = torch.nn.functional.mse_loss(
                output["params_decode"]["trans"], batch["params_decode"]["trans"], reduction="none"
            ).mean(dim=-1)
            # 只在FK之后才进行监督
            loss_dict["loss_vertices"] = (loss_vertices_wotrans * length_mask).sum() / length_mask_item
            loss_dict["loss_trans"] = (loss_trans * length_mask).sum() / length_mask_item

            with torch.no_grad():
                vertices_gt_wotrans = batch["params_decode"]["vertices_wotrans"]
                vertices_pred_wotrans = output["params_decode"]["vertices_wotrans"]
                vertices_gt = batch["params_decode"]["vertices"]
                vertices_pred = output["params_decode"]["vertices"]
                metric_dict["mpvpe_aligned"] = (
                    (torch.norm(vertices_gt_wotrans - vertices_pred_wotrans, dim=-1).mean(dim=-1) * length_mask).sum()
                    / length_mask_item
                    * 1000
                )  # 定义为mm
                metric_dict["mpvpe"] = (
                    (torch.norm(vertices_gt - vertices_pred, dim=-1).mean(dim=-1) * length_mask).sum()
                    / length_mask_item
                    * 1000
                )  # 定义为mm

        # 计算metric
        with torch.no_grad():
            # 计算sliding
            # B, D, 3
            static_gt = batch["params_decode"]["end_effect_stationary"]
            static_pred = output["params_decode"]["end_effect_stationary"]
            static_acc = (static_pred == static_gt).float().mean(dim=-1)
            static_acc = (static_acc * length_mask).sum() / length_mask_item
            metric_dict["endeffect_static_acc"] = static_acc

        with torch.no_grad():
            # 计算rotation error
            rot6d_gt = batch["params_decode"]["rot6d"]
            rotations_gt = rot6d_to_rotation_matrix(rot6d_gt)
            rot6d_pred = output["params_decode"]["rot6d"]
            rotations_pred = rot6d_to_rotation_matrix(rot6d_pred)
            rel_rotation = torch.matmul(rotations_gt, rotations_pred.transpose(-1, -2))
            angle = rotation_to_angle(rel_rotation)

            metric_dict["rotation_error"] = (angle.mean(dim=-1) * length_mask).sum() / length_mask_item
            metric_dict["rotation_error_root"] = (angle[:, :, 0] * length_mask).sum() / length_mask_item

            k3d_gt = batch["params_decode"]["keypoints3d"]
            transl_gt = batch["params_decode"]["trans"]
            k3d_pred = output["params_decode"]["keypoints3d"]
            transl_pred = output["params_decode"]["trans"]
            metric_dict.update(compute_metrics(k3d_gt, transl_gt, k3d_pred, transl_pred))

            latent_max = output["q_z"].mean.max()
            latent_min = output["q_z"].mean.min()
            metric_dict["latent_mean_max"] = latent_max
            metric_dict["latent_mean_min"] = latent_min
            latent_std = output["q_z"].std.max()
            metric_dict["latent_std_max"] = latent_std
            latent_std_mean = output["q_z"].std.mean()
            metric_dict["latent_std_mean"] = latent_std_mean

        for key, val in metric_dict.items():
            metric_dict[key] = val.item()

        return loss_dict, metric_dict, per_sample_loss

    def encode(self, batch):
        assert "length" in batch, "length is required for encode"
        # forward函数仅用来进行encode
        length = batch["length"]
        batch_for_network = self.encode_motion(batch, length)
        output = self.network.encode(batch_for_network, length)
        return output

    def decode(self, z, forward_body=False, forward_vertices=False):
        ret = self.network.decode(z)
        ret = self.decode_motion(ret)
        if forward_body:
            decode = self.forward_body(ret, forward_vertices=forward_vertices)
            ret["params_decode"] = decode["params_decode"]
        return ret

    def forward_in_training(self, batch):
        self.global_iteration += 1
        length = batch["length"]
        batch_for_network = self.encode_motion(batch)
        batch_for_network_decode = self.decode_motion(batch_for_network)
        batch_for_network["output_decode"] = batch_for_network_decode
        batch_decode = self.forward_body(batch_for_network_decode)
        batch_for_network["params_decode"] = batch_decode["params_decode"]

        # forward network
        output = self.network(batch_for_network, length)
        output["output_decode"] = self.decode_motion(output["recon"])
        output_decode = self.forward_body(output["output_decode"])
        output["params_decode"] = output_decode["params_decode"]

        loss_dict, metric_dict, per_sample_loss = self.compute_loss(output, batch_for_network, length)
        loss_weight = self.loss_weight.copy()
        if self.vertices_loss_start_iteration > 0 and self.global_iteration >= self.vertices_loss_start_iteration:
            # [0, vertices_loss_start_iteration] => 0
            # [vertices_loss_start_iteration, 2*vertices_loss_start_iteration] => 1
            # [2*vertices_loss_start_iteration, inf] => 1
            current_factor = (
                self.global_iteration - self.vertices_loss_start_iteration
            ) / self.vertices_loss_start_iteration
            current_factor = max(min(current_factor, 1.0), 0.0)
            for key in ["loss_vertices", "loss_trans"]:
                if key in loss_weight:
                    loss_weight[key] = loss_weight[key] * current_factor

        per_sample_loss_ = sum(per_sample_loss[k] * loss_weight[k] for k in per_sample_loss.keys() if k in loss_weight)

        loss = sum([loss_dict[k] * loss_weight[k] for k in loss_dict.keys() if k in loss_weight])

        return {
            "loss": loss,
            "loss_dict": loss_dict,
            "tensor_results": {
                "per_sample_loss": per_sample_loss_,
                "index": batch.get("index", None),
            },
        }

    @torch.no_grad()
    def validate(self, batch):
        length = batch["length"]
        # prepare gt
        batch_for_network = self.encode_motion(batch)
        batch_for_network_decode = self.decode_motion(batch_for_network)
        batch_for_network["output_decode"] = batch_for_network_decode
        batch_decode = self.forward_body(batch_for_network_decode)
        batch_for_network["params_decode"] = batch_decode["params_decode"]

        # forward network
        output = self.network(batch_for_network, length)
        output["output_decode"] = self.decode_motion(output["recon"])
        output_decode = self.forward_body(output["output_decode"])
        output["params_decode"] = output_decode["params_decode"]

        # compute loss
        loss_dict, metric_dict, per_sample_loss = self.compute_loss(output, batch_for_network, length)
        output.update(output_decode)

        return {"output": output, "batch": batch_for_network, "metrics": metric_dict}

    def set_epoch(self, epoch):
        self.epoch = epoch


class MotionVAEPipeline(nn.Module):
    def __init__(
        self,
        network_module,
        network_module_args,
        loss_weight={},
        loss_vertices_type="smooth_l1",
        fps=30,
        use_transl=True,
        use_root_rot=True,
        clip_max_length=False,
    ):
        super().__init__()
        self.loss_weight = loss_weight
        self.fps = fps
        self.loss_vertices_type = loss_vertices_type
        self.use_transl = use_transl
        self.use_root_rot = use_root_rot
        self.clip_max_length = clip_max_length
        self.network = load_object(network_module, network_module_args)
        self.body_model = SMPLSkeleton()
        self.mesh_model = SMPLMesh()
        joints_weight = torch.FloatTensor(JOINTS_WEIGHTS_SMPLH_JOINTS).reshape(1, 1, -1, 1)
        self.register_buffer("joints_weight", joints_weight)
        # 增加一个计算均值方差的；传入到网络的时候进行归一化操作
        self.epoch = 0

    def encode_motion(self, batch):
        length_max = batch["length"].max().item()
        if self.clip_max_length:
            rot6d = batch["rot6d"][:, :length_max]
            trans = batch["trans"][:, :length_max]
        else:
            rot6d = batch["rot6d"]
            trans = batch["trans"]
        shapes = batch["shapes"]
        # forward
        ret = {
            "rot6d": [],
            "rot6d_hand": [],
            "trans_vel_local": [],
            "joints": [],
            "stationary": [],
            "shapes": [],
        }
        for bs in range(rot6d.shape[0]):
            keypoints3d = self.body_model({"rot6d": rot6d[bs], "trans": trans[bs], "shapes": shapes[bs][None]})[
                "keypoints3d"
            ]
            # (T, 22, 6) => (T, 22*6)
            ret["rot6d"].append(rot6d[bs, :, :22].reshape(rot6d.shape[1], -1))
            # (T, 30, 6) => (T, 30*6)
            ret["rot6d_hand"].append(rot6d[bs, :, 22:].reshape(rot6d.shape[1], -1))
            # trans_vel_local
            root_rotations = rot6d_to_rotation_matrix(rot6d[bs, :, 0])
            transl_vel = get_local_transl_vel(trans[bs], root_rotations, self.fps)
            ret["trans_vel_local"].append(transl_vel)
            # 计算stationary
            joints_vel = torch.norm(keypoints3d[1:, :22] - keypoints3d[:-1, :22], dim=-1)
            joints_vel = torch.cat([joints_vel, joints_vel[-1:]], dim=0)
            # 5mm
            stationary = joints_vel < 5e-3
            ret["stationary"].append(stationary)
            # joints normalize
            joints = keypoints3d[:, :22, :]
            joints = joints - joints[:, :1]
            ret["joints"].append(joints.reshape(joints.shape[0], -1))
            ret["shapes"].append(shapes[bs][None].repeat(keypoints3d.shape[0], 1))
        for key in ret.keys():
            ret[key] = torch.stack(ret[key], dim=0)  # type: ignore
        return ret

    def decode_motion(self, latent):
        rot6d = latent["rot6d"]
        rot6d_hand = latent["rot6d_hand"]
        trans_vel_local = latent["trans_vel_local"]
        # dict_keys(['rot6d', 'rot6d_hand', 'trans_vel_local', 'stationary', 'shapes'])
        shapes = latent["shapes"]

        ret = {
            "rot6d": rot6d,
            "rot6d_hand": rot6d_hand,
            "transl_vel": trans_vel_local,
            "shapes": shapes,
            "stationary": latent["stationary"],
        }
        return ret

    def compute_loss(self, output, batch, length):
        loss_dict = {}
        metric_dict = {}
        mean = output["mean"]
        p_z = torch.distributions.normal.Normal(loc=torch.zeros_like(mean), scale=torch.ones_like(mean))

        length_mask = length_to_mask(length, batch["rot6d"].shape[1])
        length_mask_item = max(length_mask.sum(), 1)

        # FIXME: 这里需要增加mask
        loss_kl = torch.distributions.kl.kl_divergence(output["q_z"], p_z).mean(dim=-1)
        # loss_kl: (B, T)
        loss_kl = (loss_kl * length_mask).sum() / length_mask_item
        loss_dict["loss_kl"] = loss_kl

        if self.loss_weight["loss_rot6d"] > 0:
            loss_rot6d = torch.nn.functional.mse_loss(output["rot6d"], batch["rot6d"], reduction="none")
            loss_rot6d = loss_rot6d.mean(dim=-1).mean(dim=-1)
            loss_dict["loss_rot6d"] = (loss_rot6d * length_mask).sum() / length_mask_item

        if self.loss_weight["loss_shape"] > 0:
            loss_shape = torch.nn.functional.mse_loss(output["shapes"][:, 0], batch["shapes"][:, 0], reduction="mean")
            loss_dict["loss_shape"] = loss_shape

        if self.loss_weight["loss_transl_vel"] > 0:
            loss_transl_vel = torch.nn.functional.mse_loss(
                output["trans_vel_local"], batch["trans_vel_local"], reduction="none"
            )
            loss_transl_vel = loss_transl_vel.mean(dim=-1)
            loss_dict["loss_transl_vel"] = (loss_transl_vel * length_mask).sum() / length_mask_item

        if "vertices" in batch and self.loss_weight["loss_vertices"] > 0:
            vertices_gt_wotrans = batch["vertices_wotrans"]
            vertices_pred_wotrans = output["vertices_wotrans"]
            if self.loss_vertices_type == "mse":
                loss_vertices_local = (
                    torch.nn.functional.mse_loss(vertices_pred_wotrans, vertices_gt_wotrans, reduction="none")
                    .mean(dim=-1)
                    .mean(dim=-1)
                )
            elif self.loss_vertices_type == "smooth_l1":
                loss_vertices_local = (
                    torch.nn.functional.smooth_l1_loss(
                        vertices_pred_wotrans, vertices_gt_wotrans, reduction="none", beta=0.2
                    )
                    .mean(dim=-1)
                    .mean(dim=-1)
                )
            else:
                raise ValueError(f"Invalid loss_vertices_type: {self.loss_vertices_type}")
            # 这个把vertices detach掉；只考虑对translation的梯度
            loss_vertices_local = (loss_vertices_local * length_mask).sum() / length_mask_item
            if self.use_transl:
                vertices_gt = vertices_gt_wotrans.detach() + batch["params_decode"]["trans"][:, :, None, :]
                vertices_pred_hat = vertices_gt_wotrans.detach() + output["params_decode"]["trans"][:, :, None, :]
                vertices_pred = vertices_pred_wotrans.detach() + output["params_decode"]["trans"][:, :, None, :]
                loss_vertices_transl = torch.nn.functional.mse_loss(vertices_pred_hat, vertices_gt, reduction="none")
                loss_vertices_transl = loss_vertices_transl.mean(dim=-1).mean(dim=-1)
                loss_vertices_transl = (loss_vertices_transl * length_mask).sum() / length_mask_item
                loss_dict["loss_vertices"] = (loss_vertices_local + loss_vertices_transl) / 2
            else:
                loss_dict["loss_vertices"] = loss_vertices_local

            with torch.no_grad():
                metric_dict["mpvpe_aligned"] = (
                    (torch.norm(vertices_gt_wotrans - vertices_pred_wotrans, dim=-1).mean(dim=-1) * length_mask).sum()
                    / length_mask_item
                    * 1000
                )  # 定义为mm
                metric_dict["mpvpe"] = (
                    (torch.norm(vertices_gt - vertices_pred, dim=-1).mean(dim=-1) * length_mask).sum()
                    / length_mask_item
                    * 1000
                )  # 定义为mm

        if "stationary" in batch:
            # batch['stationary'] is (B, L, 22)
            stationary_flag = batch["stationary"].float()
            # logits
            pred_stationary = output["stationary"]

            loss_stationary = torch.nn.functional.binary_cross_entropy_with_logits(
                pred_stationary, stationary_flag, reduction="none"
            )
            loss_stationary = loss_stationary.mean(dim=-1)
            loss_dict["loss_stationary"] = (loss_stationary * length_mask).sum() / length_mask_item

        for key, val in loss_dict.items():
            if torch.isnan(val) or torch.isinf(val):
                if torch.isnan(val):
                    print(f"{key} is nan")
                    print(val)
                if torch.isinf(val):
                    print(f"{key} is inf")
                    print(val)
                # breakpoint()
                raise ValueError(f"{key} is nan or inf")

        return loss_dict, metric_dict

    def forward(self, batch):
        # forward函数仅用来进行encode
        length = batch["length"]
        batch_for_network = self.encode_motion(batch)
        output = self.network(batch_for_network, length)
        return output

    def decode(self, z):
        ret = self.network.decode(z)
        ret = self.forward_body(ret)
        return ret

    def forward_in_training(self, batch):
        length = batch["length"]
        batch_for_network = self.encode_motion(batch)
        output = self.network(batch_for_network, length)

        # forward body model in training
        batch_for_network.update(self.forward_body(batch_for_network))
        output.update(self.forward_body(output["recon"]))

        loss_dict, metric_dict = self.compute_loss(output, batch_for_network, length)
        loss = sum([loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys()])
        return {
            "loss": loss,
            "loss_dict": loss_dict,
        }

    @staticmethod
    def rollout_local_transl_vel(local_transl_vel, global_orient, transl_0=None):
        """
        trans velocity is in local coordinate (or, SMPL-coord)
        Args:
            local_transl_vel: (*, L, 3)
            global_orient: (*, L, 3, 3)
            transl_0: (*, 1, 3), if not provided, the start point is 0
        Returns:
            trans: (*, L, 3)
        """
        transl_vel = torch.einsum("...lij,...lj->...li", global_orient, local_transl_vel)

        # set start point
        if transl_0 is None:
            transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
        transl_ = torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2)

        # rollout from start point
        trans = torch.cumsum(transl_, dim=-2)
        return trans

    def forward_body(self, latent):
        rot6d = latent["rot6d"]
        rot6d_hand = latent["rot6d_hand"]
        rot6d = rot6d.reshape(rot6d.shape[0], rot6d.shape[1], -1, 6)
        rot6d_hand = rot6d_hand.reshape(rot6d_hand.shape[0], rot6d_hand.shape[1], -1, 6)

        trans_vel_local = latent["trans_vel_local"]
        shapes = latent["shapes"]
        stationary = latent["stationary"]

        rotations = rot6d_to_rotation_matrix(torch.cat([rot6d, rot6d_hand], dim=-2))

        rotations_root = rotations[:, :, 0]
        transl = self.rollout_local_transl_vel(trans_vel_local / self.fps, rotations_root)

        if shapes.shape[1] == 1:
            shapes = shapes.repeat(1, rot6d.shape[1], 1)
        rot6d_flat = rot6d.reshape(rot6d.shape[0] * rot6d.shape[1], -1, 6)
        rot6d_hand_flat = rot6d_hand.reshape(rot6d_hand.shape[0] * rot6d_hand.shape[1], -1, 6)
        rot6d_compose_flat = torch.cat([rot6d_flat, rot6d_hand_flat], dim=-2)
        transl_flat = transl.reshape(transl.shape[0] * transl.shape[1], 3)
        shapes_flat = shapes.reshape(shapes.shape[0] * shapes.shape[1], -1)
        params = {
            "rot6d": rot6d_compose_flat,
            "trans": transl_flat,
            "shapes": shapes_flat,
        }

        out = self.mesh_model(params)
        ret = {
            "vertices_wotrans": out["vertices_wotrans"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3),
            "vertices": out["vertices"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3),
        }
        out_keypoints = self.body_model(params)
        ret["keypoints3d"] = out_keypoints["keypoints3d"].reshape(rot6d.shape[0], rot6d.shape[1], -1, 3)

        ret["rot6d"] = torch.cat([rot6d, rot6d_hand], dim=-2)
        ret["shapes"] = shapes
        ret["trans_vel_local"] = trans_vel_local
        ret["stationary"] = stationary

        ret["params_decode"] = {
            "rotations": rotations,
            "trans": transl,
            "shapes": shapes,
            "stationary": stationary,
        }
        return ret

    @torch.no_grad()
    def validate(self, batch):
        length = batch["length"]
        length_max = length.max()
        batch_for_network = self.encode_motion(batch)
        output = self.network(batch_for_network, length)

        batch_for_network.update(self.forward_body(batch_for_network))
        output.update(self.forward_body(output["recon"]))
        if False:
            outname = os.path.join("output", "body_vae", f'vis_{self.epoch}_{batch["index"][0].item()}')
            visualize_skeleton(
                k3d_cat, outname, vis_direction=[0, 2], another_direction=[1, 2], bbox_size=1.7, vis_size=1024, fps=30
            )
        loss_dict, metric_dict = self.compute_loss(output, batch_for_network, length)
        output.update(self.decode_motion(output["recon"]))

        return {"output": output, "batch": batch_for_network, "metrics": metric_dict}

    def set_epoch(self, epoch):
        self.epoch = epoch


class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(CausalConv1d, self).__init__()
        self.pad = (kernel_size - 1) * dilation + (1 - stride)
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation  # no padding here
        )

    def forward(self, x):
        x = nn.functional.pad(x, (self.pad, 0))  # only pad on the left
        return self.conv(x)


class SkeletonVAENetwork(nn.Module):
    def __init__(
        self,
        input_heads={
            "rot6d": 22 * 6,  # 带上
            "rot6d_hand": 2 * 15 * 6,
            "trans_vel_local": 3,
            "joints": 22 * 3,
            "shapes": 16,
        },
        output_heads={
            "rot6d": 22 * 6,  # 带上
            "rot6d_hand": 2 * 15 * 6,
            "trans_vel_local": 3,
            "stationary": 22,  # 判断躯干的点是否静止
            "shapes": 16,
        },
        depth=8,
        d_model=512,
        nhead=16,
        code_dim=256,
        attention_type="self",
        max_len=32,
        downsample=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.depth = depth

        self.input_heads = nn.ModuleDict(
            {key: Mlp(input_heads[key], hidden_features=512, out_features=d_model) for key in input_heads.keys()}
        )

        self.attention_type = attention_type
        self.max_len = max_len
        self.downsample = downsample

        if downsample:
            self.max_len = max_len // 2
            self.downsample_conv = CausalConv1d(d_model, d_model, 3, stride=2)

        if attention_type == "self":
            self.encoder_blocks = nn.ModuleList(
                [
                    torch.nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        batch_first=True,
                        dim_feedforward=d_model * 4,
                        norm_first=True,
                    )
                    for _ in range(depth)
                ]
            )
        elif attention_type == "rope":
            from ..network.layers.rope_attention_block import EncoderRoPEBlock

            self.encoder_blocks = nn.ModuleList([EncoderRoPEBlock(d_model, nhead) for _ in range(depth)])
        self.encoder_head = nn.Linear(d_model, code_dim * 2)

        # build decoder
        self.proj_code = nn.Linear(code_dim, d_model)
        if attention_type == "self":
            self.decoder_blocks = nn.ModuleList(
                [
                    torch.nn.TransformerEncoderLayer(
                        d_model=d_model,
                        nhead=nhead,
                        batch_first=True,
                        dim_feedforward=d_model * 4,
                        norm_first=True,
                    )
                    for _ in range(depth)
                ]
            )
        elif attention_type == "rope":
            from ..network.layers.rope_attention_block import EncoderRoPEBlock

            self.decoder_blocks = nn.ModuleList([EncoderRoPEBlock(d_model, nhead) for _ in range(depth)])
        if downsample:
            self.upsample_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest-exact"), CausalConv1d(d_model, d_model, 3)
            )
        self.output_heads = nn.ModuleDict({key: nn.Linear(d_model, output_heads[key]) for key in output_heads.keys()})

    def create_attnmask(self, x):
        B, L, D = x.shape
        attnmask_causal = ~torch.tril(torch.ones((L, L), device=x.device, dtype=torch.bool), diagonal=0)

        if L > self.max_len:
            attnmask = torch.ones((L, L), device=x.device, dtype=torch.bool)
            for i in range(L):
                min_ind = max(0, i - self.max_len // 2)
                max_ind = min(L, i + self.max_len // 2)
                max_ind = max(self.max_len, max_ind)
                min_ind = min(L - self.max_len, min_ind)
                attnmask[i, min_ind:max_ind] = False
            attnmask = attnmask | attnmask_causal
        else:
            attnmask = attnmask_causal
        return attnmask

    def encode(self, x):
        # x: (B, T, D)
        feat = 0
        for key in self.input_heads.keys():
            feat += self.input_heads[key](x[key])
        x = feat
        if self.downsample:
            x_bdt = x.permute(0, 2, 1)
            x_bdt = self.downsample_conv(x_bdt)
            x = x_bdt.permute(0, 2, 1)

        batch_size, seq_len, _ = x.shape
        if self.attention_type == "rope":
            attnmask = self.create_attnmask(x)
        else:
            attnmask = None

        for block in self.encoder_blocks:
            # skeleton attention
            if self.attention_type == "self":
                x = x.reshape(batch_size * seq_len, 1, x.shape[-1])
                x = block(x)
                x = x.reshape(batch_size, seq_len, x.shape[-1])
            elif self.attention_type == "rope":
                x = block(x, attn_mask=attnmask)
        latent = self.encoder_head(x)
        mu, std = latent.chunk(2, dim=-1)
        std_softplus = torch.nn.functional.softplus(std)
        print(
            f"[{self.__class__.__name__}] {mu.device} mu: {mu.mean()}, std {std.min():.6f}, {std.max():.6f}: {std_softplus.mean():.5f}, min: {std_softplus.min():.5f}, max: {std_softplus.max():.5f}"
        )
        return torch.distributions.normal.Normal(mu, torch.nn.functional.softplus(std))

    def decode(self, z):
        z = self.proj_code(z)
        batch_size, seq_len, _ = z.shape
        if self.attention_type == "rope":
            attnmask = self.create_attnmask(z)
        else:
            attnmask = None
        for block in self.decoder_blocks:
            if self.attention_type == "self":
                z = z.reshape(batch_size * seq_len, 1, z.shape[-1])
                z = block(z)
                z = z.reshape(batch_size, seq_len, z.shape[-1])
            elif self.attention_type == "rope":
                z = block(z, attn_mask=attnmask)
        if self.downsample:
            z_bdt = z.permute(0, 2, 1)
            z_bdt = self.upsample_conv(z_bdt)
            z = z_bdt.permute(0, 2, 1)
        ret = {}
        for key in self.output_heads.keys():
            ret[key] = self.output_heads[key](z)
        return ret

    def forward(self, x, lengths=None):
        q_z = self.encode(x)
        if self.training:
            q_z_sample = q_z.rsample()
        else:
            q_z_sample = q_z.mode
        x_recon = self.decode(q_z_sample)
        return {"recon": x_recon, "q_z": q_z, "mean": q_z.mean, "std": q_z.scale}


def report_mean_std(values, name):
    mean = values.mean(dim=0, keepdim=True)
    std = values.std(dim=0, keepdim=True)
    value_norm = (values - mean) / (std + 1e-8)
    value_norm_max = value_norm.abs().max(dim=0).values
    format_string = "["
    for i in range(value_norm_max.shape[0]):
        format_string += f"{value_norm_max[i]:.2f}, "
    format_string += "]"
    print(f"{name:20s}: {format_string}")


if __name__ == "__main__":
    # python -m hymotion.pipeline.motion_vae
    cmu_set = sorted(
        os.listdir(
            "/apdcephfs_cq10/share_1467498/datasets/motion_data/HunyuanMotion/Academic/20250916/motions_o6dp_v0930/HumanML3D-CMU"
        )
    )
    cmu_set = [f'HumanML3D-CMU/{s.replace(".npy", "")}' for s in cmu_set]
    motion_set = cmu_set
    data_cfg = {
        "roots": [
            {
                "root": "/apdcephfs_cq10/share_1467498/datasets/motion_data/HunyuanMotion/Academic/20250916",
                "checklist": {"file_list": motion_set},
            },
        ],
        "max_len": 360,
        "motion_dir": "motions",
        "n_fold": -1,
        "append_mirror": False,
    }
    dataset = ExampleDataset(**data_cfg)
    print(f"len(dataset): {len(dataset)}")
    data_all = dataset.load_all_data()
    trans_all = []
    trans_vel_all = []
    rot6d_all = []
    rot6d_vel_all = []
    for data in data_all:
        trans_all.append(data["trans"])
        trans_vel = data["trans"][1:] - data["trans"][:-1]
        trans_vel_all.append(trans_vel)
        rot6d = data["rot6d"]
        rot6d_all.append(rot6d)
        rot6d_vel = rot6d[1:] - rot6d[:-1]
        rot6d_vel_all.append(rot6d_vel)
    trans_all = torch.cat(trans_all, dim=0)
    report_mean_std(trans_all, "trans")
    trans_vel_all = torch.cat(trans_vel_all, dim=0)
    report_mean_std(trans_vel_all, "trans_vel")

    rot6d_all = torch.cat(rot6d_all, dim=0)
    rot6d_vel_all = torch.cat(rot6d_vel_all, dim=0)
    for j in range(22):
        rot6d_j = rot6d_all[:, j]
        report_mean_std(rot6d_j, f"joint {j} rot6d")
        rot6d_vel_j = rot6d_vel_all[:, j]
        report_mean_std(rot6d_vel_j, f"joint {j} rot6d_vel")

    breakpoint()
    dataset = ExampleDataset(
        roots=[{"root": "/apdcephfs_cq10/share_1467498/datasets/motion_data/HunyuanMotion/Taobao/20250901"}],
        split="train",
        max_len=320,
    )

    data = dataset[0]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False, num_workers=16)

    pipeline = MotionVAEPipeline(
        "hymotion.pipeline.motion_vae.SkeletonVAENetwork",
        {},
        {"loss_rot6d": 1.0, "loss_vertices": 1.0},
        loss_vertices_type="smooth_l1",
        fps=30,
        use_transl=True,
        use_root_rot=True,
    )

    for batch in dataloader:
        print(batch["rot6d"].shape)
        print(batch["trans"].shape)
        print(batch["shapes"].shape)
        print(batch["length"], batch["length"].max())
        output = pipeline.validate(batch)
        breakpoint()
        loss = pipeline.forward_in_training(batch)
        print(output["recon"]["rot6d"].shape)
        print(output["recon"]["trans"].shape)
        print(output["recon"]["shapes"].shape)
        print(output["recon"]["stationary"].shape)
        print(output["recon"]["rot6d_hand"].shape)
        print(output["recon"]["trans_vel_local"].shape)
        breakpoint()
    breakpoint()
