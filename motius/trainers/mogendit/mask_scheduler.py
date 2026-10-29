import torch
import numpy as np
import random
import time
from .utils import random_index


class MotionMaskScheduler:
    def __init__(self, motion_rep=None):
        self.temporal_mask_modes = [
            "random_frame",
            "random_phrase",
            "random_start_end",
            "fix_start_end",
            "uncond",
        ]
        self.spatial_mask_modes = ["block_pose", "block_trans", "joint_only"]
        self.motion_rep = motion_rep

    def get_temporal_mask(
        self, motion, length, mode="random_frame", loop_k=None, dtype="float"
    ):
        assert dtype in ["float", "bool"]
        assert mode in self.temporal_mask_modes
        batch_size, n_frames, d_motion = motion.shape
        temporal_mask = torch.zeros(
            (batch_size, n_frames, d_motion), dtype=torch.float32, device=motion.device
        )
        lengths = length.reshape(-1)
        if mode == "uncond":
            pass
        elif mode == "random_frame":
            # Observe frames every trans_length frames
            # used for inference
            obs_rate = random.uniform(0.05, 0.1)  # 观察帧比例
            for i, length in enumerate(lengths.cpu().numpy()):
                length = int(length)
                obs_indices = random_index(data_len=length, sampling_rate=obs_rate)
                # 时间维度现在是第二个维度，调整索引位置
                temporal_mask[i, obs_indices] = 1  # set keyframes

        elif mode == "random_phrase":
            # Observe frames in random phrases
            # used for inference
            for i, length in enumerate(lengths.cpu().numpy()):
                length = int(length)
                length_phrase = min(random.randint(2, 20), length)
                n_phrase = length // length_phrase
                phrase_idx_dict = {}
                for j in range(n_phrase):
                    phrase_idx_dict.update(
                        {
                            j: list(
                                range(
                                    j * length_phrase,
                                    min((j + 1) * length_phrase, length),
                                )
                            )
                        }
                    )
                # 注意：原代码中random_index未定义，这里保持原样
                ramdom_phrase_idx = random_index(
                    data_len=n_phrase,
                    sampling_rate=random.uniform(min(0.05, 1 / n_phrase), 0.1),
                )
                for j in ramdom_phrase_idx:
                    # 时间维度现在是第二个维度，调整索引位置
                    temporal_mask[i, phrase_idx_dict[j]] = 1  # set keyframes

        elif mode == "random_start_end":
            # 保障起始1到n帧和末尾部分mask=1，中间随机连续50%-90% mask=0
            for i, length in enumerate(lengths.cpu().numpy()):
                length = int(length)

                # 确定起始部分的长度n（1到总长度的1/10之间）
                max_start_length = max(1, length // 5)  # 起始部分最长不超过总长度的1/5
                start_length = random.randint(1, max_start_length)

                remaining_length = length - start_length
                # 计算中间mask=0区域的长度（剩余长度的50%-90%）
                min_mask0_length = int(remaining_length * 0.5)
                max_mask0_length = int(remaining_length * 0.9)
                mask0_length = random.randint(min_mask0_length, max_mask0_length)

                # 计算末尾部分的长度
                end_length = length - start_length - mask0_length
                end_length = max(end_length, 1)  # 确保末尾部分至少有1帧

                # 设置起始部分mask=1，调整索引位置
                temporal_mask[i, :start_length] = 1

                # 设置末尾部分mask=1，调整索引位置
                temporal_mask[i, length-end_length:length] = 1
        elif mode == "fix_start_end":
            assert loop_k is not None
            # 设置起始部分mask=1，调整索引位置
            temporal_mask[:, :loop_k] = 1
            # 设置末尾部分mask=1，调整索引位置
            for i, valid_length in enumerate(lengths.tolist()):
                temporal_mask[i, max(0, int(valid_length)-loop_k):int(valid_length)] = 1
        temporal_mask *= self.get_length_mask_bool(motion, lengths)
        if dtype == "bool":
            temporal_mask = temporal_mask >= 1
        return temporal_mask

    def get_spatial_mask(self, motion, length, mode="uncond", dtype="float"):
        assert dtype in ["float", "bool"]
        assert mode in self.spatial_mask_modes
        assert self.motion_rep is not None  # 需要提供motion_rep
        batch_size, n_frames, d_motion = motion.shape
        spatial_mask = torch.zeros(
            (batch_size, n_frames, d_motion), dtype=torch.float32, device=motion.device
        )
        lengths = length.reshape(-1)
        if mode == "block_pose":
            spatial_mask = torch.ones_like(spatial_mask)
            spatial_mask[..., self.motion_rep.pose_mask.to(motion.device)] *= 0

        elif mode == "block_trans":
            spatial_mask = torch.ones_like(spatial_mask)
            spatial_mask[..., self.motion_rep.trans_mask.to(motion.device)] *= 0
        elif mode == "joint_only":
            spatial_mask[..., self.motion_rep.joint_mask.to(motion.device)] += 1
        else:
            raise NotImplementedError
        for i, length in enumerate(lengths.cpu().numpy()):
            length = int(length)
            spatial_mask[i, length:] *= 0
        if dtype == "bool":
            spatial_mask = spatial_mask >= 1
        return spatial_mask

    def get_length_mask_bool(self, motion, length):
        batch_size, n_frames, d_motion = motion.shape
        length_mask = torch.zeros(
            (batch_size, n_frames, d_motion), dtype=torch.bool, device=motion.device
        )
        lengths = length.reshape(-1).cpu().numpy()
        for i, length in enumerate(lengths):
            length = int(length)
            length_mask[i, :length] = True  # set keyframes
        return length_mask

    def get_formulated_mask(
        self, motion, length, mode_formula: dict = {"random_frame": 1.0}
    ):
        assert len(motion.shape) == 3  # batch, seq_len, dim
        mask = []
        modes = mode_formula.keys()
        n_modes = len(modes)
        processed_samples = 0
        for i, _mode in enumerate(modes):
            if _mode in self.temporal_mask_modes:
                mask_func = self.get_temporal_mask
            elif _mode in self.spatial_mask_modes:
                mask_func = self.get_spatial_mask
            else:
                raise NotImplementedError(f"Unknown mask mode: {_mode}")
            if i == n_modes - 1:
                mask.append(
                    mask_func(
                        motion=motion[processed_samples:],
                        length=length[processed_samples:],
                        mode=_mode,
                    )
                )
                continue
            else:
                n_sample = int(motion.shape[0] * mode_formula[_mode])
                # import pdb; pdb.set_trace()
                mask.append(
                    mask_func(
                        motion=motion[processed_samples : processed_samples + n_sample],
                        length=length[processed_samples : processed_samples + n_sample],
                        mode=_mode,
                    )
                )
                processed_samples += n_sample
        mask = torch.cat(mask, dim=0)
        return mask.to(motion.device)
