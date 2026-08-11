from typing import Dict, Tuple, Union, List

from mmcv import BaseTransform
import numpy as np
import torch

from motius.models.vermo.task_utils.task_lib.completion_tasks.motion_inbetween import (
    MotionInbetween,
)
from motius.models.vermo.task_utils.task_lib.completion_tasks.motion_prediction import (
    MotionPrediction,
)
from motius.registry import TRANSFORMS


# TODO: merge the logic of MotionLLaMA and VersatileMotion
@TRANSFORMS.register_module()
class SplitPrediction(BaseTransform):
    """
    将 results[self.key] 的 motion 分割为 past / future，用于 motion prediction 训练。
    在原功能基础上，加入“以一定随机概率仅使用单帧作为 past 条件”的选项。

    兼容输入形状：
      - [T, C]
      - [P, T, C]  （多人或多实例堆叠）
    统一在时间维（倒数第二维）上切分。

    参数
    ----
    key : str
        motion 在 results 中的键名（默认 "motion"）
    past_ratio : float
        非随机比例时，past 的占比（与原行为一致）
    random_ratio : bool
        若为 True，则从 [0.1, 0.5] 均匀采样 past 比例（与原行为一致）
    single_frame_prob : float
        以该概率启用“单帧 past”模式（past 仅含 1 帧）
    min_future_frames : int
        切分后 future 至少包含的最小帧数（默认 4，与原逻辑一致）
    """

    def __init__(
        self,
        key: str = "motion",
        past_ratio: float = 0.4,
        random_ratio: bool = False,
        single_frame_prob: float = 0.25,
        min_future_frames: int = 17,
    ):
        self.key = key
        self.random_ratio = bool(random_ratio)
        if not self.random_ratio:
            self.past_ratio = float(past_ratio)

        self.single_frame_prob = float(single_frame_prob)
        self.min_future_frames = int(min_future_frames)
        assert (
            0.0 <= self.single_frame_prob <= 1.0
        ), "single_frame_prob must be in [0,1]"
        assert self.min_future_frames >= 1, "min_future_frames must be >= 1"

    def _choose_past_frames(self, num_frames: int) -> int:
        """
        决定 past 的帧数：
          - 以 single_frame_prob 的概率：past_frames = 1（若可行）
          - 否则沿用原策略（固定比例或随机比例），并下限为 4（与原版一致）
        然后统一裁剪到 [1, num_frames - min_future_frames] 区间，保证 future ≥ min_future_frames。
        当样本过短（num_frames <= min_future_frames），退化为 past=1, future=num_frames-1。
        """
        # 确保可切分
        if num_frames <= 1:
            # 无法切分，回退为 past=1, future=0（后续会因断言失败抛错）
            return 1

        # 若样本极短，尽量保留至少 1 帧作为 future
        if num_frames <= self.min_future_frames:
            return max(1, num_frames - 1)

        # 单帧 past 分支
        if self.single_frame_prob > 0.0 and np.random.rand() < self.single_frame_prob:
            past_frames = 1
        else:
            # 原策略
            if self.random_ratio:
                past_ratio = np.random.uniform(0.1, 0.5)
                past_frames = max(4, int(num_frames * past_ratio))
            else:
                past_frames = int(num_frames * self.past_ratio)

        # 统一裁剪，保证 future 至少 min_future_frames 帧，且 past 至少 1 帧
        past_frames = int(np.clip(past_frames, 1, num_frames - self.min_future_frames))
        return past_frames

    def split_past_future(self, motion: torch.Tensor):
        """
        按时间维（-2）将 motion 切分为 (past, future)，并返回实际 past_ratio。
        motion 形状可以是 [T, C] 或 [P, T, C]。
        """
        assert isinstance(
            motion, torch.Tensor
        ), f"Expect torch.Tensor, got {type(motion)}"
        num_frames = motion.shape[-2]  # ... T ...
        past_frames = self._choose_past_frames(num_frames)

        # 切分
        past = motion[..., :past_frames, :]
        future = motion[..., past_frames:, :]

        # 计算实际 past_ratio
        past_ratio = float(past_frames / max(1, num_frames))
        return (past, future), past_ratio

    def transform(self, results: Dict) -> Dict:
        # 仅在指定任务下生效；保持与原版一致的任务过滤
        if results.get("task") not in [MotionPrediction]:
            return results

        motion = results[self.key]
        (results[f"past_{self.key}"], results[f"future_{self.key}"]), past_ratio = (
            self.split_past_future(motion)
        )

        # 断言切分有效
        def _numel(x: torch.Tensor) -> int:
            return (
                int(x.numel())
                if isinstance(x, torch.Tensor)
                else int(np.asarray(x).size)
            )

        assert (
            _numel(results[f"past_{self.key}"]) > 0
            and _numel(results[f"future_{self.key}"]) > 0
        ), f"Invalid split: past/future empty. motion shape={tuple(motion.shape)}"

        # 帧数与时长（fps 需为标量；若你的管线里 fps 可能为列表，请在上游先规整）
        fps = results.get("fps", None)
        past_T = results[f"past_{self.key}"].shape[-2]
        future_T = results[f"future_{self.key}"].shape[-2]

        results[f"past_num_frames"] = int(past_T)
        results[f"future_num_frames"] = int(future_T)
        results["past_ratio"] = float(past_ratio)

        # Propagate per-person frame counts to past/future sub-motions.
        if "per_person_num_frames" in results:
            results["past_per_person_num_frames"] = np.array([
                min(int(t), past_T) for t in results["per_person_num_frames"]
            ], dtype=np.int64)
            results["future_per_person_num_frames"] = np.array([
                max(0, int(t) - past_T) for t in results["per_person_num_frames"]
            ], dtype=np.int64)

        if fps is not None and not isinstance(fps, (list, tuple)):
            results[f"past_duration"] = past_T / fps
            results[f"future_duration"] = future_T / fps
        else:
            # 若存在多人且 fps 为列表，保守地不计算 duration，避免歧义
            results[f"past_duration"] = None
            results[f"future_duration"] = None

        return results


@TRANSFORMS.register_module()
class SplitInbetween(BaseTransform):
    """
    将 results[key] 的序列按时间维(-2)切为 past / middle / future，用于 motion in-betweening 训练。
    在原功能的基础上，加入“成对单帧条件”的随机选项：以给定概率同时使用 1 帧的 past 与 1 帧的 future。

    兼容输入形状：
      - [T, C]
      - [P, T, C]  （多人/多实例堆叠）

    参数
    ----
    keys : Union[str, List[str]]
        需要切分的结果键，默认 "motion"；支持多个键同步切分。
    past_ratio : float
        非随机比例时，past 的占比（与原逻辑一致）。
    future_ratio : float
        非随机比例时，future 的占比（与原逻辑一致）。
    random_ratio : bool
        若为 True，则从 [0.1, 0.25] 分别均匀采样 past 与 future 的比例，
        并让三段至少满足最小帧数（见下两个参数）。
    single_frame_pair_prob : float
        以该概率启用“成对单帧条件”模式：past_frames = 1 且 future_frames = 1。
        若序列过短无法满足中段最小帧数，则自动回退到常规切分策略。
    min_edge_frames : int
        常规切分下，past 与 future 的**最小**帧数（默认 4，保持与原注释一致）。
        *注意*：当触发成对单帧模式时，此限制对两端不生效（两端固定为 1）。
    min_middle_frames : int
        中间段（middle）的最小帧数（默认 4；原版未显式校验，此处补齐稳健性）。

    返回到 results 的字段（按每个 key）：
      - past_{key}, middle_{key}, future_{key}  : 张量，形如 [..., T_segment, C]
      - past_{key}_num_frames, middle_{key}_num_frames, future_{key}_num_frames : int
      - 以及全局：results["past_ratio"], results["future_ratio"], results["used_single_frame_pair"]
    """

    def __init__(
        self,
        keys: Union[str, List[str]] = "motion",
        past_ratio: float = 0.2,
        future_ratio: float = 0.2,
        random_ratio: bool = False,
        single_frame_pair_prob: float = 0.25,
        min_edge_frames: int = 4,
        min_middle_frames: int = 4,
    ):
        assert keys, "Keys should not be empty."
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys

        self.random_ratio = bool(random_ratio)
        if not self.random_ratio:
            self.past_ratio = float(past_ratio)
            self.future_ratio = float(future_ratio)

        self.single_frame_pair_prob = float(single_frame_pair_prob)
        self.min_edge_frames = int(min_edge_frames)
        self.min_middle_frames = int(min_middle_frames)

        assert (
            0.0 <= self.single_frame_pair_prob <= 1.0
        ), "single_frame_pair_prob must be in [0, 1]"
        assert self.min_edge_frames >= 1, "min_edge_frames must be >= 1"
        assert self.min_middle_frames >= 1, "min_middle_frames must be >= 1"

    # --------- 内部工具：根据总帧数决定三段长度 ---------
    def _choose_edge_frames(self, num_frames: int) -> Tuple[int, int, bool]:
        """
        决定 past_frames 与 future_frames，并返回是否采用了“成对单帧模式”。

        规则：
          1) 先尝试以 single_frame_pair_prob 的概率启用“成对单帧”：
             若 num_frames >= 1 + min_middle_frames + 1，则返回 (1, 1, True)；
             否则回退到常规策略。
          2) 常规策略：
             - 若 random_ratio=True：past_ratio~U[0.1,0.25]、future_ratio~U[0.1,0.25]，
               然后 past/future 均需 >= min_edge_frames。
             - 若 random_ratio=False：使用给定的 past_ratio / future_ratio，
               同样裁剪到 >= min_edge_frames。
             - 再保证 middle >= min_middle_frames；
               如三段总和超限，按顺序优先收缩 future，再收缩 past，
               直至 middle 满足最小帧，否则最终让 middle=1、两端尽量分配（稳健降级）。
        """
        # --- 成对单帧模式 ---
        if (
            self.single_frame_pair_prob > 0.0
            and np.random.rand() < self.single_frame_pair_prob
        ):
            if num_frames >= (1 + self.min_middle_frames + 1):
                return 1, 1, True
            # 过短无法满足，回退到常规策略

        # --- 常规策略 ---
        if self.random_ratio:
            pr = float(np.random.uniform(0.1, 0.25))
            fr = float(np.random.uniform(0.1, 0.25))
            past_frames = max(self.min_edge_frames, int(num_frames * pr))
            future_frames = max(self.min_edge_frames, int(num_frames * fr))
        else:
            past_frames = max(self.min_edge_frames, int(num_frames * self.past_ratio))
            future_frames = max(
                self.min_edge_frames, int(num_frames * self.future_ratio)
            )

        # 保证 middle >= min_middle_frames：必要时收缩两端
        total_needed = past_frames + self.min_middle_frames + future_frames
        if total_needed > num_frames:
            overflow = total_needed - num_frames
            # 先收缩 future
            reduce_future = min(overflow, max(0, future_frames - self.min_edge_frames))
            future_frames -= reduce_future
            overflow -= reduce_future
            # 再收缩 past
            if overflow > 0:
                reduce_past = min(overflow, max(0, past_frames - self.min_edge_frames))
                past_frames -= reduce_past
                overflow -= reduce_past

        # 如果仍然溢出（极端短序列），做最小稳健降级：middle 至少 1
        if past_frames + future_frames > num_frames - self.min_middle_frames:
            max_edge_sum = max(0, num_frames - self.min_middle_frames)
            # 平分可用的边缘帧数
            share = max_edge_sum // 2
            past_frames = max(1, share)
            future_frames = max(1, max_edge_sum - past_frames)

        # 兜底：保证两端至少 1 帧，且留出 middle
        past_frames = int(np.clip(past_frames, 1, max(1, num_frames - 1)))
        future_frames = int(
            np.clip(future_frames, 1, max(1, num_frames - past_frames - 1))
        )
        return past_frames, future_frames, False

    def split_past_middle_future(self, motion: torch.Tensor):
        """
        对单个张量做切分，返回 (past, middle, future, past_ratio, future_ratio, used_single_pair)。
        motion 可为 [T, C] 或 [P, T, C]；统一在时间维(-2)切割。
        """
        assert isinstance(
            motion, torch.Tensor
        ), f"Expect torch.Tensor, got {type(motion)}"
        num_frames = int(motion.shape[-2])

        # 极短序列的快速兜底
        if num_frames < 3:
            # 无法形成“三段”，直接退化为 1 / 1 / (num_frames-2)；若仍不够将触发后续断言
            past_frames, future_frames, used_pair = 1, 1, True
        else:
            past_frames, future_frames, used_pair = self._choose_edge_frames(num_frames)

        middle_start = past_frames
        middle_end = num_frames - future_frames
        # 保证 middle 至少 1 帧（极端情况兜底）
        if middle_end <= middle_start:
            # 让 future 至少 1，past 至少 1，其余给 middle
            mid_len = max(1, num_frames - 2)
            past_frames = 1
            future_frames = 1
            middle_start = past_frames
            middle_end = past_frames + mid_len
            used_pair = True  # 实际也变成了成对单帧边界

        past = motion[..., :past_frames, :]
        middle = motion[..., middle_start:middle_end, :]
        future = motion[..., -future_frames:, :]

        past_ratio = float(past_frames / max(1, num_frames))
        future_ratio = float(future_frames / max(1, num_frames))
        return (past, middle, future), past_ratio, future_ratio, used_pair

    def transform(self, results: Dict) -> Dict:
        # 仅在指定任务下生效
        if results.get("task") not in [MotionInbetween]:
            return results

        used_single_pair_any = False
        last_past_ratio = None
        last_future_ratio = None

        for key in self.keys:
            if key not in results:
                continue

            motion = results[key]
            (past, middle, future), past_ratio, future_ratio, used_pair = (
                self.split_past_middle_future(motion)
            )

            # 写回
            results[f"past_{key}"] = past
            results[f"middle_{key}"] = middle
            results[f"future_{key}"] = future

            # 断言有效
            def _ok(x: torch.Tensor) -> bool:
                return isinstance(x, torch.Tensor) and x.numel() > 0

            assert (
                _ok(past) and _ok(middle) and _ok(future)
            ), f"Invalid split for key='{key}', motion shape={tuple(motion.shape)}"

            results[f"past_{key}_num_frames"] = int(past.shape[-2])
            results[f"middle_{key}_num_frames"] = int(middle.shape[-2])
            results[f"future_{key}_num_frames"] = int(future.shape[-2])

            # Propagate per-person frame counts to past/middle/future sub-motions.
            if "per_person_num_frames" in results:
                past_T = int(past.shape[-2])
                middle_T = int(middle.shape[-2])
                middle_start_frame = past_T
                middle_end_frame = past_T + middle_T
                results[f"past_per_person_num_frames"] = np.array([
                    min(int(t), past_T) for t in results["per_person_num_frames"]
                ], dtype=np.int64)
                results[f"middle_per_person_num_frames"] = np.array([
                    max(0, min(int(t), middle_end_frame) - middle_start_frame)
                    for t in results["per_person_num_frames"]
                ], dtype=np.int64)
                results[f"future_per_person_num_frames"] = np.array([
                    max(0, int(t) - middle_end_frame)
                    for t in results["per_person_num_frames"]
                ], dtype=np.int64)

            last_past_ratio = past_ratio
            last_future_ratio = future_ratio
            used_single_pair_any = used_single_pair_any or bool(used_pair)

        # 与原实现一致：记录一次（以最后一个 key 的比例为准）
        if last_past_ratio is not None:
            results["past_ratio"] = float(last_past_ratio)
        if last_future_ratio is not None:
            results["future_ratio"] = float(last_future_ratio)

        return results


@TRANSFORMS.register_module(force=True)
class PrepareM2MForVersatileMotion(BaseTransform):
    def __init__(
        self,
        keys: Union[str, List[str]] = "motion",
        task_key: str = "task",
        # Prediction params
        pred_past_ratio: float = 0.4,
        # Inbetween params
        inbtw_past_ratio: float = 0.2,
        inbtw_future_ratio: float = 0.2,
        random_ratio: bool = False,
    ):
        assert keys, "Keys should not be empty."
        self.keys = [keys] if isinstance(keys, str) else keys
        self.task_key = task_key

        self.random_ratio = random_ratio
        self.pred_past_ratio = pred_past_ratio
        self.inbtw_past_ratio = inbtw_past_ratio
        self.inbtw_future_ratio = inbtw_future_ratio

    def _split_past_future(self, motion):
        num_frames = motion.shape[-2]

        if self.random_ratio:
            past_ratio = np.random.uniform(0.1, 0.5)
            past_frames = max(4, int(num_frames * past_ratio))
        else:
            past_frames = int(num_frames * self.pred_past_ratio)

        past_frames = min(past_frames, num_frames - 4)
        return (
            (motion[..., :past_frames, :], motion[..., past_frames:, :]),
            past_frames / num_frames,
            None,
        )

    def _split_past_middle_future(self, motion):
        num_frames = motion.shape[-2]

        if self.random_ratio:
            past_ratio = np.random.uniform(0.1, 0.25)
            future_ratio = np.random.uniform(0.1, 0.25)
            past_frames = max(4, int(num_frames * past_ratio))
            future_frames = max(4, int(num_frames * future_ratio))
        else:
            past_frames = int(num_frames * self.inbtw_past_ratio)
            future_frames = int(num_frames * self.inbtw_future_ratio)

        past_frames = min(past_frames, num_frames - future_frames - 4)
        future_frames = min(future_frames, num_frames - past_frames - 4)

        return (
            (
                motion[..., :past_frames, :],
                motion[..., past_frames:-future_frames, :],
                motion[..., -future_frames:, :],
            ),
            past_frames / num_frames,
            future_frames / num_frames,
        )

    def transform(self, results):
        if results[self.task_key] == MotionPrediction:
            split_fn = self._split_past_future
            required_segments = 2
        elif results[self.task_key] == MotionInbetween:
            split_fn = self._split_past_middle_future
            required_segments = 3
        else:
            return results

        for key in self.keys:
            if key not in results:
                continue

            split_result, past_ratio, future_ratio = split_fn(results[key])

            if required_segments == 2:
                results[f"past_{key}"], results[f"future_{key}"] = split_result
                results[f"past_{key}_num_frames"] = split_result[0].shape[-2]
                results[f"future_{key}_num_frames"] = split_result[1].shape[-2]
            else:
                (
                    results[f"past_{key}"],
                    results[f"middle_{key}"],
                    results[f"future_{key}"],
                ) = split_result
                results[f"past_{key}_num_frames"] = split_result[0].shape[-2]
                results[f"middle_{key}_num_frames"] = split_result[1].shape[-2]
                results[f"future_{key}_num_frames"] = split_result[2].shape[-2]

            results["past_ratio"] = past_ratio
            if future_ratio is not None:
                results["future_ratio"] = future_ratio

            assert all(
                tensor.numel() > 0 for tensor in split_result
            ), f"Split segments for {key} cannot be empty"

        return results


@TRANSFORMS.register_module(force=True)
class PrepareM2MCompletion(BaseTransform):
    """Convert full motion into M2M src/tgt/mask format for completion training.

    This transform randomly splits motion into past/middle/future segments
    (in-between style) and produces:
      - ``src_motion``: full motion (all frames)
      - ``tgt_motion``: full motion (same as src_motion, for loss computation)
      - ``src_mask``: binary mask (T, D), 1=needs generation (middle), 0=keep
      - ``tgt_length``: number of valid frames
      - ``src_length``: same as tgt_length

    Designed for use with :class:`HyMotionM2MTrainer` which uses
    VACE-style conditioning (src_motion * mask → inactive / reactive).

    Parameters
    ----------
    key : str
        Motion key in results dict.
    past_ratio : float
        Fraction of frames for the past segment.
    future_ratio : float
        Fraction of frames for the future segment.
    random_ratio : bool
        If True, randomly sample ratios from [0.1, 0.3].
    min_edge_frames : int
        Minimum frames for past/future.
    min_middle_frames : int
        Minimum frames for the middle (masked) region.
    """

    def __init__(
        self,
        key: str = 'motion',
        past_ratio: float = 0.2,
        future_ratio: float = 0.2,
        random_ratio: bool = True,
        min_edge_frames: int = 4,
        min_middle_frames: int = 4,
    ):
        self.key = key
        self.past_ratio = past_ratio
        self.future_ratio = future_ratio
        self.random_ratio = random_ratio
        self.min_edge_frames = min_edge_frames
        self.min_middle_frames = min_middle_frames

    def transform(self, results: Dict) -> Dict:
        motion = results[self.key]
        assert isinstance(motion, torch.Tensor), f"Expected Tensor, got {type(motion)}"

        T = motion.shape[-2]
        D = motion.shape[-1]

        # Determine split points
        if self.random_ratio:
            pr = float(np.random.uniform(0.1, 0.3))
            fr = float(np.random.uniform(0.1, 0.3))
        else:
            pr = self.past_ratio
            fr = self.future_ratio

        past_frames = max(self.min_edge_frames, int(T * pr))
        future_frames = max(self.min_edge_frames, int(T * fr))

        # Ensure middle region is at least min_middle_frames
        total_edge = past_frames + future_frames
        if total_edge + self.min_middle_frames > T:
            # Shrink edges proportionally
            available = max(0, T - self.min_middle_frames)
            past_frames = max(1, available // 2)
            future_frames = max(1, available - past_frames)

        past_frames = int(np.clip(past_frames, 1, T - 2))
        future_frames = int(np.clip(future_frames, 1, T - past_frames - 1))

        # Build mask: 0 for known (past/future), 1 for unknown (middle)
        mask = torch.zeros(T, D, dtype=motion.dtype)
        mask[past_frames:T - future_frames] = 1.0

        results['src_motion'] = motion.clone()
        results['tgt_motion'] = motion.clone()
        results['src_mask'] = mask
        results['tgt_length'] = T
        results['src_length'] = T

        return results
