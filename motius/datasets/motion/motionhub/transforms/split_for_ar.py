"""
Transforms for autoregressive (with-init) task variants.

SplitMotionForAR  — splits motion into past (init frame) + future for m2d_ar / s2g_ar tasks.
SplitMusicForAR   — splits music into past (init segment) + future for d2m_ar task.
"""

from typing import Dict, List, Union

import numpy as np
import torch
from mmcv import BaseTransform

from motius.registry import TRANSFORMS


@TRANSFORMS.register_module()
class SplitMotionForAR(BaseTransform):
    """Split motion into past (initial frame(s)) + future for autoregressive tasks.

    Used by Music2DanceWithInit and Speech2GestureWithInit.

    When the assigned task is one of the ``target_tasks``, split
    ``results[key]`` along the time dimension into ``past_{key}`` and
    ``future_{key}``.

    Args:
        key: motion key in results (default "motion").
        single_frame_prob: probability of using exactly 1 frame as past
            (default 1.0 — the design doc specifies single-frame init).
        past_ratio: when not using single-frame, fraction of frames for past.
        min_future_frames: minimum frames kept for future.
        target_task_abbrs: abbreviations of tasks this transform applies to.
    """

    def __init__(
        self,
        key: str = "motion",
        single_frame_prob: float = 1.0,
        past_ratio: float = 0.1,
        min_future_frames: int = 8,
        target_task_abbrs: List[str] = None,
    ):
        self.key = key
        self.single_frame_prob = float(single_frame_prob)
        self.past_ratio = float(past_ratio)
        self.min_future_frames = int(min_future_frames)
        self.target_task_abbrs = set(target_task_abbrs or ["m2d_ar", "s2g_ar"])

    def _choose_past_frames(self, num_frames: int) -> int:
        if num_frames <= 1:
            raise ValueError(
                f"SplitMotionForAR: motion has only {num_frames} frame(s), "
                f"cannot split into past + future"
            )
        if num_frames <= self.min_future_frames + 1:
            return 1

        if self.single_frame_prob > 0.0 and np.random.rand() < self.single_frame_prob:
            return 1

        past_frames = max(1, int(num_frames * self.past_ratio))
        past_frames = int(np.clip(past_frames, 1, num_frames - self.min_future_frames))
        return past_frames

    def transform(self, results: Dict) -> Dict:
        task = results.get("task")
        if task is None:
            return results
        if getattr(task, "abbr", None) not in self.target_task_abbrs:
            return results

        motion = results[self.key]
        assert isinstance(motion, torch.Tensor), f"Expected Tensor, got {type(motion)}"

        num_frames = motion.shape[-2]
        past_frames = self._choose_past_frames(num_frames)

        past = motion[..., :past_frames, :]
        future = motion[..., past_frames:, :]

        assert past.shape[-2] > 0 and future.shape[-2] > 0, (
            f"SplitMotionForAR: invalid split — past {past.shape[-2]} frames, "
            f"future {future.shape[-2]} frames from {num_frames}-frame motion"
        )

        results[f"past_{self.key}"] = past
        results[f"future_{self.key}"] = future
        results["past_num_frames"] = int(past.shape[-2])
        results["future_num_frames"] = int(future.shape[-2])

        fps = results.get("fps", None)
        if fps is not None and not isinstance(fps, (list, tuple)):
            results["past_duration"] = past.shape[-2] / fps
            results["future_duration"] = future.shape[-2] / fps

        return results


@TRANSFORMS.register_module()
class SplitMusicForAR(BaseTransform):
    """Split music audio into past (initial segment) + future for autoregressive D2M.

    Used by Dance2MusicWithInit. Splits ``results["music"]`` (a 1-D waveform
    tensor) into ``past_music`` and ``future_music`` based on a temporal ratio.

    The split ratio is randomly sampled. The default is to use a short
    initial segment (10-30% of total) as the past condition.

    Args:
        key: audio key in results (default "music").
        past_ratio: default past fraction when not random.
        random_ratio: if True, sample past_ratio from [min_ratio, max_ratio].
        min_ratio: lower bound for random sampling.
        max_ratio: upper bound for random sampling.
        min_future_samples: minimum audio samples kept for future.
        target_task_abbrs: abbreviations of tasks this transform applies to.
    """

    def __init__(
        self,
        key: str = "music",
        past_ratio: float = 0.2,
        random_ratio: bool = True,
        min_ratio: float = 0.1,
        max_ratio: float = 0.3,
        min_future_samples: int = 4000,
        target_task_abbrs: List[str] = None,
    ):
        self.key = key
        self.past_ratio = float(past_ratio)
        self.random_ratio = bool(random_ratio)
        self.min_ratio = float(min_ratio)
        self.max_ratio = float(max_ratio)
        self.min_future_samples = int(min_future_samples)
        self.target_task_abbrs = set(target_task_abbrs or ["d2m_ar"])

    def _choose_split_point(self, total_samples: int) -> int:
        if self.random_ratio:
            ratio = np.random.uniform(self.min_ratio, self.max_ratio)
        else:
            ratio = self.past_ratio

        split = int(total_samples * ratio)
        # Ensure future has enough samples
        split = min(split, max(0, total_samples - self.min_future_samples))
        split = max(1, split)
        return split

    def transform(self, results: Dict) -> Dict:
        task = results.get("task")
        if task is None:
            return results
        if getattr(task, "abbr", None) not in self.target_task_abbrs:
            return results

        music = results.get(self.key)
        if music is None:
            return results

        assert isinstance(music, (torch.Tensor, np.ndarray)), (
            f"Expected Tensor or ndarray, got {type(music)}"
        )

        total_samples = music.shape[-1]
        split = self._choose_split_point(total_samples)

        results[f"past_{self.key}"] = music[..., :split]
        results[f"future_{self.key}"] = music[..., split:]

        return results
