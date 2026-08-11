from random import randint, uniform
from typing import Tuple, Dict, Union, List, Optional

import numpy as np
import torch
from mmcv import BaseTransform

from motius.registry import TRANSFORMS


@TRANSFORMS.register_module(force=True)
class RecomputeAbsRelTranslation(BaseTransform):
    """Rebuild relative translation after temporal crop or padding.

    ``LoadSmplx55`` packs translation as ``[absolute, relative]`` before later
    temporal transforms run. Cropping changes the sequence origin, while
    replicate/reflect padding changes the absolute trajectory, so the packed
    relative channels must be rebuilt from the final absolute sequence.
    """

    def __init__(self, keys: Union[str, List[str]] = "motion"):
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys

    @staticmethod
    def _recompute(motion):
        if motion.shape[-1] < 6:
            raise ValueError(
                "RecomputeAbsRelTranslation expects at least 6 channels, "
                f"got shape={tuple(motion.shape)}"
            )
        out = motion.clone() if isinstance(motion, torch.Tensor) else motion.copy()
        absolute = out[..., :3]
        relative = out[..., 3:6]
        relative[...] = 0
        relative[..., 1:, :] = absolute[..., 1:, :] - absolute[..., :-1, :]
        return out

    def transform(self, results: Dict):
        for key in self.keys:
            if key not in results:
                continue
            value = results[key]
            if isinstance(value, (list, tuple)):
                results[key] = [self._recompute(item) for item in value]
            else:
                results[key] = self._recompute(value)
        return results


@TRANSFORMS.register_module(force=True)
class CropMotionByTextTime(BaseTransform):
    """Crop motion using the selected text feature's time tags.

    The official HY-Motion T2M dataset samples one text embedding first, then
    crops the motion by that embedding's ``start_time`` / ``end_time`` tags
    before length filtering and padding.  This transform mirrors that behavior.
    """

    def __init__(
        self,
        keys: Union[str, List[str]] = "motion",
        fps_key: str = "fps",
        start_time_key: str = "caption_start_time",
        end_time_key: str = "caption_end_time",
        min_frame: int = 10,
        max_frame: int = 360,
    ):
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys
        self.fps_key = fps_key
        self.start_time_key = start_time_key
        self.end_time_key = end_time_key
        self.min_frame = int(min_frame)
        self.max_frame = int(max_frame)

    def transform(self, results: Dict):
        start_time = float(results.get(self.start_time_key, 0.0) or 0.0)
        end_time = float(results.get(self.end_time_key, 0.0) or 0.0)
        if start_time != start_time:
            start_time = 0.0
        if end_time != end_time:
            end_time = 0.0

        if start_time == 0.0 and end_time == 0.0:
            motion_length = None
            start_frame = 0
            end_frame = None
        else:
            fps = float(results.get(self.fps_key, 30.0) or 30.0)
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
            motion_length = end_frame - start_frame + 1
            if motion_length < self.min_frame or motion_length > self.max_frame:
                return None

        for key in self.keys:
            if key not in results:
                continue
            motion = results[key]
            if isinstance(motion, (list, tuple)):
                cropped = []
                for item in motion:
                    if motion_length is None:
                        cur = item
                    else:
                        cur = item[start_frame : end_frame + 1]
                    if cur.shape[0] < self.min_frame or cur.shape[0] > self.max_frame:
                        return None
                    cropped.append(cur)
                results[key] = cropped
                actual_length = cropped[0].shape[0] if cropped else 0
            else:
                if motion_length is None:
                    cur = motion
                else:
                    cur = motion[start_frame : end_frame + 1]
                if cur.shape[0] < self.min_frame or cur.shape[0] > self.max_frame:
                    return None
                results[key] = cur
                actual_length = cur.shape[0]

            results["num_frames"] = int(actual_length)
            results[f"{key}_num_frames"] = int(actual_length)

        results["text_crop_start_frame"] = int(start_frame)
        results["text_crop_end_frame"] = (
            int(end_frame) if end_frame is not None else int(results.get("num_frames", 0)) - 1
        )
        return results


@TRANSFORMS.register_module(force=True)
class HeadCropMotion(BaseTransform):
    """Deterministically keep the first ``clip_len`` frames of motion tensors."""

    def __init__(
        self,
        keys: Union[str, List[str]] = "motion",
        num_person_key: str = "num_person",
        clip_len: int = 300,
        start_frame_key: str = "start_frame",
    ):
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys
        self.num_person_key = num_person_key
        self.clip_len = int(clip_len)
        self.start_frame_key = start_frame_key

    @staticmethod
    def _crop_tensor(data: torch.Tensor, clip_len: int) -> torch.Tensor:
        if data.shape[-2] <= clip_len:
            return data
        return data[..., :clip_len, :]

    def transform(self, results: Dict):
        fps = results.get("fps")
        max_frames = 0
        for key in self.keys:
            if key not in results:
                continue
            data = results[key]
            if isinstance(data, (list, tuple)):
                cropped = [self._crop_tensor(item, self.clip_len) for item in data]
                results[key] = cropped
                if cropped:
                    max_frames = max(max_frames, max(int(item.shape[-2]) for item in cropped))
            else:
                cropped = self._crop_tensor(data, self.clip_len)
                results[key] = cropped
                max_frames = max(max_frames, int(cropped.shape[-2]))
            results[f"{key}_num_frames"] = max_frames

        if max_frames > 0:
            results["num_frames"] = max_frames
            if fps:
                results["duration"] = max_frames / float(fps)
            if "per_person_num_frames" in results:
                results["per_person_num_frames"] = np.asarray(
                    [min(int(t), self.clip_len) for t in results["per_person_num_frames"]],
                    dtype=np.int64,
                )
            results[self.start_frame_key] = 0
        return results


@TRANSFORMS.register_module(force=True)
class RandomCropPadding(BaseTransform):
    def __init__(
        self,
        keys: Union[str, List[str]] = "motion",
        num_person_key: str = "num_person",
        clip_len: int = 64,
        start_frame_key: str = "start_frame",
        allow_shorter: bool = True,
        allow_longer: bool = False,
        motion_path_key: str = "motion_path",
        # --- NEW: padding options for shorter sequences ---
        pad_mode: Optional[
            str
        ] = None,  # None/'none' | 'constant' | 'replicate' | 'reflect'
        pad_constant_value: float = 0.0,  # only used when pad_mode == 'constant'
        make_pad_mask: bool = False,  # export per-frame mask (1 for real, 0 for padded)
        pad_mask_key: str = "pad_mask",
    ):
        """
        Randomly crop a motion sequence to a fixed length; optionally pad if shorter.

        Args:
            allow_shorter: If False, raise on T < clip_len. If True, keep or pad (see pad_mode).
            pad_mode: None/'none' -> keep original shorter clip;
                      'constant' -> right-pad with constant (default 0.0);
                      'replicate' -> right-pad by repeating the last frame;
                      'reflect' -> right-pad by reflecting the sequence (bounce), T==1 falls back to replicate.
            make_pad_mask: If True, returns a boolean mask marking valid (1) vs padded (0) frames.
        """
        assert keys, "Keys should not be empty."
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys
        self.num_person_key = num_person_key
        self.clip_len = clip_len
        self.start_frame_key = start_frame_key
        self.allow_shorter = allow_shorter
        self.allow_longer = allow_longer
        self.motion_path_key = motion_path_key

        self.pad_mode = (pad_mode or "none").lower()
        assert self.pad_mode in (
            "none",
            "constant",
            "replicate",
            "reflect",
        ), f"Unsupported pad_mode={pad_mode}"
        self.pad_constant_value = pad_constant_value
        self.make_pad_mask = make_pad_mask
        self.pad_mask_key = pad_mask_key

    # --------- helpers ---------
    @staticmethod
    def random_crop(
        motion: torch.Tensor, clip_len: int = 64, start_frame: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """Crop a 1-person tensor [T, ...] to `clip_len` starting at `start_frame`."""
        if clip_len >= motion.shape[0]:
            # Not enough frames to crop: return as-is
            return motion, start_frame
        if start_frame is None:
            start_frame = randint(0, motion.shape[0] - clip_len)
        return motion[start_frame : start_frame + clip_len], start_frame

    def _raise_if_too_short(self, results: Dict, key: str, motion: torch.Tensor):
        """Raise ValueError when motion length < clip_len and allow_shorter=False."""
        if not self.allow_shorter and motion.shape[0] < self.clip_len:
            motion_path = (
                results.get(self.motion_path_key)
                or results.get(f"{key}_path")
                or results.get("path")
                or "UNKNOWN"
            )
            raise ValueError(
                f"[RandomCrop] sequence shorter than clip_len: "
                f"path={motion_path}, frames={motion.shape[0]}, required={self.clip_len}"
            )

    def _raise_if_too_long(self, results: Dict, key: str, motion: torch.Tensor):
        """Raise ValueError when motion length > clip_len and allow_longer=False."""
        if not self.allow_longer and motion.shape[0] > self.clip_len:
            motion_path = (
                results.get(self.motion_path_key)
                or results.get(f"{key}_path")
                or results.get("path")
                or "UNKNOWN"
            )
            raise ValueError(
                f"[RandomCrop] sequence longer than clip_len: "
                f"path={motion_path}, frames={motion.shape[0]}, required={self.clip_len}"
            )

    def _pad_tail_reflect(self, motion: torch.Tensor, needed: int) -> torch.Tensor:
        """
        Build a 'bounce' reflection tail of length `needed` from `motion` along dim 0.
        Pattern after the last frame: [T-2, ..., 0, 1, ..., T-2, 0, 1, ...]
        If T==1, fall back to replicate.
        """
        T = motion.shape[0]
        if needed <= 0:
            return motion
        if T == 1:
            # reflect is undefined for length-1; fall back to replicate
            tail = motion[-1:].repeat(needed, *([1] * (motion.dim() - 1)))
            return torch.cat([motion, tail], dim=0)

        # Build bounce index pattern once, then tile
        device = motion.device
        idx_fwd = list(range(T))  # [0, 1, ..., T-1]
        idx_bwd = list(range(T - 2, -1, -1))  # [T-2, ..., 0]
        bounce = idx_bwd + idx_fwd[1:-1]  # exclude endpoints duplication
        if len(bounce) == 0:
            # T==2 -> bounce = [0]
            bounce = [0]

        idx = torch.tensor(bounce, device=device, dtype=torch.long)
        reps = (needed + len(bounce) - 1) // len(bounce)
        idx = idx.repeat(reps)[:needed]
        tail = motion.index_select(0, idx)
        return torch.cat([motion, tail], dim=0)

    def _pad_to_len(
        self, motion: torch.Tensor, target_len: int
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Right-pad `motion` (shape [T, ...]) to `target_len` along time dim using `pad_mode`.
        Returns (padded_motion, pad_mask or None).
        """
        T = motion.shape[0]
        if T >= target_len or self.pad_mode == "none":
            mask = None
            if self.make_pad_mask:
                # mask always corresponds to current (possibly unpadded) sequence
                mask = torch.ones(T, dtype=torch.bool, device=motion.device)
                if T < target_len and self.pad_mode == "none":
                    # keep-as-is path: mask length == T
                    pass
            return motion, mask

        need = target_len - T
        mask = None
        if self.make_pad_mask:
            mask = torch.cat(
                [
                    torch.ones(T, dtype=torch.bool, device=motion.device),
                    torch.zeros(need, dtype=torch.bool, device=motion.device),
                ],
                dim=0,
            )

        if self.pad_mode == "constant":
            pad = torch.full(
                (need, *motion.shape[1:]),
                fill_value=self.pad_constant_value,
                dtype=motion.dtype,
                device=motion.device,
            )
            return torch.cat([motion, pad], dim=0), mask

        if self.pad_mode == "replicate":
            last = motion[-1:].repeat(need, *([1] * (motion.dim() - 1)))
            return torch.cat([motion, last], dim=0), mask

        if self.pad_mode == "reflect":
            # use custom bounce reflection tail
            padded = self._pad_tail_reflect(motion, need)
            return padded, mask

        # should not reach here
        raise ValueError(f"Unsupported pad_mode={self.pad_mode}")

    # --------- main transform ---------
    def transform(self, results: Dict):
        start_frame: Optional[int] = None
        num_person = results.get(self.num_person_key, None)

        for key in self.keys:
            if key not in results:
                continue
            data = results[key]

            if num_person is not None and num_person > 1:
                # data: list/tuple of per-person [T, ...]
                assert (
                    len(data) == num_person
                ), f"[RandomCrop] Expected {num_person} per-person data under key '{key}', got {len(data)}."
                # Early error if disallowed
                for m in data:
                    self._raise_if_too_short(results, key, m)
                    self._raise_if_too_long(results, key, m)

                # choose one start when cropping is possible
                if data is not None and data[0].shape[0] >= self.clip_len:
                    if start_frame is None:
                        start_frame = randint(0, data[0].shape[0] - self.clip_len)

                # crop
                cropped_pairs = [
                    self.random_crop(m, self.clip_len, start_frame) for m in data
                ]
                cropped_tensors, start_frames = zip(*cropped_pairs)
                chosen_start = start_frames[0] if start_frames else start_frame

                # pad (only if needed & configured)
                padded_list, mask_list = [], []
                for m in cropped_tensors:
                    # for cropped tensor, update num_frames, for padded tensor, no need to update
                    results["num_frames"] = m.shape[0]
                    results[f"{key}_num_frames"] = m.shape[0]
                    m_pad, m_mask = self._pad_to_len(m, self.clip_len)
                    padded_list.append(m_pad)
                    if self.make_pad_mask:
                        # ensure mask exists and is length clip_len
                        if m_mask is None:
                            # if no padding/mode=none and T>=clip_len, synthesize clip_len ones
                            mm = torch.ones(
                                self.clip_len, dtype=torch.bool, device=m.device
                            )
                        else:
                            # if keep-as-is (T<clip_len & pad_mode='none'), m_mask is length T
                            # To keep stacking possible, extend to clip_len with zeros
                            if m_mask.numel() < self.clip_len:
                                extend = torch.zeros(
                                    self.clip_len - m_mask.numel(),
                                    dtype=torch.bool,
                                    device=m_mask.device,
                                )
                                mm = torch.cat([m_mask, extend], dim=0)
                            else:
                                mm = m_mask
                        mask_list.append(mm)

                stacked = torch.stack(
                    padded_list, dim=0
                )  # [P, T', ...] ; T'==clip_len if padding enabled
                results[key] = stacked
                results[self.start_frame_key] = chosen_start
                # use temporal dim=1 after stacking
                if self.make_pad_mask:
                    results[self.pad_mask_key] = torch.stack(mask_list, dim=0)  # [P, T]

            else:
                # single-person: tensor [T, ...]
                motion = data
                self._raise_if_too_short(results, key, motion)

                cropped, start_frame = self.random_crop(
                    motion, self.clip_len, start_frame
                )

                results["num_frames"] = cropped.shape[0]
                results[f"{key}_num_frames"] = cropped.shape[0]

                padded, mask = self._pad_to_len(cropped, self.clip_len)

                results[key] = padded
                results[self.start_frame_key] = start_frame
                if self.make_pad_mask:
                    # ensure length equals num_frames
                    if mask is None or mask.numel() != padded.shape[0]:
                        # synthesize full-ones or extend with zeros
                        L = padded.shape[0]
                        if cropped.shape[0] == L:
                            mask = torch.ones(L, dtype=torch.bool, device=padded.device)
                        else:
                            mask = torch.cat(
                                [
                                    torch.ones(
                                        cropped.shape[0],
                                        dtype=torch.bool,
                                        device=padded.device,
                                    ),
                                    torch.zeros(
                                        L - cropped.shape[0],
                                        dtype=torch.bool,
                                        device=padded.device,
                                    ),
                                ],
                                dim=0,
                            )
                    results[self.pad_mask_key] = mask

        return results


@TRANSFORMS.register_module(force=True)
class RandomCrop(BaseTransform):
    def __init__(
        self,
        keys: Union[str, List[str]] = "motion",
        num_person_key: str = "num_person",
        clip_len: int = 64,
        start_frame_key: str = "start_frame",
        allow_shorter: bool = False,  # NEW: allow sequences shorter than clip_len
        motion_path_key: str = "motion_path",  # NEW: where to read the path for error messages
    ):
        """
        Randomly crop a motion sequence to a fixed length.

        Args:
            keys: Keys of motion sequences to be cropped.
            num_person_key: Key for the number of persons in the sample.
            clip_len: Target clip length.
            start_frame_key: Key to store the chosen start frame.
            allow_shorter: If False, raise an error when motion length < clip_len.
                           If True, keep the original behavior (return as-is).
            motion_path_key: Key in `results` that stores the motion file path
                             (used only for error messages).
        """
        assert keys, "Keys should not be empty."
        if not isinstance(keys, list):
            keys = [keys]
        self.keys = keys
        self.num_person_key = num_person_key
        self.clip_len = clip_len
        self.start_frame_key = start_frame_key
        self.allow_shorter = allow_shorter
        self.motion_path_key = motion_path_key

    @staticmethod
    def random_crop(
        motion: torch.Tensor, clip_len: int = 64, start_frame: Optional[int] = None
    ) -> Tuple[torch.Tensor, Optional[int]]:
        """Crop a 1-person tensor [T, ...] to `clip_len` starting at `start_frame`."""
        if clip_len >= motion.shape[0]:
            # Keep original behavior: return whole sequence and the (possibly None) start_frame
            return motion, start_frame
        if start_frame is None:
            start_frame = randint(0, motion.shape[0] - clip_len)
        return motion[start_frame : start_frame + clip_len], start_frame

    def _raise_if_too_short(self, results: Dict, key: str, motion: torch.Tensor):
        """Raise ValueError when motion length < clip_len and allow_shorter=False."""
        if not self.allow_shorter and motion.shape[0] < self.clip_len:
            motion_path = (
                results.get(self.motion_path_key)
                or results.get(f"{key}_path")
                or results.get("path")
                or "UNKNOWN"
            )
            raise ValueError(
                f"[RandomCrop] sequence shorter than clip_len: "
                f"path={motion_path}, frames={motion.shape[0]}, required={self.clip_len}"
            )

    def transform(self, results: Dict):
        start_frame: Optional[int] = None
        num_person = results.get(self.num_person_key, None)

        for key in self.keys:
            if key not in results:
                continue

            data = results[key]

            # Multi-person: expect an iterable of per-person tensors [T, ...]
            if num_person is not None and num_person > 1:
                assert isinstance(
                    data, (list, tuple)
                ), f"[RandomCrop] Expected list/tuple for multi-person under key '{key}'."

                # Length checks / early error if disallowed
                for m in data:
                    self._raise_if_too_short(results, key, m)

                # Choose a single start_frame if cropping is possible; otherwise pass None
                if data and data[0].shape[0] >= self.clip_len:
                    if start_frame is None:
                        start_frame = randint(0, data[0].shape[0] - self.clip_len)

                # Crop each person (use the same start_frame if set)
                cropped_pairs = [
                    self.random_crop(m, self.clip_len, start_frame) for m in data
                ]
                cropped_tensors, start_frames = zip(*cropped_pairs)
                # Prefer the first returned start_frame (others are equal when set)
                chosen_start = start_frames[0] if start_frames else start_frame
                cropped = torch.stack(list(cropped_tensors), dim=0)  # [P, T' or T, ...]

                results[key] = cropped
                results[self.start_frame_key] = chosen_start
                # Frames = temporal dim: for multi-person stacked as [P, T, ...], that is dim=1
                results["num_frames"] = (
                    cropped.shape[1] if cropped.dim() >= 2 else len(cropped)
                )
                results[f"{key}_num_frames"] = results["num_frames"]

            else:
                # Single-person: tensor [T, ...]
                motion = data
                self._raise_if_too_short(results, key, motion)

                cropped, start_frame = self.random_crop(
                    motion, self.clip_len, start_frame
                )
                results[key] = cropped
                results[self.start_frame_key] = start_frame
                results["num_frames"] = len(cropped)
                results[f"{key}_num_frames"] = len(cropped)
        return results


@TRANSFORMS.register_module()
class MotionAudioRandomCrop(BaseTransform):
    _SHARED_START_KEY = "_motion_audio_crop_start"
    _SHARED_DURATION_KEY = "_motion_audio_crop_duration"
    _SHARED_START_FRAME_KEY = "_motion_audio_crop_start_frame"
    _SHARED_NUM_FRAMES_KEY = "_motion_audio_crop_num_frames"
    _SHARED_APPLIED_KEY = "_motion_audio_crop_applied"

    def __init__(
        self,
        motion_key: str = "motion",
        audio_key: str = "audio",
        clip_duration: float = 5.0,
        duration_diff_threshold: float = 0.1,
        pair_only: bool = False,
        allow_caption_crop: bool = False,
    ):
        """
        :param motion_keys: motion key to crop
        :param audio_keys: audio key to crop
        :param clip_duration: clip motion and audio into clip_duration length
        :param fps: key to load fps value, or directly set fps value of motion
        :param sr: key to load sr value, or directly set sr value of audio
        no cropping is performed.
        """
        self.motion_key = motion_key
        self.audio_key = audio_key
        self.clip_duration = clip_duration
        self.duration_diff_thershold = duration_diff_threshold
        self.pair_only = pair_only
        self.allow_caption_crop = bool(allow_caption_crop)

    def _has_audio_source(self, results: Dict) -> bool:
        if self.audio_key is None:
            return False
        if results.get(self.audio_key) is not None:
            return True
        return results.get(f"{self.audio_key}_path") is not None

    @staticmethod
    def _task_has_caption(task_cls) -> bool:
        try:
            from motius.models.vermo.task_utils.modality import Caption

            for modal in task_cls.all_modality():
                if isinstance(modal, type) and issubclass(modal, Caption):
                    return True
                if isinstance(modal, Caption):
                    return True
        except Exception:
            return False
        return False

    def _get_or_plan_crop_window(self, results: Dict) -> Tuple[float, float]:
        fps = results["fps"]
        motion_num_frames = int(results[self.motion_key].shape[-2])
        motion_duration = motion_num_frames / fps
        generated_full_window = (
            results.get("_motion_audio_crop_metadata_explicit") is False
            and motion_duration > self.clip_duration
        )
        if (
            not generated_full_window
            and self._SHARED_START_KEY in results
            and self._SHARED_DURATION_KEY in results
        ):
            return (
                float(results[self._SHARED_START_KEY]),
                float(results[self._SHARED_DURATION_KEY]),
            )

        clip_num_frames = min(int(round(fps * self.clip_duration)), motion_num_frames)

        if motion_duration <= self.clip_duration:
            motion_start_frame = 0
            crop_num_frames = motion_num_frames
        else:
            max_start_frame = max(motion_num_frames - clip_num_frames, 0)
            motion_start_frame = randint(0, max_start_frame)
            crop_num_frames = clip_num_frames

        start_timestamp = motion_start_frame / fps
        crop_duration = crop_num_frames / fps
        results[self._SHARED_START_KEY] = float(start_timestamp)
        results[self._SHARED_DURATION_KEY] = float(crop_duration)
        results[self._SHARED_START_FRAME_KEY] = int(motion_start_frame)
        results[self._SHARED_NUM_FRAMES_KEY] = int(crop_num_frames)
        return float(start_timestamp), float(crop_duration)

    def check_align(self, results):
        """Align duration of audio and motion pairs.
        :param results: contains audio and motions
        :return:
        """
        motion_frames = results[self.motion_key].shape[-2]
        audio_frames = len(results[self.audio_key])

        fps = results["fps"]
        sr = results["sr"]
        motion_duration = motion_frames / fps
        audio_duration = audio_frames / sr
        assert (
            abs(motion_duration - audio_duration) <= self.duration_diff_thershold
        ), f"Check {results[f'{self.motion_key}_path']} and {results[f'{self.audio_key}_path']}, got motion_duration={motion_duration:.4f}, audio_duration={audio_duration:.4f}"

        return results

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        task_cls = results.get("task")
        motion_duration = results[self.motion_key].shape[-2] / results["fps"]
        has_caption_context = (
            (task_cls is not None and self._task_has_caption(task_cls))
            or results.get("caption") is not None
        )
        if has_caption_context and not self.allow_caption_crop:
            if motion_duration > self.clip_duration:
                raise ValueError(
                    "Caption-aligned sample exceeds clip_duration and cannot be "
                    f"safely cropped: motion_path={results.get(f'{self.motion_key}_path')}, "
                    f"duration={motion_duration:.3f}s, clip_duration={self.clip_duration:.3f}s"
                )
            return results

        if_crop_audio = self._has_audio_source(results)
        if self.pair_only and not if_crop_audio:
            return results

        audio_loaded = self.audio_key is not None and results.get(self.audio_key) is not None
        crop_applied = bool(results.get(self._SHARED_APPLIED_KEY, False))

        # Only validate full-length alignment before the shared crop window has
        # been applied. After that, later transforms may still see full audio
        # buffers for other modalities that need to be cropped with the same
        # window.
        if audio_loaded and not crop_applied:
            self.check_align(results)

        start_timestamp, crop_duration = self._get_or_plan_crop_window(results)
        motion_start_frame = int(results[self._SHARED_START_FRAME_KEY])
        crop_num_frames = int(results[self._SHARED_NUM_FRAMES_KEY])

        if not crop_applied:
            fps = results["fps"]
            motion_duration = results[self.motion_key].shape[-2] / fps

            if motion_duration > self.clip_duration:
                # motion shape: [T, C] or [P, T, C]
                motion = results[self.motion_key][
                    ...,
                    motion_start_frame : motion_start_frame + crop_num_frames,
                    :,
                ]
                assert motion.shape[-2] > 0, (
                    results[f"{self.motion_key}_path"],
                    start_timestamp,
                )
                results[self.motion_key] = motion
                crop_T = motion.shape[-2]
                results["num_frames"] = crop_T
                results["duration"] = crop_T / fps
                results[self._SHARED_NUM_FRAMES_KEY] = int(crop_T)
                results[self._SHARED_DURATION_KEY] = float(crop_T / fps)

                # Adjust per-person frame counts for the crop window.
                # After crop, each person's valid frames = min(original_valid - start, crop_T).
                if "per_person_num_frames" in results:
                    crop_end = motion_start_frame + crop_T
                    results["per_person_num_frames"] = np.array([
                        max(0, min(int(t), crop_end) - motion_start_frame)
                        for t in results["per_person_num_frames"]
                    ], dtype=np.int64)
            else:
                results["duration"] = motion_duration

            results[self._SHARED_APPLIED_KEY] = True

        if audio_loaded:
            # shape in [n], should be mono
            audio = results[self.audio_key]
            sr = results["sr"]
            audio_duration = len(audio) / sr if sr else 0.0
            audio_start_frame = int(round(start_timestamp * sr))
            expected_audio_frames = int(round(crop_duration * sr))

            if audio_duration > crop_duration:
                audio = audio[
                    audio_start_frame : audio_start_frame + expected_audio_frames
                ]
            if audio.shape[0] > expected_audio_frames:
                audio = audio[:expected_audio_frames]
            elif audio.shape[0] < expected_audio_frames:
                audio = np.pad(audio, (0, expected_audio_frames - audio.shape[0]))
            results[self.audio_key] = audio
            results[f"{self.audio_key}_duration"] = audio.shape[0] / sr
            results[f"{self.audio_key}_num_frames"] = audio.shape[0]

        return results


@TRANSFORMS.register_module(force=True)
class MotionAudioMaxDurationFilter(BaseTransform):
    """Reject samples whose motion exceeds a maximum duration.

    Unlike ``MotionAudioRandomCrop`` this transform never crops — it either
    passes through unchanged or raises ``ValueError`` (triggering the
    dataset's refetch logic) so that text–motion alignment is preserved.
    """

    def __init__(
        self,
        motion_key: str = "motion",
        audio_key: str = "audio",
        max_duration: float = 12.0,
        duration_diff_threshold: float = 0.1,
        pair_only: bool = False,
    ):
        self.motion_key = motion_key
        self.audio_key = audio_key
        self.max_duration = max_duration
        self.duration_diff_thershold = duration_diff_threshold
        self.pair_only = pair_only

    def transform(self, results: Dict) -> Dict:
        if_has_audio = self.audio_key in results and results.get(self.audio_key) is not None
        if self.pair_only and not if_has_audio:
            return results

        fps = results["fps"]
        motion_frames = results[self.motion_key].shape[-2]
        motion_duration = motion_frames / fps

        if motion_duration > self.max_duration:
            raise ValueError(
                f"MotionAudioMaxDurationFilter: motion duration {motion_duration:.2f}s "
                f"exceeds max_duration {self.max_duration:.2f}s, skipping sample"
            )

        # Validate audio alignment if present
        if if_has_audio:
            audio_frames = len(results[self.audio_key])
            sr = results["sr"]
            audio_duration = audio_frames / sr
            if abs(motion_duration - audio_duration) > self.duration_diff_thershold:
                raise ValueError(
                    f"MotionAudioMaxDurationFilter: motion ({motion_duration:.4f}s) "
                    f"and audio ({audio_duration:.4f}s) duration mismatch"
                )

        return results
