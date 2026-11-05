# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted for Motius component ownership; source provenance in licenses/motionbricks.
"""Load each clip once and share it."""

from dataclasses import dataclass
from pathlib import Path

import warp as wp

import motius.motion.io.motionbricks.bvh as bvh_utils
from motius.configuration.motionbricks_runtime_contract import ModelRig
from motius.motion.io.motionbricks.csv import load_csv
from motius.motion.runtime.motionbricks.animation_buffer import AnimationBuffer
from motius.motion.runtime.motionbricks.dof_animation_buffer import DOFAnimationBuffer
from motius.motion.runtime.motionbricks.skeleton import Skeleton


@dataclass(frozen=True)
class ClipAsset:
    """A source clip shared by target poses and playback. Its poses must not change."""

    path: Path
    animation: AnimationBuffer | DOFAnimationBuffer

    @property
    def name(self) -> str:
        """Clip name, from its file name without the suffix."""
        return self.path.stem

    @property
    def last_frame(self) -> int:
        """Index of the clip's last frame."""
        return self.animation.num_frames - 1

    @property
    def duration(self) -> float:
        """Clip duration in seconds."""
        return self.animation.duration


class ClipLoadError(RuntimeError):
    """Raised when a clip cannot be loaded for the registry's rig."""


class ClipRegistry:
    """Load each clip file once for one rig.

    BVH clips are remapped to ``skeleton``. CSV clips must match ``joint_names``.
    All clips use the registry's device. Edits such as loop correction go in
    separate buffers.
    """

    def __init__(
        self,
        rig: ModelRig,
        *,
        skeleton: Skeleton | None = None,
        joint_names: tuple[str, ...] = (),
        device=None,
    ) -> None:
        """Create a registry for one rig's clips.

        Args:
            rig: Rig of the clips.
            skeleton: Skeleton that SOMA clips are remapped to. SOMA requires it. Its
                device holds the clips.
            joint_names: Joint order that G1 clips must match. G1 requires it.
            device: Device for G1 clips. None uses the current device. SOMA ignores it.
        """
        self.rig = ModelRig(rig)
        if self.rig is ModelRig.SOMA:
            if skeleton is None or joint_names:
                raise ValueError("SOMA clips need a skeleton, without joint_names.")
            self.suffix = ".bvh"
            self.device = skeleton.reference_local_transforms.device
        else:
            if skeleton is not None or not joint_names:
                raise ValueError(
                    "Unitree G1 clips need joint_names, without a skeleton."
                )
            self.suffix = ".csv"
            self.device = wp.get_device(device)
        self.skeleton = skeleton
        self.joint_names = tuple(joint_names)
        self._clips: dict[Path, ClipAsset] = {}

    def load(self, path: str | Path) -> ClipAsset:
        """Return the clip at path, loading it on first use.

        Failures raise ClipLoadError.
        """
        path = Path(path).expanduser().resolve()
        if path.suffix.lower() != self.suffix:
            raise ClipLoadError(f"This registry loads {self.suffix} clips: {path}.")
        clip = self._clips.get(path)
        if clip is None:
            if not path.is_file():
                raise ClipLoadError(f"Could not find clip: {path}.")
            try:
                with wp.ScopedDevice(self.device):
                    if self.rig is ModelRig.SOMA:
                        animation = bvh_utils.load_bvh(path, self.skeleton)
                    else:
                        animation = load_csv(path, device=self.device)
                        if animation.joint_names != self.joint_names:
                            raise ValueError(
                                "CSV joint order must match the G1 model's joint order."
                            )
            except (KeyError, OSError, RuntimeError, ValueError) as exc:
                raise ClipLoadError(f"Could not load clip {path}: {exc}") from exc
            clip = ClipAsset(path, animation)
            self._clips[path] = clip
        return clip

    def load_looping(self, path: str | Path) -> AnimationBuffer | DOFAnimationBuffer:
        """Return a copy of the clip at path whose last frame joins its first.

        Each call returns a new copy, so the shared clip keeps its poses. Failures raise
        ClipLoadError.
        """
        looping = self.load(path).animation.copy()
        looping.make_loopable()
        return looping
