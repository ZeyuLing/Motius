"""Inference-only SMPL-22 preprocessing used by the VerMo motion tokenizer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from torch import nn

from motius.registry import MODELS


@MODELS.register_module()
class VermoSMPL22Processor(nn.Module):
    """Normalize VerMo's 138-D SMPL-22 ``abs_rel`` representation.

    The released VerMo M2T tokenizer has ``use_static=False``. It only needs
    translation rollout and the training statistics; loading a full SMPL-X
    mesh model in the captioning path is unnecessary.
    """

    def __init__(
        self,
        stats_file: str,
        do_normalize: bool = True,
        rot_type: str = "rotation_6d",
        transl_type: str = "abs_rel",
        smpl_type: str = "smpl_22",
        eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__()
        if rot_type not in {"rotation_6d", "rot6d"}:
            raise ValueError(f"VerMo M2T expects rotation_6d, got {rot_type!r}")
        if transl_type != "abs_rel":
            raise ValueError(f"VerMo M2T expects abs_rel translation, got {transl_type!r}")
        if smpl_type != "smpl_22":
            raise ValueError(f"VerMo M2T expects smpl_22, got {smpl_type!r}")
        self.stats_file = str(Path(stats_file).expanduser().resolve())
        self.do_normalize = bool(do_normalize)
        self.transl_type = transl_type
        self.eps = float(eps)
        with open(self.stats_file, encoding="utf-8") as handle:
            stats = json.load(handle)
        mean = np.concatenate(
            [
                stats["transl"]["mean"],
                stats["transl_vel"]["mean"],
                stats["global_orient"]["rotation_6d"]["mean"],
                stats["body_pose"]["rotation_6d"]["mean"],
            ]
        ).astype(np.float32)
        std = np.concatenate(
            [
                stats["transl"]["std"],
                stats["transl_vel"]["std"],
                stats["global_orient"]["rotation_6d"]["std"],
                stats["body_pose"]["rotation_6d"]["std"],
            ]
        ).astype(np.float32)
        if mean.shape != (138,) or std.shape != (138,):
            raise ValueError(
                f"Expected VerMo SMPL-22 stats with 138 channels, got {mean.shape}."
            )
        self.register_buffer("mean", torch.from_numpy(mean), persistent=True)
        self.register_buffer(
            "std", torch.from_numpy(std).clamp_min(self.eps), persistent=True
        )

    def normalize(self, motion: torch.Tensor) -> torch.Tensor:
        if not self.do_normalize:
            return motion
        return (motion - self.mean.to(motion)) / self.std.to(motion)

    def denormalize(self, motion: torch.Tensor) -> torch.Tensor:
        if not self.do_normalize:
            return motion
        return motion * self.std.to(motion) + self.mean.to(motion)

    def inv_convert_transl(
        self,
        translation: Union[np.ndarray, torch.Tensor],
        transl_type: Optional[str] = None,
        use_rollout: Union[bool, str] = True,
    ):
        if (transl_type or self.transl_type) != "abs_rel":
            raise ValueError("VermoSMPL22Processor only supports abs_rel translation.")
        if use_rollout is False or use_rollout == "absolute":
            return translation[..., :3]
        first = translation[..., :1, :3]
        velocity = translation[..., 1:, 3:6]
        if isinstance(translation, torch.Tensor):
            absolute = torch.cumsum(torch.cat([first, velocity], dim=-2), dim=-2)
        else:
            absolute = np.cumsum(np.concatenate([first, velocity], axis=-2), axis=-2)
        return absolute

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self


__all__ = ["VermoSMPL22Processor"]
