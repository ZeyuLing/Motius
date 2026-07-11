"""KIMODO inference pipeline facade."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch

from motius.registry import PIPELINES


def _to_tensor(value, *, dtype=torch.float32, device=None):
    if isinstance(value, torch.Tensor):
        out = value
    else:
        out = torch.tensor(value, dtype=dtype)
    if dtype is not None:
        out = out.to(dtype=dtype)
    if device is not None:
        out = out.to(device)
    return out


_ROW_TO_COL = [0, 2, 4, 1, 3, 5]
_COL_TO_ROW = [0, 3, 1, 4, 2, 5]


def _rot6d_to_rotmat_row_major(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert row-major 6D rotations from ``motion_135`` to matrices."""
    d6 = rot6d[..., _ROW_TO_COL]
    x_raw = d6[..., 0:3]
    y_raw = d6[..., 3:6]
    x = torch.nn.functional.normalize(x_raw, dim=-1)
    z = torch.cross(x, y_raw, dim=-1)
    z = torch.nn.functional.normalize(z, dim=-1)
    y = torch.cross(z, x, dim=-1)
    return torch.stack([x, y, z], dim=-1)


def _rotmat_to_rot6d_row_major_np(rotmat: np.ndarray) -> np.ndarray:
    rotmat = np.asarray(rotmat, dtype=np.float32)
    col6d = np.concatenate([rotmat[..., 0:3, 0], rotmat[..., 0:3, 1]], axis=-1)
    return col6d[..., _COL_TO_ROW]


def _to_numpy_sequence(value) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    arr = np.asarray(value)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr.astype(np.float32)


def _fit_length(arr: np.ndarray, n_out: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) > n_out:
        return arr[:n_out]
    if 0 < len(arr) < n_out:
        pad = np.repeat(arr[-1:], n_out - len(arr), axis=0)
        return np.concatenate([arr, pad], axis=0)
    return arr


def _resample_nearest(arr: np.ndarray, n_out: int) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    if len(arr) == n_out or len(arr) < 2:
        return arr[:n_out]
    idx = np.rint(np.linspace(0, len(arr) - 1, n_out)).astype(np.int64)
    return arr[idx]


def _resample_positions(pos: np.ndarray, n_out: int) -> np.ndarray:
    pos = np.asarray(pos, dtype=np.float32)
    if len(pos) == n_out or len(pos) < 2:
        return pos[:n_out]
    src = np.linspace(0.0, 1.0, len(pos))
    dst = np.linspace(0.0, 1.0, n_out)
    flat = pos.reshape(len(pos), -1)
    out = np.empty((n_out, flat.shape[1]), dtype=np.float32)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(dst, src, flat[:, c])
    return out.reshape((n_out,) + pos.shape[1:])


def _split_num_frames(num_frames: int, safe_len: Optional[int] = None) -> list[int]:
    safe = int(safe_len or 180)
    if num_frames <= safe:
        return [int(num_frames)]
    chunks: list[int] = []
    remaining = int(num_frames)
    while remaining > 0:
        n = min(safe, remaining)
        chunks.append(n)
        remaining -= n
    return chunks


def _motion135_to_local_root(motion_135: np.ndarray, device: torch.device):
    motion = torch.from_numpy(np.asarray(motion_135, dtype=np.float32)).to(device)
    root = motion[:, :3]
    rot6d = motion[:, 3:135].reshape(len(motion), 22, 6)
    local = _rot6d_to_rotmat_row_major(rot6d)
    return local, root


def _make_prefix_constraint(model, gt_motion_135: np.ndarray, cond_frames: int):
    from motius.models.kimodo.network.constraints import FullBodyConstraintSet

    device = torch.device(model.device)
    skeleton = model.skeleton.to(device)
    local, root = _motion135_to_local_root(gt_motion_135, device)
    global_rots, positions, _ = skeleton.fk(local, root)

    k = max(1, min(int(cond_frames), int(len(gt_motion_135))))
    frame_idx = torch.arange(k, device=device, dtype=torch.long)
    smooth_root_2d = positions[frame_idx, skeleton.root_idx, :][:, [0, 2]]
    constraint = FullBodyConstraintSet(
        skeleton,
        frame_indices=frame_idx,
        global_joints_positions=positions[frame_idx],
        global_joints_rots=global_rots[frame_idx],
        smooth_root_2d=smooth_root_2d,
        to_crop=False,
    )
    return [constraint], local.detach().cpu().numpy(), root.detach().cpu().numpy()


def _recompute_debug_arrays(model, local_np: np.ndarray, root_np: np.ndarray) -> Dict[str, np.ndarray]:
    device = torch.device(model.device)
    skeleton = model.skeleton.to(device)
    local = torch.from_numpy(np.asarray(local_np, dtype=np.float32)).to(device)
    root = torch.from_numpy(np.asarray(root_np, dtype=np.float32)).to(device)
    global_rots, posed, _ = skeleton.fk(local, root)
    return {
        "local_rot_mats": local.detach().cpu().numpy().astype(np.float32),
        "global_rot_mats": global_rots.detach().cpu().numpy().astype(np.float32),
        "root_positions": root.detach().cpu().numpy().astype(np.float32),
        "posed_joints": posed.detach().cpu().numpy().astype(np.float32),
    }


@PIPELINES.register_module()
class KIMODOPipeline:
    """Unified KIMODO wrapper for text and kinematic-control generation."""

    def __init__(self, bundle):
        self.bundle = bundle

    @classmethod
    def from_config(cls, cfg: Optional[dict] = None, **kwargs):
        """Build a KIMODO pipeline from a bundle config."""
        from motius.models.kimodo import KIMODOBundle

        bundle = KIMODOBundle.from_config(cfg, **kwargs)
        return cls(bundle)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        """Build a KIMODO pipeline from a Motius artifact or KIMODO name."""
        from motius.models.kimodo import KIMODOBundle

        bundle = KIMODOBundle.from_pretrained(pretrained_model_name_or_path, **kwargs)
        return cls(bundle)

    @property
    def skeleton(self):
        return self.bundle.skeleton

    def __call__(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        if str(batch.get("task", "")).lower() == "tp2m":
            prompts = batch.get("prompt", batch.get("prompts", batch.get("caption")))
            gt_motions = batch.get("gt_motions_135", batch.get("gt_motion_135"))
            if prompts is None or gt_motions is None:
                raise ValueError("KIMODO TP2M batch needs prompts and gt_motions_135.")
            return {
                "samples": self.infer_tp2m(
                    prompts if isinstance(prompts, (list, tuple)) else [prompts],
                    gt_motions if isinstance(gt_motions, (list, tuple)) else [gt_motions],
                    condition_frames=int(batch.get("condition_frames", batch.get("condition_num_frames", 1))),
                    target_fps=float(batch.get("target_fps", 30.0)),
                    force_clean_prefix=bool(batch.get("force_clean_prefix", True)),
                    force_single_segment=bool(batch.get("force_single_segment", True)),
                    postprocess=batch.get("postprocess"),
                    max_segment_frames=batch.get("max_segment_frames"),
                )
            }
        prompts = batch.get("prompt", batch.get("prompts", batch.get("caption")))
        if prompts is None:
            raise ValueError("KIMODOPipeline batch needs prompt/prompts/caption.")
        num_frames = batch.get("num_frames")
        if num_frames is None:
            duration = batch.get("duration", batch.get("duration_sec"))
            if duration is None:
                raise ValueError("KIMODOPipeline batch needs num_frames or duration.")
            fps = float(getattr(self.bundle.model, "fps", 30))
            num_frames = int(float(duration) * fps)
        constraints = batch.get("constraints")
        return self.bundle.generate(
            prompts=prompts,
            num_frames=num_frames,
            constraints=constraints,
            multi_prompt=bool(batch.get("multi_prompt", False)),
            return_numpy=bool(batch.get("return_numpy", True)),
            **batch.get("generation_kwargs", {}),
        )

    def text_to_motion(self, prompt: str, num_frames: int, **kwargs) -> Dict[str, Any]:
        return self.bundle.generate(prompt, num_frames, constraints=None, **kwargs)

    def multi_prompt(
        self,
        prompts: Sequence[str],
        num_frames: Sequence[int],
        **kwargs,
    ) -> Dict[str, Any]:
        return self.bundle.generate(
            list(prompts),
            list(num_frames),
            constraints=None,
            multi_prompt=True,
            **kwargs,
        )

    def constrained_motion(
        self,
        prompt: str,
        num_frames: int,
        constraints: Sequence[Any],
        **kwargs,
    ) -> Dict[str, Any]:
        return self.bundle.generate(
            prompt,
            num_frames,
            constraints=list(constraints),
            **kwargs,
        )

    def infer_tp2m(
        self,
        prompts: Sequence[str],
        gt_motions_135: Sequence[np.ndarray],
        condition_frames: int,
        target_fps: float = 30.0,
        force_clean_prefix: bool = True,
        force_single_segment: bool = True,
        postprocess: Optional[bool] = None,
        max_segment_frames: Optional[int] = None,
    ) -> list[Dict[str, Any]]:
        """Generate KIMODO TP2M samples from text plus a GT ``motion_135`` prefix."""
        if len(prompts) != len(gt_motions_135):
            raise ValueError("prompts and gt_motions_135 must have equal length")
        if int(condition_frames) < 1:
            raise ValueError("condition_frames must be >= 1")

        model = self.bundle.model
        do_postprocess = self.bundle.post_processing if postprocess is None else bool(postprocess)
        outputs: list[Dict[str, Any]] = []
        for prompt, gt_motion in zip(prompts, gt_motions_135):
            gt = np.asarray(gt_motion, dtype=np.float32)
            num_frames_30 = int(len(gt))
            model_fps = float(model.fps)
            model_frames = max(10, int(round(num_frames_30 * model_fps / float(target_fps))))

            constraints, gt_local, gt_root = _make_prefix_constraint(model, gt, int(condition_frames))
            seg_lens = (
                [model_frames]
                if force_single_segment
                else _split_num_frames(model_frames, safe_len=max_segment_frames)
            )
            is_multi = len(seg_lens) > 1
            constraint_arg = constraints if is_multi else [constraints]
            raw = model(
                [str(prompt)] * len(seg_lens),
                seg_lens,
                num_denoising_steps=self.bundle.diffusion_steps,
                cfg_weight=[2.0, 2.0],
                num_samples=1,
                return_numpy=True,
                multi_prompt=is_multi,
                constraint_lst=constraint_arg,
                post_processing=do_postprocess,
            )

            local = _to_numpy_sequence(raw.get("local_rot_mats"))
            global_rot = _to_numpy_sequence(raw.get("global_rot_mats"))
            root = _to_numpy_sequence(raw.get("root_positions"))
            posed = _to_numpy_sequence(raw.get("posed_joints"))
            if local is None:
                raise KeyError("KIMODO output has no local_rot_mats")
            if root is None:
                if posed is None:
                    raise KeyError("KIMODO output has neither root_positions nor posed_joints")
                root = posed[:, 0]

            if abs(model_fps - float(target_fps)) > 1e-6:
                local = _resample_nearest(local, num_frames_30)
                root = _resample_positions(root[:, None, :], num_frames_30)[:, 0]
                if global_rot is not None:
                    global_rot = _resample_nearest(global_rot, num_frames_30)
                if posed is not None:
                    posed = _resample_positions(posed, num_frames_30)

            local = _fit_length(local, num_frames_30)
            root = _fit_length(root, num_frames_30)
            k = max(1, min(int(condition_frames), num_frames_30))
            if force_clean_prefix:
                local[:k] = gt_local[:k]
                root[:k] = gt_root[:k]
                payload = _recompute_debug_arrays(model, local, root)
            else:
                payload = {
                    "local_rot_mats": local.astype(np.float32),
                    "root_positions": root.astype(np.float32),
                }
                if global_rot is not None:
                    payload["global_rot_mats"] = _fit_length(global_rot, num_frames_30).astype(np.float32)
                if posed is not None:
                    payload["posed_joints"] = _fit_length(posed, num_frames_30).astype(np.float32)
                else:
                    payload.update(_recompute_debug_arrays(model, local, root))

            payload["motion_135"] = np.concatenate(
                [
                    payload["root_positions"],
                    _rotmat_to_rot6d_row_major_np(payload["local_rot_mats"]).reshape(num_frames_30, 132),
                ],
                axis=1,
            ).astype(np.float32)
            outputs.append(payload)
        return outputs

    def constraints_from_json(self, path_or_data, *, device=None, dtype=torch.float32):
        self.bundle.load_model()
        from motius.models.kimodo.network.constraints import load_constraints_lst

        return load_constraints_lst(
            str(path_or_data) if isinstance(path_or_data, Path) else path_or_data,
            self.skeleton,
            device=device,
            dtype=dtype,
        )

    def root2d_constraint(
        self,
        frame_indices,
        smooth_root_2d,
        global_root_heading: Optional[Any] = None,
        *,
        device=None,
    ):
        self.bundle.load_model()
        from motius.models.kimodo.network.constraints import Root2DConstraintSet

        return Root2DConstraintSet(
            self.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(smooth_root_2d, device=device),
            global_root_heading=(
                None
                if global_root_heading is None
                else _to_tensor(global_root_heading, device=device)
            ),
        )

    def fullbody_keyframe_constraint(
        self,
        frame_indices,
        global_joints_positions,
        global_joints_rots,
        smooth_root_2d: Optional[Any] = None,
        *,
        device=None,
    ):
        self.bundle.load_model()
        from motius.models.kimodo.network.constraints import FullBodyConstraintSet

        return FullBodyConstraintSet(
            self.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(global_joints_positions, device=device),
            _to_tensor(global_joints_rots, device=device),
            smooth_root_2d=(
                None
                if smooth_root_2d is None
                else _to_tensor(smooth_root_2d, device=device)
            ),
        )

    def end_effector_constraint(
        self,
        frame_indices,
        global_joints_positions,
        global_joints_rots,
        smooth_root_2d,
        joint_names: Sequence[str],
        *,
        device=None,
    ):
        self.bundle.load_model()
        from motius.models.kimodo.network.constraints import EndEffectorConstraintSet

        return EndEffectorConstraintSet(
            self.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(global_joints_positions, device=device),
            _to_tensor(global_joints_rots, device=device),
            _to_tensor(smooth_root_2d, device=device),
            joint_names=list(joint_names),
        )

    def left_hand_constraint(self, *args, **kwargs):
        return self._named_end_effector_constraint("LeftHandConstraintSet", *args, **kwargs)

    def right_hand_constraint(self, *args, **kwargs):
        return self._named_end_effector_constraint("RightHandConstraintSet", *args, **kwargs)

    def left_foot_constraint(self, *args, **kwargs):
        return self._named_end_effector_constraint("LeftFootConstraintSet", *args, **kwargs)

    def right_foot_constraint(self, *args, **kwargs):
        return self._named_end_effector_constraint("RightFootConstraintSet", *args, **kwargs)

    def _named_end_effector_constraint(
        self,
        class_name: str,
        frame_indices,
        global_joints_positions,
        global_joints_rots,
        smooth_root_2d,
        *,
        device=None,
    ):
        self.bundle.load_model()
        from motius.models.kimodo.network import constraints

        cls = getattr(constraints, class_name)
        return cls(
            self.skeleton,
            _to_tensor(frame_indices, dtype=torch.long, device=device),
            _to_tensor(global_joints_positions, device=device),
            _to_tensor(global_joints_rots, device=device),
            _to_tensor(smooth_root_2d, device=device),
        )
