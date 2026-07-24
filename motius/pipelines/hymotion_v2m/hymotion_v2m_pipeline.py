"""HyMotion-V2M inference pipeline (stage 1: pre-extracted feature -> motion).

Wraps ``HyMotionV2MBundle`` and reproduces the original ``V2MRuntime.run_v2m``
sliding-window inference:

  feature (T, Dctx) + camera_RT (T, 4, 4)
    -> windowed flow-matching ODE (``train_frames`` window, overlap blend)
    -> concat segments (translation-continuous)
    -> SMPL forward kinematics
    -> RANSAC floor fit + ground alignment
    -> {rot6d, transl, shapes, keypoints3d, ...}

Single-person / single-result minimal coverage: pass one feature stream and a
single seed.  Stage 2 (video -> feature) will call ``infer_from_feature`` after
running YOLOX + SAM-3D-Body preprocessing.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch import Tensor

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.models.hymotion_v2m.bundle import HyMotionV2MBundle


@PIPELINES.register_module()
class HyMotionV2MPipeline(BasePipeline):
    """Pipeline for HyMotion-V2M feature-to-motion generation."""

    BUNDLE_CLS = HyMotionV2MBundle

    def __init__(self, bundle: HyMotionV2MBundle, **kwargs):
        super().__init__(bundle, **kwargs)

    # ------------------------------------------------------------------
    # Segment concatenation (translation-continuous), ported from
    # V2MRuntime._concat_segment_outputs.
    # ------------------------------------------------------------------
    @staticmethod
    def _concat_segment_outputs(outputs: List[Dict[str, Tensor]], hop: int) -> Dict[str, Any]:
        """Stitch sliding-window segments into one continuous sequence.

        Uses a running accumulator with a *per-segment* effective overlap
        ``min(overlap, running_len, seg_len)``.  In the common case where every
        non-final segment is ``train_frames`` long and ``hop >= overlap`` this is
        identical to the original ``V2MRuntime._concat_segment_outputs``; the
        clamping only kicks in for ragged short tail segments (e.g. when
        ``hop < overlap``), keeping concatenation robust instead of crashing.
        """
        if len(outputs) == 0:
            return {}
        if hop <= 0:
            raise ValueError(f"hop must be > 0, got {hop}")

        ref_T = None
        for v in outputs[0].values():
            if isinstance(v, Tensor) and v.dim() >= 2:
                ref_T = int(v.shape[1])
                break
        if ref_T is None:
            return dict(outputs[0])

        overlap = max(0, ref_T - hop)

        ret: Dict[str, Any] = {}
        keys = list(outputs[0].keys())
        for k in keys:
            v0 = outputs[0][k]
            if not isinstance(v0, Tensor) or v0.dim() < 2:
                ret[k] = [out.get(k) for out in outputs]
                continue

            is_trans = k == "trans" and v0.dim() == 3 and int(v0.shape[-1]) == 3
            acc = v0
            for si in range(1, len(outputs)):
                v = outputs[si][k]
                ov_eff = min(overlap, int(acc.shape[1]), int(v.shape[1]))
                if is_trans:
                    if ov_eff > 0:
                        prev_tail = acc[:, -ov_eff:]
                        cur_head = v[:, :ov_eff]
                        delta = (prev_tail - cur_head).mean(dim=1)
                        v_shifted = v + delta.unsqueeze(1)
                        w = torch.linspace(
                            0.0, 1.0, steps=ov_eff + 2, device=v.device, dtype=v.dtype
                        )[1:-1].view(1, ov_eff, 1)
                        blended = (1.0 - w) * prev_tail + w * v_shifted[:, :ov_eff]
                        acc = torch.cat(
                            [acc[:, :-ov_eff], blended, v_shifted[:, ov_eff:]], dim=1
                        )
                    else:
                        delta = acc[:, -1] - v[:, 0]
                        v_shifted = v + delta.unsqueeze(1)
                        acc = torch.cat([acc, v_shifted], dim=1)
                else:
                    acc = torch.cat([acc, v[:, ov_eff:]], dim=1)
            ret[k] = acc

        ret["_segments"] = len(outputs)
        ret["_hop"] = hop
        ret["_overlap_drop"] = overlap
        return ret

    @staticmethod
    def _trim_temporal_outputs(
        output: Dict[str, Any],
        length: int,
    ) -> Dict[str, Any]:
        """Trim decoded per-frame fields to the source sequence length."""
        temporal_keys = (
            "rot6d",
            "trans",
            "global_orient",
            "local_transl_vel",
            "end_effector_vel",
        )
        trimmed = dict(output)
        for key in temporal_keys:
            value = trimmed.get(key)
            if isinstance(value, Tensor):
                if value.dim() < 2 or int(value.shape[1]) < length:
                    raise ValueError(
                        f"{key} cannot cover {length} frames: "
                        f"shape={tuple(value.shape)}"
                    )
                trimmed[key] = value[:, :length]
        return trimmed

    @staticmethod
    def _require_finite_window(
        output: Dict[str, Any],
        *,
        start: int,
        valid_length: int,
    ) -> None:
        """Reject unstable windows before overlap blending can hide the source."""
        diagnostics = {}
        for key, value in output.items():
            if not isinstance(value, Tensor) or not value.is_floating_point():
                continue
            count = int((~torch.isfinite(value)).sum().item())
            if count:
                diagnostics[key] = {
                    "shape": list(value.shape),
                    "nonfinite": count,
                }
        if diagnostics:
            raise RuntimeError(
                "HYMotion-V2M produced non-finite values before stitching: "
                f"window_start={start}, valid_length={valid_length}, "
                f"diagnostics={diagnostics}"
            )

    @staticmethod
    def _compute_wv_transform(camera_R0: Tensor) -> Tensor:
        """World->WV (gravity-aligned) rotation from the first-frame camera.

        Verbatim port of ``MotionCameraTransform._compute_wv_transform`` in the
        source repo: +y is gravity-up, +z is the camera-0 forward direction
        projected onto the ground plane.
        """
        axis_z_in_c = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        axis_z_in_w = camera_R0.t() @ axis_z_in_c
        axis_up_in_w = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)
        axis_newx_in_w = torch.cross(axis_up_in_w, axis_z_in_w, dim=-1)
        axis_newx_in_w = axis_newx_in_w / axis_newx_in_w.norm(dim=-1, keepdim=True)
        axis_newz_in_w = torch.cross(axis_newx_in_w, axis_up_in_w, dim=-1)
        axis_newz_in_w = axis_newz_in_w / axis_newz_in_w.norm(dim=-1, keepdim=True)
        relative_transform = torch.stack(
            [axis_newx_in_w, axis_up_in_w, axis_newz_in_w], dim=-1
        ).t()
        return relative_transform

    @classmethod
    def _prepare_camera_conditioning(cls, camera_RT: Tensor) -> tuple:
        """Convert raw world->camera extrinsics into the model's conditioning.

        Mirrors the demo/eval dataset path
        (``transform_camera_to_wv`` + ``_compute_camera_transforms``):

          - ``camera_R`` = R_to_first_frame  (T, 9), rotation of camera t w.r.t.
            camera 0 in the WV frame.
          - ``camera_T`` = camera-center velocity in WV * 30  (T, 3).

        ``transl0`` only offsets the camera center by a constant and therefore
        cancels out in the velocity (a finite difference), so it is dropped here.
        """
        camera_RT = torch.as_tensor(camera_RT, dtype=torch.float32)
        rrel = cls._compute_wv_transform(camera_RT[0, :3, :3])
        trans = torch.eye(4, dtype=torch.float32)
        trans[:3, :3] = rrel  # transl0 == 0
        camera_rt_wv = camera_RT @ torch.inverse(trans)

        rot = camera_rt_wv[:, :3, :3]
        r_to_first = rot @ rot[0].t()[None]  # (T, 3, 3)
        # camera center expressed in the WV frame
        t_camera_wv = torch.einsum(
            "tij,tj->ti", rot.transpose(1, 2), -camera_rt_wv[:, :3, 3]
        )
        center_velocity = t_camera_wv[1:] - t_camera_wv[:-1]
        center_velocity = torch.cat([center_velocity, center_velocity[-1:]], dim=0)

        camera_r = r_to_first.reshape(-1, 9)
        camera_t = center_velocity * 30.0
        return camera_r, camera_t

    @staticmethod
    def _build_feature_window(
        feat_window: Tensor,
        cam_r_window: Tensor,
        cam_t_window: Tensor,
        train_frames: int,
        device: torch.device,
    ) -> Dict[str, Tensor]:
        T, _D = feat_window.shape
        if T == 0:
            raise ValueError("Empty window: feat_window has length 0")
        if T < train_frames:
            pad_len = train_frames - T
            feat_window = torch.cat(
                [feat_window, feat_window[-1:].expand(pad_len, -1)], dim=0
            )
            cam_r_window = torch.cat(
                [cam_r_window, cam_r_window[-1:].expand(pad_len, -1)], dim=0
            )
            cam_t_window = torch.cat(
                [cam_t_window, cam_t_window[-1:].expand(pad_len, -1)], dim=0
            )
        else:
            feat_window = feat_window[:train_frames]
            cam_r_window = cam_r_window[:train_frames]
            cam_t_window = cam_t_window[:train_frames]
        return {
            "feature": feat_window.unsqueeze(0).to(device),
            "camera_R": cam_r_window.unsqueeze(0).to(device),
            "camera_T": cam_t_window.unsqueeze(0).to(device),
        }

    @staticmethod
    def _forward_smpl_batch(
        body_model,
        rot6d: Tensor,  # (B, L, J, 6)
        shapes: Tensor,  # (B, L, 16) or (B, 1, 16)
        trans: Tensor,  # (B, L, 3)
    ) -> Tensor:
        B, L = trans.shape[:2]
        J = rot6d.shape[2]
        # decode returns shapes averaged over time as (B, 1, 16); expand to L so
        # the per-frame FK batch matches rot6d/trans.  betas are constant.
        if shapes.shape[1] != L:
            shapes = shapes.expand(B, L, shapes.shape[-1])
        rot6d_flat = rot6d.reshape(B * L, J, 6)
        shapes_flat = shapes.reshape(B * L, shapes.shape[-1])
        trans_flat = trans.reshape(B * L, 3)
        out = body_model(
            {"rot6d": rot6d_flat, "shapes": shapes_flat, "trans": trans_flat}
        )
        k3d = out["keypoints3d"]
        return k3d.reshape(B, L, k3d.shape[1], 3)

    @staticmethod
    def _fit_floor_height(k3d: Tensor, method: str = "ransac", axis: str = "y") -> Tensor:
        if axis not in ("x", "y", "z"):
            raise ValueError(f"axis must be one of x/y/z, got {axis}")
        axis_idx = {"x": 0, "y": 1, "z": 2}[axis]
        if k3d.numel() == 0:
            raise ValueError("k3d is empty")
        axis_values = k3d[..., axis_idx]
        finite = torch.isfinite(axis_values)
        zs = axis_values.masked_fill(~finite, torch.inf).amin(dim=-1)
        zs = zs[torch.isfinite(zs)]
        if zs.numel() == 0:
            raise ValueError("k3d has no finite coordinates for floor fitting")
        if method == "lowest":
            offset = zs.amin()
        elif method == "average":
            offset = zs.mean()
        elif method == "ransac":
            zs1 = zs.reshape(-1)
            zs1, _ = torch.sort(zs1)
            alpha = 1.0
            min_z = zs1.min()
            max_z = zs1.max()
            zs1 = zs1[zs1 <= min_z + (max_z - min_z) * alpha]
            inlier_thresh = 0.05
            best_inliers = -1
            best_z = zs1[0]
            n = zs1.numel()
            for _ in range(10_000):
                z = zs1[torch.randint(0, n, (1,), device=zs1.device)]
                inliers = (zs1 - z).abs() < inlier_thresh
                cnt = int(inliers.sum().item())
                if cnt > best_inliers:
                    best_inliers = cnt
                    best_z = z
            offset = zs1[(zs1 - best_z).abs() < inlier_thresh].median()
        else:
            raise ValueError(f"Unknown method: {method}")
        height_offset = torch.zeros(3, device=k3d.device, dtype=k3d.dtype)
        height_offset[axis_idx] = offset
        return height_offset

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer_from_feature(
        self,
        feature: Tensor,
        camera_RT: Optional[Tensor] = None,
        camera_K: Optional[Tensor] = None,
        seeds: Optional[List[int]] = None,
        cfg_scale: float = 1.0,
        camera_is_static: bool = True,
        overlap_frames: int = 30,
        floor_method: str = "ransac",
        floor_axis: str = "y",
        ground_align: bool = True,
    ) -> Dict[str, Any]:
        """Generate single-person SMPL motion from a pre-extracted feature stream.

        Args:
            feature: (T, Dctx) SAM-3D feature tokens (Dctx defaults to 1024).
            camera_RT: (T, 4, 4) world->camera extrinsics.  Identity if None.
            camera_K: (T, 3, 3) intrinsics, stored in the output if given.
            seeds: list of seeds; one sample per seed (default ``[0]``).
            cfg_scale: classifier-free guidance scale.
            camera_is_static: whether the camera is static.
            overlap_frames: sliding-window overlap (frames).
            floor_method / floor_axis: floor-fit configuration.
            ground_align: subtract the fitted floor height from translation.

        Returns:
            dict with ``rot6d`` (B,L,J,6), ``transl`` (B,L,3, grounded),
            ``shapes`` (B,1,16), ``keypoints3d`` (B,L,J,3, grounded),
            ``trans_raw``, ``end_vel``, ``height_offset``, ``K``, ``RT``.
        """
        if seeds is None:
            seeds = [0]
        bundle = self.bundle
        bundle.eval()
        model = bundle.model
        device = next(model.parameters()).device
        train_frames = int(bundle.train_frames)

        feature = torch.as_tensor(feature, dtype=torch.float32)
        if feature.dim() != 2:
            raise ValueError(f"feature must be (T, D), got {tuple(feature.shape)}")
        total_len = int(feature.shape[0])

        if camera_RT is None:
            camera_RT = torch.eye(4).unsqueeze(0).repeat(total_len, 1, 1)
        camera_RT = torch.as_tensor(camera_RT, dtype=torch.float32)
        if camera_RT.shape[0] < total_len:
            pad = camera_RT[-1:].expand(total_len - camera_RT.shape[0], 4, 4)
            camera_RT = torch.cat([camera_RT, pad], dim=0)

        # Convert raw extrinsics into the model's (R_to_first_frame, velocity*30)
        # conditioning over the *full* sequence, then slice per window.  Identity
        # extrinsics collapse to camera_R == I, camera_T == 0 (static-camera case).
        cam_r, cam_t = self._prepare_camera_conditioning(camera_RT)

        # sliding window hop
        if total_len > train_frames:
            ov = min(max(0, int(overlap_frames)), max(0, train_frames - 1))
            hop = train_frames - ov
            if hop <= 0:
                hop = train_frames
        else:
            hop = train_frames

        outputs: List[Dict[str, Tensor]] = []
        start = 0
        while start < total_len:
            feat_window = feature[start : start + train_frames]
            cam_r_window = cam_r[start : start + train_frames]
            cam_t_window = cam_t[start : start + train_frames]
            cur_len = int(feat_window.shape[0])
            feat_dict = self._build_feature_window(
                feat_window, cam_r_window, cam_t_window, train_frames, device
            )
            out = bundle.generate_from_feature(
                feature=feat_dict,
                seeds=seeds,
                length=cur_len,
                camera_is_static=camera_is_static,
                cfg_scale=cfg_scale,
                do_postproc=False,
            )
            self._require_finite_window(
                out,
                start=start,
                valid_length=cur_len,
            )
            outputs.append(out)
            start += hop

        model_output = (
            outputs[0] if len(outputs) == 1 else self._concat_segment_outputs(outputs, hop=hop)
        )
        model_output = self._trim_temporal_outputs(model_output, total_len)

        B = int(model_output["trans"].shape[0])
        L = int(model_output["trans"].shape[1])

        k3d = self._forward_smpl_batch(
            bundle.body_model,
            rot6d=model_output["rot6d"].clone(),
            shapes=model_output["shapes"].clone(),
            trans=model_output["trans"].clone(),
        )

        height_offset = torch.zeros(3, device=k3d.device, dtype=k3d.dtype)
        if ground_align:
            height_offset = self._fit_floor_height(
                k3d, method=floor_method, axis=floor_axis
            )

        trans_grounded = model_output["trans"] - height_offset.to(model_output["trans"].device)
        k3d_grounded = k3d - height_offset.to(k3d.device)

        result: Dict[str, Any] = {
            "rot6d": model_output["rot6d"].detach().cpu(),
            "transl": trans_grounded.detach().cpu(),
            "trans_raw": model_output["trans"].detach().cpu(),
            "shapes": model_output["shapes"].detach().cpu(),
            "keypoints3d": k3d_grounded.detach().cpu(),
            "global_orient": model_output["global_orient"].detach().cpu(),
            "end_vel": model_output["end_effector_vel"].detach().cpu(),
            "height_offset": height_offset.detach().cpu(),
            "camera_RT": camera_RT[:L].detach().cpu(),
        }
        if camera_K is not None:
            result["camera_K"] = torch.as_tensor(camera_K, dtype=torch.float32)[:L].cpu()
        return result

    # ------------------------------------------------------------------
    # End-to-end: raw video -> motion (stage 2)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def infer_v2m(
        self,
        video_path: str,
        *,
        work_dir: str = "outputs/tmp/v2m_e2e",
        max_frames: Optional[int] = None,
        transcode: bool = True,
        seeds: Optional[List[int]] = None,
        cfg_scale: float = 1.0,
        overlap_frames: int = 30,
        floor_method: str = "ransac",
        floor_axis: str = "y",
        ground_align: bool = True,
        bbox_xyxy: Optional[Tensor] = None,
        preprocessor: Optional[Any] = None,
        preprocessor_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """End-to-end video-to-motion: ``mp4 -> motion`` (single best person).

        Runs the stage-2 front end (ffmpeg transcode -> YOLOX + ByteTrack ->
        SAM-3D-Body per-frame tokens -> pinhole camera) and feeds the resulting
        feature stream into :meth:`infer_from_feature`.

        The token dimension is taken from the loaded checkpoint
        (``bundle.feature_dim``: 1024 body-only, 3072 with hands), so the hand
        variant automatically extracts hand tokens too.

        Heavy stage-2 dependencies (ffmpeg / yolox / supervision / sam_3d_body
        + the gated SAM-3D-Body weights) are imported lazily by the
        preprocessor; a missing one raises ``V2MDependencyError`` with the exact
        fix.  Override resource paths via ``preprocessor_kwargs`` or the
        ``HYMOTION_V2M_*`` environment variables.

        Returns the same dict as :meth:`infer_from_feature`, plus ``bbox_xyxy``
        (T, 4) and ``video_path`` (the transcoded clip actually used).
        """
        from motius.models.hymotion_v2m.preprocess import (
            V2MVideoPreprocessor,
        )

        device = next(self.bundle.model.parameters()).device
        if preprocessor is None:
            preprocessor = V2MVideoPreprocessor(
                device=str(device), **(preprocessor_kwargs or {})
            )

        pre = preprocessor.run(
            video_path,
            token_dim=int(self.bundle.feature_dim),
            work_dir=work_dir,
            max_frames=max_frames,
            transcode=transcode,
            bbox_xyxy=bbox_xyxy,
        )

        result = self.infer_from_feature(
            feature=pre["feature"],
            camera_RT=pre["camera_RT"],
            camera_K=pre["camera_K"],
            seeds=seeds,
            cfg_scale=cfg_scale,
            camera_is_static=pre["camera_is_static"],
            overlap_frames=overlap_frames,
            floor_method=floor_method,
            floor_axis=floor_axis,
            ground_align=ground_align,
        )
        result["bbox_xyxy"] = pre["bbox_xyxy"].detach().cpu()
        result["camera_K_full"] = pre["camera_K_full"].detach().cpu()
        result["video_path"] = pre["video_path"]
        return result

    def __call__(self, *args, **kwargs):
        return self.infer_from_feature(*args, **kwargs)
