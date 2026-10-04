"""Official 491-D FlowHMR sampling with explicit video/camera contracts."""

from pathlib import Path

import numpy as np
import torch

from motius.models.flowhmr import FlowHMRBundle
from motius.models.flowhmr.bundle import SOURCE_REVISION
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


@PIPELINES.register_module()
class FlowHMRPipeline(BasePipeline):
    BUNDLE_CLS = FlowHMRBundle

    def __init__(self, bundle, *, preprocessor_kwargs=None, vggt_checkpoint=None, **kwargs):
        super().__init__(bundle, **kwargs)
        self.preprocessor_kwargs = dict(preprocessor_kwargs or {})
        artifact_root = getattr(bundle, "artifact_root", None)
        if artifact_root:
            defaults = {"sam3d_ckpt": "preprocessor/sam/model.ckpt",
                        "sam3d_config": "preprocessor/sam/model_config.yaml",
                        "sam3d_mhr": "preprocessor/sam/assets/mhr_model.pt",
                        "yolox_ckpt": "preprocessor/yolox_l.pth"}
            for name, relative in defaults.items():
                if (artifact_root / relative).is_file():
                    self.preprocessor_kwargs.setdefault(name, str(artifact_root / relative))
            candidate = artifact_root / "preprocessor/vggt_omega_1b_512.pt"
            if vggt_checkpoint is None and candidate.is_file():
                vggt_checkpoint = str(candidate)
        self.vggt_checkpoint = vggt_checkpoint

    @torch.no_grad()
    def infer_from_feature(self, feature, camera_RT, *, seeds=(0,), cfg_scale=1.0,
                           ground_align=True, postprocess=False):
        """SAM tokens (T,3072), OpenCV camera-from-world RT (T,4,4), 30 fps.

        The OpenCV → Y-up change of basis matches the official video path.
        Output is native SMPL-H, with a separate first-frame floor-aligned view.
        Camera translation is a conditioning asset, not a calibrated transform
        into the generated motion's independently reconstructed world.
        """
        from motius.pipelines.flowhmr.camera import validate_extrinsics
        from motius.models.flowhmr.vendor.flowhmr.utils.runtime_vggt_camera import _vggt_opencv_to_yup_rt
        feature = torch.as_tensor(feature, dtype=torch.float32)
        if feature.ndim != 2 or len(feature) == 0 or not torch.isfinite(feature).all():
            raise ValueError("feature must be a finite nonempty (T,D) tensor")
        rt = validate_extrinsics(camera_RT, len(feature))
        yup = _vggt_opencv_to_yup_rt(rt)
        output = self.bundle.generate_from_feature(feature[None],
                   yup[:, :3, :3].reshape(1, len(feature), 9), seeds=seeds, cfg_scale=cfg_scale)
        raw = {k: v.clone() for k, v in output.items() if torch.is_tensor(v)}
        if postprocess:
            from motius.models.flowhmr.vendor.flowhmr.utils.runtime_v2m_postprocess_simple import PostprocessPipeline
            if len(seeds) != 1:
                raise ValueError("Official postprocessing requires exactly one seed")
            processor = PostprocessPipeline.default(body_model=self.bundle.inference_model.body_model,
                                                    smpl_mesh=self.bundle.inference_model.mesh_model)
            with torch.enable_grad():
                output = processor.run(output)
        elif ground_align:
            first = self.bundle.inference_model.mesh_model({
                "rot6d": output["rot6d"][:, 0],
                "shapes": output["shapes"][:, 0], "trans": output["trans"][:, 0],
            })
            floor = first["vertices"][..., 1].min(dim=-1).values
            output["trans"] = output["trans"].clone()
            output["trans"][..., 1] -= floor[:, None]
        b, t = output["trans"].shape[:2]
        params = {"rot6d": output["rot6d"].reshape(b*t, 52, 6),
                  "shapes": output["shapes"].expand(b, t, -1).reshape(b*t, -1),
                  "trans": output["trans"].reshape(b*t, 3)}
        joints = self.bundle.inference_model.body_model(params)["keypoints3d"].reshape(b, t, 52, 3)
        return {**{k: v.detach().cpu() for k, v in output.items() if torch.is_tensor(v)},
                "raw_trans": raw["trans"].cpu(), "joints_world": joints.cpu(),
                "camera_RT": rt.cpu(), "camera_R_condition": yup[:, :3, :3].cpu()}

    @torch.no_grad()
    def infer_v2m(self, video_path, *, work_dir="outputs/inference/flowhmr",
                  camera_RT=None, camera_K=None, bbox_xyxy=None,
                  max_frames=None, transcode=True, preprocessor=None,
                  vggt_frame_interval=1, **sampling_kwargs):
        from motius.models.flowhmr.preprocess import FlowHMRVideoPreprocessor
        from motius.models.flowhmr.vendor.flowhmr.utils.runtime_vggt_camera import extract_vggt_camera
        device = next(self.bundle.parameters()).device
        pre = preprocessor or FlowHMRVideoPreprocessor(device=str(device), **self.preprocessor_kwargs)
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        used = pre.transcode_to_30fps(str(video_path), str(root / "video"), max_frames) if transcode else str(video_path)
        if not transcode:
            import cv2
            cap = cv2.VideoCapture(used)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if abs(fps - 30) > 0.05:
                raise ValueError("FlowHMR expects 30 fps; enable transcode for other frame rates")
        if bbox_xyxy is None:
            bbox_xyxy, _ = pre.detect_and_track(used)
        else:
            bbox_xyxy, _ = pre.prepare_bboxes(used, bbox_xyxy, smooth_window=1)
        if max_frames is not None:
            bbox_xyxy = bbox_xyxy[:max_frames]
        frames = len(bbox_xyxy)
        if (camera_RT is None) != (camera_K is None):
            raise ValueError("Supply camera_RT and camera_K together")
        if camera_RT is None:
            args = {} if self.vggt_checkpoint is None else {"checkpoint_path": self.vggt_checkpoint}
            # No path-only cache: replacing a video under the same filename must
            # not silently reuse stale camera predictions.
            camera = extract_vggt_camera(used, frame_range=(0, frames-1), expected_len=frames,
                                        device=device, frame_interval=vggt_frame_interval, **args)
            camera_RT, camera_K = camera.camera_RT, camera.K
        camera_K = torch.as_tensor(camera_K, dtype=torch.float32)
        if camera_K.shape != (frames, 3, 3) or not torch.isfinite(camera_K).all():
            raise ValueError("camera_K must contain one finite (3,3) intrinsic per output frame")
        feature = pre.extract_features(used, bbox_xyxy, camera_K, token_dim=3072, max_frames=frames)
        result = self.infer_from_feature(feature, camera_RT, **sampling_kwargs)
        result.update(camera_K=camera_K.cpu(), bbox_xyxy=bbox_xyxy.cpu(), video_path=used)
        return result

    def infer_monocular_motion_capture(self, video_path, **kwargs):
        from motius.motion.representation.monocular_capture import (
            MonocularCaptureResult, MonocularTrack, GRAVITY_WORLD_Y_UP)
        from motius.pipelines.prompthmr import SMPL_SMPLX_BODY22_NAMES
        import cv2
        if not self.bundle.checkpoint_sha256:
            raise ValueError("Load a checkpoint before exporting a provenance-tracked capture")
        cap = cv2.VideoCapture(str(video_path))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        result = self.infer_v2m(video_path, **kwargs)
        if result["joints_world"].shape[0] != 1:
            raise ValueError("Monocular capture export requires exactly one seed")
        joints = result["joints_world"][0].numpy()
        t = len(joints)
        track = MonocularTrack(track_id="person_0", frame_ids=np.arange(t),
            valid=np.ones(t, dtype=bool), body_model="SMPL-H neutral",
            joint_names=SMPL_SMPLX_BODY22_NAMES, joints_world=joints[:, :22],
            shape_parameters=result["shapes"][0].numpy(), root_translation_world=joints[:, 0],
            native_parameters={"rot6d": result["rot6d"][0].numpy(),
                               "trans": result["trans"][0].numpy(),
                               "bbox_xyxy": result["bbox_xyxy"].numpy()},
            availability={"world": "predicted", "camera": "unavailable: no metric alignment"})
        return MonocularCaptureResult(source_model="FlowHMR", source_revision=SOURCE_REVISION,
            checkpoint_sha256=self.bundle.checkpoint_sha256, original_fps=original_fps,
            output_fps=30, tracks=(track,), world_coordinate_system=GRAVITY_WORLD_Y_UP,
            camera_intrinsics=result["camera_K"].numpy(), frame_timestamps=np.arange(t)/30,
            metadata={"camera_translation": "conditioning only; no metric world alignment"})
