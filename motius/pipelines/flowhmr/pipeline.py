"""Official 491-D FlowHMR sampling with explicit video/camera contracts."""

from pathlib import Path
from functools import wraps

import numpy as np
import torch

from motius.models.flowhmr import FlowHMRBundle
from motius.models.flowhmr.bundle import SOURCE_REVISION
from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES


def _strict_frontend_float32(function):
    """FP32 comparison disables TF32 and restores process settings on exit."""
    @wraps(function)
    def wrapped(self, *args, **kwargs):
        if self.frontend_precision != 'float32':
            return function(self, *args, **kwargs)
        matmul = torch.backends.cuda.matmul.allow_tf32
        cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            return function(self, *args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = matmul
            torch.backends.cudnn.allow_tf32 = cudnn
    return wrapped


@PIPELINES.register_module()
class FlowHMRPipeline(BasePipeline):
    BUNDLE_CLS = FlowHMRBundle

    def __init__(self, bundle, *, preprocessor_kwargs=None, vggt_checkpoint=None,
                 body_model_path=None, lightweight=False, attention_query_chunk_size=256, profile=False,
                 sam_backend="auto", vggt_memory_mode="auto", sam_frame_batch_size=None, vggt_attention_backend="auto",
                 vggt_frame_batch_size=8, vggt_query_chunk_size=4096, vggt_token_chunk_size=4096,
                 frontend_precision="source",
                 **kwargs):
        super().__init__(bundle, **kwargs)
        if vggt_memory_mode not in ("auto", "reference", "streamed"):
            raise ValueError("vggt_memory_mode must be auto, reference or streamed")
        self.vggt_memory_mode = ("streamed" if lightweight else "reference") if vggt_memory_mode == "auto" else vggt_memory_mode
        if vggt_attention_backend not in ("auto", "torch", "cudnn"):
            raise ValueError("vggt_attention_backend must be auto, torch or cudnn")
        self.vggt_attention_backend = vggt_attention_backend
        self.vggt_schedule = dict(frame_batch_size=vggt_frame_batch_size, query_chunk_size=vggt_query_chunk_size, token_chunk_size=vggt_token_chunk_size)
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in self.vggt_schedule.values()):
            raise ValueError("VGGT scheduling parameters must be positive integers")
        if frontend_precision not in ("source", "float16", "float32"):
            raise ValueError("frontend_precision must be source, float16 or float32")
        self.frontend_precision = frontend_precision
        self.profile = bool(profile)
        self.stage_timings = []
        self.lightweight = bool(lightweight)
        if self.lightweight:
            if not isinstance(attention_query_chunk_size, int) or isinstance(attention_query_chunk_size, bool) or attention_query_chunk_size < 1:
                raise ValueError('attention_query_chunk_size must be a positive integer')
            bundle.network.set_attention_query_chunk_size(attention_query_chunk_size)
        if sam_backend not in ('auto', 'official', 'fast'):
            raise ValueError('sam_backend must be auto, official or fast')
        self.preprocessor_kwargs = dict(preprocessor_kwargs or {})
        if sam_backend != 'auto':
            self.preprocessor_kwargs.update(sam_fast=sam_backend == 'fast',
                                            sam_token_only=self.lightweight or sam_backend == 'fast')
        else:
            # Honor the pre-0.3.7 explicit request for full SAM execution.
            use_fast = self.lightweight and self.preprocessor_kwargs.get('sam_token_only', True)
            self.preprocessor_kwargs.setdefault('sam_fast', use_fast)
            if self.lightweight or self.preprocessor_kwargs['sam_fast']:
                self.preprocessor_kwargs.setdefault('sam_token_only', True)
        if frontend_precision != 'source':
            self.preprocessor_kwargs['sam_precision'] = frontend_precision
        if sam_frame_batch_size is not None and (isinstance(sam_frame_batch_size, bool) or not isinstance(sam_frame_batch_size, int) or sam_frame_batch_size < 1):
            raise ValueError('sam_frame_batch_size must be a positive integer')
        if sam_frame_batch_size is not None:
            self.preprocessor_kwargs['sam_frame_batch_size'] = sam_frame_batch_size
        self.preprocessor_kwargs.setdefault('sam_frame_batch_size', 8 if self.lightweight and self.preprocessor_kwargs.get('sam_token_only') else 1)
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
        from .sampling import FlowHMRSampler
        from motius.motion.representation.flowhmr import FlowHMRMotionCodec
        if self.lightweight and not (body_model_path or bundle.body_model_path):
            from .lightweight import resolve_body_model
            body_model_path = resolve_body_model()
        self.sampler = FlowHMRSampler(bundle, FlowHMRMotionCodec.from_bundle(bundle, body_model_path))
        self.codec = self.sampler.codec

    @torch.no_grad()
    def generate_from_feature(self, feature, camera_R, *, seeds=(0,), cfg_scale=1.0):
        if feature.ndim != 3 or feature.shape[0] != 1 or feature.shape[1] == 0:
            raise ValueError("Expected one nonempty feature sequence shaped (1,T,D)")
        expected = self.bundle.config["network_module_args"]["ctxt_input_dim"]["feature"]
        if feature.shape[-1] != expected or camera_R.shape != (*feature.shape[:2], 9):
            raise ValueError("Feature or camera_R dimensions do not match the model")
        if not seeds or cfg_scale < 1:
            raise ValueError("Provide at least one seed and cfg_scale >= 1")
        device = next(self.bundle.parameters()).device
        self.codec.synchronize_statistics(self.bundle)
        conditioning = {"feature": feature.to(device, torch.float32),
                        "camera_R": camera_R.to(device, torch.float32)}
        return self.sampler.generate(conditioning, seeds=list(seeds),
                                     length=feature.shape[1], cfg_scale=cfg_scale, progress=getattr(self, 'progress_callback', None))

    @torch.no_grad()
    def infer_from_feature(self, feature, camera_RT, *, seeds=(0,), cfg_scale=1.0,
                           ground_align=True, postprocess=False):
        """SAM tokens (T,3072), OpenCV camera-from-world RT (T,4,4), 30 fps.

        Camera rotations are relative to the first frame, matching training.
        Output is native SMPL-H, with a separate first-frame floor-aligned view.
        Camera translation is a conditioning asset, not a calibrated transform
        into the generated motion's independently reconstructed world.
        """
        from motius.pipelines.flowhmr.camera import validate_extrinsics
        feature = torch.as_tensor(feature, dtype=torch.float32)
        if feature.ndim != 2 or len(feature) == 0 or not torch.isfinite(feature).all():
            raise ValueError("feature must be a finite nonempty (T,D) tensor")
        rt = validate_extrinsics(camera_RT, len(feature))
        # Match training's compute_camera_features and official build_feature_dict.
        # A common world-axis change (including OpenCV -> Y-up) cancels here.
        camera_r = rt[:, :3, :3]
        camera_r = camera_r @ camera_r[0].transpose(-1, -2)
        output = self.generate_from_feature(feature[None],
                   camera_r.reshape(1, len(feature), 9), seeds=seeds, cfg_scale=cfg_scale)
        raw = {k: v.clone() for k, v in output.items() if torch.is_tensor(v)}
        if postprocess:
            from motius.pipelines.flowhmr.postprocessing import PostprocessPipeline
            if len(seeds) != 1:
                raise ValueError("Official postprocessing requires exactly one seed")
            processor = PostprocessPipeline.default(body_model=self.codec.body_model,
                                                    smpl_mesh=self.codec.mesh_model)
            with torch.enable_grad():
                output = processor.run(output)
        elif ground_align:
            first = self.codec.mesh_model({
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
        joints = self.codec.body_model(params)["keypoints3d"].reshape(b, t, 52, 3)
        return {**{k: v.detach().cpu() for k, v in output.items() if torch.is_tensor(v)},
                "raw_trans": raw["trans"].cpu(), "joints_world": joints.cpu(),
                "camera_RT": rt.cpu(), "camera_R_condition": camera_r.cpu()}

    @torch.no_grad()
    @_strict_frontend_float32
    def infer_v2m(self, video_path, *, work_dir="outputs/inference/flowhmr",
                  camera_RT=None, camera_K=None, bbox_xyxy=None,
                  max_frames=None, transcode=True, preprocessor=None,
                  vggt_frame_interval=1, **sampling_kwargs):
        from motius.pipelines.flowhmr.preprocess import FlowHMRVideoPreprocessor
        from motius.pipelines.flowhmr.camera_estimation import extract_vggt_camera
        from .profiling import measure_stage
        self.stage_timings.clear()
        progress = getattr(self, 'progress_callback', lambda *args: None)
        device = next(self.bundle.parameters()).device
        if device.type == 'cuda' and self.frontend_precision == 'source' and torch.cuda.get_device_capability(device)[0] < 8:
            raise RuntimeError('Upstream BF16 precision is unavailable on this GPU; explicitly select float32 frontend arithmetic instead of silently using FP16')
        camera_backend, camera_selection = "provided", None
        pre = preprocessor or FlowHMRVideoPreprocessor(device=str(device), **self.preprocessor_kwargs)
        pre.progress_callback = progress
        root = Path(work_dir)
        root.mkdir(parents=True, exist_ok=True)
        from .lightweight import frontend_resources, release_preprocessor
        with frontend_resources(self, pre, device):
            with measure_stage(self, "transcode", device):
                used = pre.transcode_to_30fps(str(video_path), str(root / "video"), max_frames) if transcode else str(video_path)
            if not transcode:
                import cv2
                cap = cv2.VideoCapture(used)
                fps = cap.get(cv2.CAP_PROP_FPS)
                cap.release()
                if abs(fps - 30) > 0.05:
                    raise ValueError("FlowHMR expects 30 fps; enable transcode for other frame rates")
            progress('Track selected person')
            with measure_stage(self, "tracking", device):
                if bbox_xyxy is None:
                    bbox_xyxy, _ = pre.detect_and_track(used)
                else:
                    bbox_xyxy, _ = pre.prepare_bboxes(used, bbox_xyxy, smooth_window=1)
            if self.lightweight:
                release_preprocessor(pre, 'release_detector')
            if max_frames is not None:
                bbox_xyxy = bbox_xyxy[:max_frames]
            frames = len(bbox_xyxy)
            if (camera_RT is None) != (camera_K is None):
                raise ValueError("Supply camera_RT and camera_K together")
            progress('Estimate camera')
            with measure_stage(self, "camera", device):
                if camera_RT is None:
                    args = {} if self.vggt_checkpoint is None else {"checkpoint_path": self.vggt_checkpoint}
                    # No path-only cache: replacing a video under the same filename must
                    # not silently reuse stale camera predictions.
                    if self.lightweight:
                        args['cache_model'] = False
                    args["memory_mode"] = self.vggt_memory_mode
                    args["progress"] = progress
                    args["attention_backend"] = self.vggt_attention_backend
                    args.update(self.vggt_schedule)
                    if self.frontend_precision != "source":
                        args["precision"] = self.frontend_precision
                    camera = extract_vggt_camera(used, frame_range=(0, frames-1), expected_len=frames,
                                                device=device, frame_interval=vggt_frame_interval, **args)
                    camera_RT, camera_K = camera.camera_RT, camera.K
                    camera_backend = getattr(camera, "attention_backend", "unknown")
                    camera_selection = getattr(camera, "attention_selection", None)
            camera_K = torch.as_tensor(camera_K, dtype=torch.float32)
            if camera_K.shape != (frames, 3, 3) or not torch.isfinite(camera_K).all():
                raise ValueError("camera_K must contain one finite (3,3) intrinsic per output frame")
            progress('Extract body and hand features')
            with measure_stage(self, "features", device):
                feature = pre.extract_features(used, bbox_xyxy, camera_K, token_dim=3072, max_frames=frames)
        progress('Sample motion')
        with measure_stage(self, "sampling_and_decode", device):
            result = self.infer_from_feature(feature, camera_RT, **sampling_kwargs)
        result.update(camera_K=camera_K.cpu(), bbox_xyxy=bbox_xyxy.cpu(), video_path=used,
                      inference_settings={
                          'lightweight': self.lightweight and bool(getattr(pre, 'sam_fast', self.preprocessor_kwargs.get('sam_fast', False))),
                          'staged_resources': self.lightweight,
                          'attention_query_chunk_size': self.bundle.network.attention_query_chunk_size,
                          'full_hand_features': True,
                          'sam_backend': 'fast' if getattr(pre, 'sam_fast', self.preprocessor_kwargs.get('sam_fast', False)) else 'official',
                          'sam_token_only': bool(getattr(pre, 'sam_token_only', self.preprocessor_kwargs.get('sam_token_only', False))),
                          'frontend_precision': self.frontend_precision,
                          'sam_checkpoint_backbone_dtypes': getattr(pre, 'sam_checkpoint_dtypes', []),
                          'sam_conditioning_optimized': bool(getattr(pre, 'sam_token_only', self.preprocessor_kwargs.get('sam_token_only', False)) and
                              getattr(pre, 'sam_optimize_conditioning', self.preprocessor_kwargs.get('sam_optimize_conditioning', True))),
                          'camera_frame_interval': vggt_frame_interval,
                          'vggt_memory_mode': self.vggt_memory_mode,
                          'vggt_schedule': self.vggt_schedule,
                          'vggt_attention_backend': camera_backend,
                          'vggt_attention_selection': camera_selection,
                          'sam_frame_batch_size': getattr(pre, 'sam_frame_batch_size', 1),
                          'sam_hand_frames_per_batch': min(getattr(pre, 'sam_frame_batch_size', 1), 4) if getattr(pre, 'sam_fast', self.preprocessor_kwargs.get('sam_fast', False)) else None,
                          'camera_subsampled': vggt_frame_interval != 1,
                      })
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
            metadata={"camera_translation": "conditioning only; no metric world alignment",
                      "inference_settings": result["inference_settings"],
                      "stage_timings": list(self.stage_timings)})
