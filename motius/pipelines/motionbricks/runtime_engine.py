"""Native PyTorch generation with Warp pose services on a shared CUDA stream.

No TensorRT plan, ONNX graph, or upstream package participates in this engine.
The pipeline owns the neural graph; this service owns pose buffers and timing.
"""

from __future__ import annotations

from contextlib import contextmanager
from threading import RLock

import torch
import warp as wp

from motius.configuration.motionbricks_runtime import MotionBricksInferenceConfig
from motius.configuration.motionbricks_runtime_contract import valid_frame_count
from motius.motion.conditioning.motionbricks_release import release_input_shapes

from .runtime_generation import GenerationTimer, read_token_count

_PRECISION_LOCK = RLock()


@contextmanager
def _fp32_inference():
    # Match the published FP32 graph and restore the caller's backend settings.
    with _PRECISION_LOCK:
        previous = torch.backends.cuda.matmul.allow_tf32
        previous_cudnn = torch.backends.cudnn.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            with torch.autocast("cuda", enabled=False):
                yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous
            torch.backends.cudnn.allow_tf32 = previous_cudnn


class MotionBricksEngine:
    def __init__(self, pipeline, *, model_to_runtime=None, stream=None):
        if pipeline.bundle.device.type != "cuda":
            raise ValueError("The full MotionBricks runtime requires a CUDA bundle")
        self.pipeline = pipeline
        self._torch_stream = torch.cuda.current_stream(pipeline.bundle.device)
        self.stream = stream or wp.stream_from_torch(self._torch_stream)
        self.device = self.stream.device
        if stream is not None:
            self._torch_stream = wp.stream_to_torch(stream)
        if wp.device_to_torch(self.device) != str(pipeline.bundle.device):
            raise ValueError("Warp stream and model must use the same CUDA device")
        self._closed = False
        self._model_to_runtime = (
            wp.transform_identity()
            if model_to_runtime is None
            else wp.transform(model_to_runtime)
        )
        self._runtime_to_model = wp.transform_inverse(self._model_to_runtime)
        self.timer = GenerationTimer(self.stream)
        self._alignment = wp.empty(1, dtype=wp.transform, device=self.device)
        self._generated_frame_count = wp.empty(1, dtype=wp.int32, device=self.device)
        shapes = release_input_shapes(pipeline.bundle.model)
        self._input_names = tuple(shapes)
        if pipeline.bundle.model == "g1":
            shapes.update(
                pred_qpos=(64, 36), pred_foot_contacts=(64, 4), pred_num_tokens=()
            )
        else:
            shapes.update(pred_global_motions=(64, 366), pred_num_tokens=())
        self._tensors = {
            name: torch.empty(
                (1, *shape),
                device=pipeline.bundle.device,
                dtype=torch.int32
                if name.startswith(("has_", "allowed_")) or name == "pred_num_tokens"
                else torch.float32,
            )
            for name, shape in shapes.items()
        }
        self._warp_tensors = {
            name: wp.from_torch(value) for name, value in self._tensors.items()
        }
        with wp.ScopedStream(self.stream):
            self._initialize_pose_layout()
        self.synchronize()

    def _ensure_open(self):
        if self._closed:
            raise RuntimeError("MotionBricks runtime engine is closed")

    def device_tensor(self, name):
        self._ensure_open()
        return self._warp_tensors[name]

    def synchronize(self):
        wp.synchronize_stream(self.stream)

    def generate_animation(self, context, target, output, *, config=None):
        self._ensure_open()
        with wp.ScopedDevice(self.device), wp.ScopedStream(self.stream):
            return self._generate_animation(context, target, output, config=config)

    def _generate_animation(self, context, target, output, *, config=None):
        config = config or MotionBricksInferenceConfig()
        self._prepare_generation(context, target, output)
        output.num_frames = 0
        with self.timer.stage("encode"):
            self._encode_guide_poses(context, target, config)
        with (
            self.timer.stage("inference"),
            torch.cuda.stream(self._torch_stream),
            torch.inference_mode(),
            _fp32_inference(),
        ):
            prediction = self.pipeline.infer_raw(
                {name: self._tensors[name] for name in self._input_names}
            )
            for name, value in prediction.items():
                self._tensors[name].copy_(value)
        self.synchronize()
        count = read_token_count(self, config)
        with self.timer.stage("decode"):
            self._decode_generated_poses(output)
        if config.correct_generated_diff:
            with self.timer.stage("post_process"):
                self._post_process_generated_poses(context, target, output)
        self.synchronize()
        self.timer.read()
        output.num_frames = valid_frame_count(count)
        return output

    def close(self):
        if not self._closed:
            self.synchronize()
            self._closed = True
            self._warp_tensors.clear()
            self._tensors.clear()

    def __enter__(self):
        self._ensure_open()
        return self

    def __exit__(self, *_):
        self.close()


# Annotation compatibility for buffer services; no external engine dependency.
InferenceEngine = MotionBricksEngine
