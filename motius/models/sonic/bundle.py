"""Motius bundle for NVIDIA SONIC's released G1 ONNX controller."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import numpy as np

from motius.models.onnx_policy import OnnxTrackingBundle, as_numpy
from motius.registry import MODEL_BUNDLES


@MODEL_BUNDLES.register_module()
class SONICBundle(OnnxTrackingBundle):
    CONFIG_NAME = "sonic_config.json"
    MODEL_TYPE = "sonic_g1"
    METHOD_NAME = "SONIC"
    PIPELINE_CLASS = "motius.pipelines.sonic.SONICPipeline"
    BUNDLE_CLASS = "motius.models.sonic.SONICBundle"
    DEFAULT_FILES = {
        "encoder": "model_encoder.onnx",
        "decoder": "model_decoder.onnx",
        "observation_config": "observation_config.yaml",
    }
    ONNX_ROLES = ("encoder", "decoder")
    ENCODER_DIM = 1762
    TOKEN_DIM = 64
    DECODER_DIM = 994
    DECODER_STATE_DIM = DECODER_DIM - TOKEN_DIM

    def _declared_dimension(
        self,
        role: str,
        tensor_name: str,
        *,
        output: bool = False,
        fallback: int,
    ) -> int:
        session = self.session(role)
        specs = getattr(session, "outputs" if output else "inputs", {})
        spec = specs.get(tensor_name)
        shape = tuple(getattr(spec, "shape", ()))
        if len(shape) == 2 and isinstance(shape[-1], int) and shape[-1] > 0:
            return shape[-1]
        return fallback

    @property
    def encoder_dim(self) -> int:
        return self._declared_dimension(
            "encoder", "obs_dict", fallback=self.ENCODER_DIM
        )

    @property
    def token_dim(self) -> int:
        return self._declared_dimension(
            "encoder",
            "encoded_tokens",
            output=True,
            fallback=self.TOKEN_DIM,
        )

    @property
    def decoder_dim(self) -> int:
        return self._declared_dimension(
            "decoder", "obs_dict", fallback=self.DECODER_DIM
        )

    @property
    def decoder_state_dim(self) -> int:
        state_dim = self.decoder_dim - self.token_dim
        if state_dim <= 0:
            raise ValueError(
                "SONIC decoder input must contain encoded tokens followed by "
                f"state features, got decoder_dim={self.decoder_dim} and "
                f"token_dim={self.token_dim}."
            )
        return state_dim

    def artifact_contract(self) -> dict[str, Any]:
        return {
            "robot": "Unitree G1 29-DOF",
            "control_hz": 50,
            "encoder_input": {"name": "obs_dict", "shape": [1, self.encoder_dim]},
            "encoder_output": {
                "name": "encoded_tokens",
                "shape": [1, self.token_dim],
            },
            "decoder_state_dimension": self.decoder_state_dim,
            "decoder_input": {"name": "obs_dict", "shape": [1, self.decoder_dim]},
            "decoder_output": {"name": "action", "shape": [1, 29]},
            "decoder_layout": "encoded_tokens_then_decoder_state",
        }

    def artifact_source(self) -> dict[str, Any]:
        return {
            "repo": "https://github.com/NVlabs/GR00T-WholeBodyControl",
            "checkpoint": "https://huggingface.co/nvidia/GEAR-SONIC",
            "paper": "https://arxiv.org/abs/2511.07820",
        }

    def encode(self, encoder_observation: Any) -> np.ndarray:
        observation = as_numpy(encoder_observation, np.dtype(np.float32))
        if observation.ndim == 1:
            observation = observation[None]
        if observation.shape != (1, self.encoder_dim):
            raise ValueError(
                "SONIC encoder_observation must have shape "
                f"[1, {self.encoder_dim}], got {observation.shape}."
            )
        return self.session("encoder").run({"obs_dict": observation})[
            "encoded_tokens"
        ]

    def forward(
        self,
        encoder_observation: Any,
        decoder_observation: Any,
    ) -> dict[str, np.ndarray]:
        tokens = self.encode(encoder_observation)
        state = as_numpy(decoder_observation, np.dtype(np.float32))
        if state.ndim == 1:
            state = state[None]
        if state.shape != (1, self.decoder_state_dim):
            raise ValueError(
                "SONIC decoder_observation must exclude the encoded token and have "
                f"shape [1, {self.decoder_state_dim}], got {state.shape}."
            )
        if tokens.shape[:-1] != state.shape[:-1]:
            raise ValueError(
                f"SONIC encoder/decoder batch mismatch: {tokens.shape} vs {state.shape}."
            )
        decoder_input = np.concatenate([tokens, state], axis=-1)
        outputs = self.session("decoder").run({"obs_dict": decoder_input})
        return {
            "action": outputs["action"],
            "encoded_tokens": tokens,
            "decoder_input": decoder_input,
        }


__all__ = ["SONICBundle"]
