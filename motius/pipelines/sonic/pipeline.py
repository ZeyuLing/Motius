"""Task API for NVIDIA SONIC G1 motion tracking."""

from __future__ import annotations

from typing import Any

from motius.pipelines.base_pipeline import BasePipeline
from motius.registry import PIPELINES
from motius.simulators.pipeline import MotionTrackingPipelineMixin


@PIPELINES.register_module()
class SONICPipeline(MotionTrackingPipelineMixin, BasePipeline):
    BUNDLE_CLS = "motius.models.sonic.SONICBundle"

    def __init__(self, bundle):
        super().__init__(bundle)
        if hasattr(bundle, 'encode_tokens'):
            widths = [
                bundle.policy.encoders[name].input_dimension
                for name in ['g1', 'teleop', 'smpl']
            ]
            if (
                widths != [640, 267, 840]
                or bundle.policy.decoders['g1_dyn'].input_dimension != 994
            ):
                raise ValueError(
                    'This SONIC deployment Pipeline requires the released default 1762/994 observation schema'
                )

    @classmethod
    def from_original_checkpoint(cls, checkpoint, *, configuration=None, device='cpu'):
        from motius.models.sonic import SONICPolicyBundle

        return cls(
            SONICPolicyBundle.from_original_checkpoint(
                checkpoint, configuration=configuration, device=device
            )
        )

    def _native_step(self, encoder_observation, decoder_observation):
        import torch
        from motius.motion.representation.sonic import released_deployment_features

        encoder = torch.as_tensor(
            encoder_observation, dtype=torch.float32, device=self.bundle.device
        )
        state = torch.as_tensor(
            decoder_observation, dtype=torch.float32, device=self.bundle.device
        )
        if encoder.ndim == 1:
            encoder = encoder[None]
        if state.ndim == 1:
            state = state[None]
        if state.shape != (encoder.shape[0], 930):
            raise ValueError(
                'SONIC decoder observations must have shape [B, 930], excluding tokens'
            )
        features, masks = released_deployment_features(encoder)
        with torch.no_grad():
            tokens = self.bundle.encode_tokens(features, masks, encoder.shape[0])
            action = self.bundle.decode_actions(tokens, state[:, None]).squeeze(1)
            flat = tokens.flatten(start_dim=1)
            decoder_input = torch.cat([flat, state], -1)
        return dict(
            action=action.cpu().numpy(),
            encoded_tokens=flat.cpu().numpy(),
            decoder_input=decoder_input.cpu().numpy(),
        )

    def infer_motion_tracking(
        self,
        encoder_observation: Any,
        decoder_observation: Any,
    ) -> dict[str, Any]:
        """Run one 50 Hz SONIC policy step from deployment observations."""

        if hasattr(self.bundle, 'encode_tokens'):
            output = self._native_step(encoder_observation, decoder_observation)
        else:
            import numpy as np
            from motius.models.onnx_policy import as_numpy

            tokens = self.bundle.encode(encoder_observation)
            state = as_numpy(decoder_observation, np.dtype(np.float32))
            if state.ndim == 1:
                state = state[None]
            if state.shape != (1, self.bundle.decoder_state_dim):
                raise ValueError(
                    'SONIC decoder observations must exclude tokens and have '
                    f'shape [1, {self.bundle.decoder_state_dim}], got {state.shape}'
                )
            decoder_input = np.concatenate([tokens, state], axis=-1)
            output = dict(
                action=self.bundle.decode(decoder_input),
                encoded_tokens=tokens,
                decoder_input=decoder_input,
            )
        output.update(
            {
                "control_hz": 50,
                "robot": "Unitree G1 29-DOF",
                "method": "SONIC",
            }
        )
        return output

    __call__ = infer_motion_tracking

    def save_pretrained(self, save_directory, **kwargs):
        return self.bundle.save_pretrained(save_directory, **kwargs)


__all__ = ["SONICPipeline"]
