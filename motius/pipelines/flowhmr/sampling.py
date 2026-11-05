"""FlowHMR ODE sampling; seed placement and update order match upstream."""
from typing import Dict, List
import torch
from torch import Tensor
from motius.diffusion.flowhmr import odeint_custom
from motius.motion.representation.flowhmr import FlowHMRMotionCodec
from motius.diffusion.flowhmr_utils import length_to_mask, randn_tensor


class FlowHMRSampler:
    def __init__(self, bundle, codec=None):
        self.bundle = bundle
        self.codec = codec or FlowHMRMotionCodec.from_bundle(bundle)
        args = bundle.config["train_pipeline_args"]
        self.latent_dim = self.codec.LATENT_DIM
        self.validation_steps = args["infer_noise_scheduler_cfg"]["validation_steps"]
        self.pred_type = args.get("pred_type", "velocity")
        self._noise_scheduler_cfg = args["noise_scheduler_cfg"]
        self.fps = self.codec.fps

    @torch.no_grad()
    def generate(
        self,
        feature: Dict[str, Tensor],
        seeds: List[int],
        length: int,
        cfg_scale: float = 1.0,
        progress=None,
    ) -> Dict[str, Tensor]:
        feat_val = next(iter(feature.values()))
        T = feat_val.shape[1]
        device = feat_val.device
        dtype = feat_val.dtype
        repeat = len(seeds)

        do_classifier_free_guidance = cfg_scale > 1

        feature = {k: v.clone() for k, v in feature.items()}
        if cfg_scale == 1:
            for key in feature:
                feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
        else:
            for key in feature:
                feature[key] = feature[key].repeat((repeat,) + (1,) * (feature[key].dim() - 1))
                if key == "feature":
                    feature[key] = torch.cat([torch.zeros_like(feature[key]), feature[key]], dim=0)
                else:
                    feature[key] = torch.cat([feature[key], feature[key]], dim=0)

        vtxt_input = torch.zeros((repeat, 1, self.bundle.network.vtxt_input_dim), device=device)

        length_tensor = torch.tensor([length], device=device)
        x_mask_temporal = length_to_mask(length_tensor, T)
        x_mask_temporal = x_mask_temporal.repeat((repeat,) + (1,) * (x_mask_temporal.dim() - 1))

        if do_classifier_free_guidance:
            x_mask_temporal = torch.cat([x_mask_temporal] * 2, dim=0)
            vtxt_input = torch.cat([vtxt_input] * 2, dim=0)

        noise_shape = (1, T, self.latent_dim)
        y0_list = []
        for seed in seeds:
            generator = torch.Generator().manual_seed(seed)
            noise = randn_tensor(noise_shape, generator=generator, dtype=dtype).to(device)
            y0_list.append(noise)
        y0 = torch.cat(y0_list, dim=0)

        evaluations = 0
        def fn(t: Tensor, x: Tensor) -> Tensor:
            nonlocal evaluations
            x_input = torch.cat([x] * 2, dim=0) if do_classifier_free_guidance else x
            x_pred = self.bundle.predict_motion(
                x=x_input,
                ctxt_input=feature,
                vtxt_input=vtxt_input,
                timesteps=t.expand(x_input.shape[0]),
                x_mask_temporal=x_mask_temporal,
                ctxt_mask_temporal=x_mask_temporal,
            )

            if self.pred_type == "velocity":
                pass
            elif self.pred_type == "x1":
                t_eps = 1 / self.validation_steps
                x_pred = (x_pred - x_input) / (1.0 - t).clamp_min(t_eps)
            elif self.pred_type == "x1raw":
                x_pred = x_pred - y0

            if do_classifier_free_guidance:
                x_pred_basic, x_pred_text = x_pred.chunk(2, dim=0)
                x_pred = x_pred_basic + cfg_scale * (x_pred_text - x_pred_basic)

            evaluations += 1
            if progress is not None:
                progress('Sample motion', evaluations, self.validation_steps if self._noise_scheduler_cfg.get('method', 'euler') == 'euler' else 0)
            return x_pred

        t = torch.linspace(0, 1, self.validation_steps + 1, device=device, dtype=dtype)
        trajectory = odeint_custom(fn, y0, t, **self._noise_scheduler_cfg)
        sampled: Tensor = trajectory[-1]

        output = self.codec.decode_motion(sampled, fps=self.fps)

        for key in output:
            if isinstance(output[key], Tensor) and output[key].dim() >= 2:
                output[key] = output[key][:, :length]

        return output
