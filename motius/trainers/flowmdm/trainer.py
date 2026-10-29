"""FlowMDM's published masked x-start objective and noise draws."""
import torch
from motius.motion.conditioning.flowmdm_text import prepare_text_conditions
from motius.registry import TRAINERS
from motius.trainers.base_trainer import BaseTrainer
from motius.configuration.flowmdm import create_gaussian_diffusion


def masked_l2(target, prediction, mask):
    error = ((target - prediction) ** 2) * mask.float()
    dimensions = list(range(1, error.ndim))
    return error.sum(dim=dimensions) / (mask.sum(dim=list(range(1, mask.ndim))) * target.shape[1] * target.shape[2])


@TRAINERS.register_module()
class FlowMDMTrainer(BaseTrainer):
    def __init__(self, bundle, **kwargs):
        super().__init__(bundle, **kwargs)
        args = bundle.flow_args
        if any(getattr(args, key, 0.) for key in ("lambda_fc", "lambda_rcxyz", "lambda_vel_rcxyz")):
            raise ValueError("Released FlowMDM training uses zero body-geometry weights; other protocols require a body codec")
        self.diffusion = create_gaussian_diffusion(args)
        bundle._trainable_modules = ["model"]
        bundle._save_ckpt_modules = ["model"]
        bundle.network.clip_model.requires_grad_(False)

    def train_step(self, batch):
        motion = batch["motion"].to(self.bundle.device)
        if motion.ndim == 3:
            motion = motion.permute(0, 2, 1).unsqueeze(2)
        lengths = batch["motion_lens"].to(self.bundle.device)
        mask = torch.arange(motion.shape[-1], device=motion.device)[None, :] < lengths[:, None]
        timestep = batch.get("timesteps")
        if timestep is None:
            timestep = torch.randint(self.diffusion.num_timesteps, (len(motion),), device=motion.device)
        timestep = timestep.to(motion.device)
        noise = batch.get("noise")
        noise = torch.randn_like(motion) if noise is None else noise.to(motion)
        noisy = self.diffusion.q_sample(motion, timestep, noise=noise)
        conditions = dict(batch.get("conditions", {}))
        conditions.update(text=list(batch["text"]), mask=mask, lengths=lengths)
        conditions = prepare_text_conditions(self.bundle.network, conditions)
        prediction = self.bundle.predict_motion(noisy, timestep, conditions)
        element_mask = mask[:, None, None, :]
        mse = masked_l2(motion, prediction, element_mask)
        velocity = 0.
        if self.bundle.flow_args.lambda_vel:
            target_velocity = motion[..., 1:] - motion[..., :-1]
            predicted_velocity = prediction[..., 1:] - prediction[..., :-1]
            velocity = masked_l2(target_velocity[:, :-1], predicted_velocity[:, :-1], element_mask[..., 1:])
        loss = mse + self.bundle.flow_args.lambda_vel * velocity
        return {"loss": loss.mean(), "flowmdm_mse": mse.detach().mean()}

    @torch.no_grad()
    def val_step(self, batch):
        return {"val_"+key:value.detach() for key,value in self.train_step(batch).items()}
