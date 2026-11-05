"""Build the released InterGen noise schedule."""
from .gaussian_diffusion import (MotionDiffusion, space_timesteps, get_named_beta_schedule,
                                 ModelMeanType, ModelVarType, LossType)


def create_diffusion(cfg, respacing):
    return MotionDiffusion(
        use_timesteps=space_timesteps(cfg.DIFFUSION_STEPS, respacing),
        betas=get_named_beta_schedule(cfg.BETA_SCHEDULER, cfg.DIFFUSION_STEPS),
        motion_rep=cfg.MOTION_REP, model_mean_type=ModelMeanType.START_X,
        model_var_type=ModelVarType.FIXED_SMALL, loss_type=LossType.MSE,
        rescale_timesteps=False)
