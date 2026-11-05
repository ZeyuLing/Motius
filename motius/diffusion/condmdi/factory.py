"""condmdi noise schedule construction."""
from argparse import Namespace
from motius.configuration.condmdi import normalize_config
from . import (DiffusionConfig, LossType, ModelMeanType, ModelVarType,
               SpacedDiffusion, get_named_beta_schedule, space_timesteps)

def build_diffusion(config=None, respacing: str = ""):
    cfg = Namespace(**normalize_config(config))
    betas = get_named_beta_schedule(cfg.noise_schedule, cfg.diffusion_steps, 1.0)
    use_timesteps = space_timesteps(
        cfg.diffusion_steps,
        respacing or [cfg.diffusion_steps],
    )
    return SpacedDiffusion(
        use_timesteps=use_timesteps,
        conf=DiffusionConfig(
            betas=betas,
            model_mean_type=(ModelMeanType.START_X if cfg.predict_xstart else ModelMeanType.EPSILON),
            model_var_type=(ModelVarType.FIXED_SMALL if cfg.sigma_small else ModelVarType.FIXED_LARGE),
            loss_type=LossType.MSE,
            rescale_timesteps=False,
            lambda_vel=cfg.lambda_vel,
            lambda_rcxyz=cfg.lambda_rcxyz,
            lambda_fc=cfg.lambda_fc,
            clip_range=cfg.clip_range,
            train_trajectory_only_xz=cfg.xz_only,
            use_random_proj=cfg.use_random_proj,
            fp16=cfg.use_fp16,
            traj_only=cfg.traj_only,
            abs_3d=cfg.abs_3d,
            apply_zero_mask=cfg.apply_zero_mask,
            traj_extra_weight=cfg.traj_extra_weight,
            time_weighted_loss=cfg.time_weighted_loss,
            train_x0_as_eps=cfg.train_x0_as_eps,
        ),
    )
