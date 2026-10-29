"""omnicontrol noise schedule construction."""
from motius.configuration.omnicontrol import normalize_config
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

def build_diffusion(config, *, mean, std, raw_mean, raw_std, respacing=""):
    cfg = normalize_config(config)
    steps = int(cfg["diffusion_steps"])
    betas = gd.get_named_beta_schedule(cfg["noise_schedule"], steps, 1.0)
    use_timesteps = space_timesteps(steps, respacing or [steps])
    return SpacedDiffusion(
        use_timesteps=use_timesteps,
        betas=betas,
        model_mean_type=gd.ModelMeanType.START_X,
        model_var_type=(
            gd.ModelVarType.FIXED_SMALL
            if cfg["sigma_small"] else gd.ModelVarType.FIXED_LARGE
        ),
        loss_type=gd.LossType.MSE,
        rescale_timesteps=False,
        lambda_rcxyz=cfg["lambda_rcxyz"],
        lambda_vel=cfg["lambda_vel"],
        lambda_fc=cfg["lambda_fc"],
        dataset="humanml",
        mean=mean,
        std=std,
        raw_mean=raw_mean,
        raw_std=raw_std,
    )
