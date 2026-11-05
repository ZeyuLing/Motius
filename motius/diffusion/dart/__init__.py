from . import gaussian as gd
from .respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(args, enable_ddim=True):
    predict_xstart = True
    steps = args.diffusion_steps
    scale_beta = 1.0
    timestep_respacing = args.respacing if enable_ddim else ''
    learn_sigma = False
    rescale_timesteps = False
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE
    if not timestep_respacing:
        timestep_respacing = [steps]
    return SpacedDiffusion(use_timesteps=space_timesteps(steps, timestep_respacing), betas=betas, model_mean_type=gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X, model_var_type=(gd.ModelVarType.FIXED_LARGE if not args.sigma_small else gd.ModelVarType.FIXED_SMALL) if not learn_sigma else gd.ModelVarType.LEARNED_RANGE, loss_type=loss_type, rescale_timesteps=rescale_timesteps)
