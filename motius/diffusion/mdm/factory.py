"""MDM diffusion schedule construction."""
from . import gaussian_diffusion as gd
from .respace import SpacedDiffusion, space_timesteps

def create_gaussian_diffusion(args):
    # MDM always predicts x_start with a fixed (small) posterior variance and no
    # learned sigma; the full schedule keeps all `diffusion_steps` (no respacing).
    predict_xstart = True
    steps = args.diffusion_steps
    scale_beta = 1.0
    timestep_respacing = ""
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    lambda_target_loc = getattr(args, "lambda_target_loc", 0.0)

    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (gd.ModelVarType.FIXED_LARGE if not args.sigma_small else gd.ModelVarType.FIXED_SMALL)
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_target_loc=lambda_target_loc,
    )
