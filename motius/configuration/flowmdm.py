"""FlowMDM architecture and scheduler construction from artifact metadata."""
from motius.diffusion.flowmdm import gaussian_diffusion as gd
from motius.diffusion.flowmdm.respace import SpacedDiffusion, space_timesteps

def get_cond_mode(args):
    return 'no_cond' if args.unconstrained else 'text'

def get_model_args(args, cond_mode=None):

    # default args
    clip_version = 'ViT-B/32'
    cond_mode = get_cond_mode(args) if cond_mode is None else cond_mode

    # SMPL defaults
    data_rep = 'rot6d'
    njoints = 25
    nfeats = 6

    if args.dataset == 'humanml':
        data_rep = 'hml_vec'
        njoints = 263
        nfeats = 1
    elif args.dataset == 'babel':
        data_rep = 'rot6d'
        njoints = 135
        nfeats = 1

    return {'modeltype': '', 'njoints': njoints, 'nfeats': nfeats,
            'translation': True, 'pose_rep': data_rep, 'glob': True, 'glob_rot': True,
            'latent_dim': args.latent_dim, 'ff_size': 1024, 'num_layers': args.layers, 'num_heads': args.num_heads,
            'dropout': 0.1, 'activation': "gelu", 'data_rep': data_rep, 'cond_mode': cond_mode,
            'cond_mask_prob': args.cond_mask_prob,
            'clip_version': clip_version, 'dataset': args.dataset,
            'diffusion_steps': args.diffusion_steps,
            'max_seq_att': args.max_seq_att,
            # FlowMDM
            'bpe_denoising_step': args.bpe_denoising_step,
            'bpe_training_ratio': args.bpe_training_ratio,
            'rpe_horizon': args.rpe_horizon,
            'use_chunked_att': args.use_chunked_att,
            }


def create_gaussian_diffusion(args, DiffusionClass=SpacedDiffusion):
    # default params
    predict_xstart = True  # we always predict x_start (a.k.a. x0), that's our deal!
    steps = args.diffusion_steps
    scale_beta = 1.  # no scaling
    timestep_respacing = ''  # can be used for ddim sampling, we don't use it.
    learn_sigma = False
    rescale_timesteps = False

    betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return DiffusionClass(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        lambda_vel=args.lambda_vel,
        lambda_rcxyz=args.lambda_rcxyz,
        lambda_fc=args.lambda_fc,
        lambda_vel_rcxyz=args.lambda_vel_rcxyz,
    )


