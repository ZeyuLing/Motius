"""Shared condmdi model and diffusion configuration."""
from copy import deepcopy

DEFAULT_CONFIG = {
    "dataset": "humanml",
    "arch": "unet",
    "latent_dim": 512,
    "ff_size": 1024,
    "layers": 8,
    "dim_mults": [2, 2, 2, 2],
    "cond_mask_prob": 0.1,
    "emb_trans_dec": False,
    "noise_schedule": "cosine",
    "diffusion_steps": 1000,
    "sigma_small": True,
    "predict_xstart": True,
    "clip_range": 6.0,
    "unet_adagn": True,
    "unet_zero": True,
    "out_mult": False,
    "xz_only": False,
    "keyframe_conditioned": True,
    "keyframe_selection_scheme": "random_frames",
    "zero_keyframe_loss": False,
    "use_fp16": True,
    "abs_3d": True,
    "lambda_rcxyz": 0.0,
    "lambda_vel": 0.0,
    "lambda_fc": 0.0,
    "use_random_proj": False,
    "traj_only": False,
    "apply_zero_mask": False,
    "traj_extra_weight": 1.0,
    "time_weighted_loss": False,
    "train_x0_as_eps": False,
}

def normalize_config(config=None) -> dict:
    merged = dict(DEFAULT_CONFIG)
    merged.update(dict(config or {}))
    merged["dim_mults"] = list(merged["dim_mults"])
    return merged
