"""Shared omnicontrol model and diffusion configuration."""
from copy import deepcopy

DEFAULT_CONFIG = {
    "dataset": "humanml",
    "cond_mode": "both_text_spatial",
    "arch": "trans_enc",
    "emb_trans_dec": False,
    "layers": 8,
    "latent_dim": 512,
    "cond_mask_prob": 0.1,
    "noise_schedule": "cosine",
    "diffusion_steps": 1000,
    "sigma_small": True,
    "lambda_rcxyz": 0.0,
    "lambda_vel": 0.0,
    "lambda_fc": 0.0,
}

def normalize_config(config=None):
    merged = deepcopy(DEFAULT_CONFIG)
    if config:
        merged.update(dict(config))
    if merged["dataset"] != "humanml":
        raise ValueError("The released OmniControl artifact supports HumanML3D only")
    return merged
