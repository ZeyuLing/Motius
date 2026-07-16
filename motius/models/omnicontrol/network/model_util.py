"""Factories for the released HumanML3D OmniControl model."""

from __future__ import annotations

from copy import deepcopy

from .cfg_sampler import ClassifierFreeSampleModel
from .cmdm import CMDM
from .diffusion import gaussian_diffusion as gd
from .diffusion.respace import SpacedDiffusion, space_timesteps


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


def build_model(config=None):
    cfg = normalize_config(config)
    return CMDM(
        modeltype="",
        njoints=263,
        nfeats=1,
        num_actions=1,
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=True,
        latent_dim=cfg["latent_dim"],
        ff_size=1024,
        num_layers=cfg["layers"],
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        data_rep="hml_vec",
        cond_mode=cfg["cond_mode"],
        cond_mask_prob=cfg["cond_mask_prob"],
        action_emb="tensor",
        arch=cfg["arch"],
        emb_trans_dec=cfg["emb_trans_dec"],
        clip_version="ViT-B/32",
        dataset="humanml",
    )


def load_model_wo_clip(model, state_dict):
    state = dict(state_dict)
    for key in list(state):
        if "sequence_pos_encoder.pe" in key:
            state.pop(key)
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [
        key for key in missing
        if not key.startswith("clip_model.") and "sequence_pos_encoder.pe" not in key
    ]
    invalid_unexpected = [key for key in unexpected if not key.startswith("rot2xyz.")]
    if invalid_missing:
        raise RuntimeError(f"missing OmniControl checkpoint keys: {invalid_missing[:8]}")
    if invalid_unexpected:
        raise RuntimeError(f"unexpected OmniControl checkpoint keys: {invalid_unexpected[:8]}")


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


__all__ = [
    "ClassifierFreeSampleModel",
    "DEFAULT_CONFIG",
    "build_diffusion",
    "build_model",
    "load_model_wo_clip",
    "normalize_config",
]
