"""CondMDI network and diffusion factories."""

from __future__ import annotations

from argparse import Namespace

from .diffusion import (
    DiffusionConfig,
    LossType,
    ModelMeanType,
    ModelVarType,
    SpacedDiffusion,
    get_named_beta_schedule,
    space_timesteps,
)
from .unet import MDM_UNET


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


def build_model(config=None):
    cfg = normalize_config(config)
    if cfg["dataset"] != "humanml":
        raise ValueError("The released CondMDI artifact supports HumanML3D only")
    return MDM_UNET(
        modeltype="",
        njoints=263,
        nfeats=1,
        num_actions=1,
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=True,
        latent_dim=cfg["latent_dim"],
        ff_size=cfg["ff_size"],
        num_layers=cfg["layers"],
        num_heads=4,
        dropout=0.1,
        activation="gelu",
        data_rep="hml_vec",
        cond_mode="text",
        cond_mask_prob=cfg["cond_mask_prob"],
        action_emb="tensor",
        arch=cfg["arch"],
        emb_trans_dec=cfg["emb_trans_dec"],
        clip_version="ViT-B/32",
        dataset="humanml",
        dim_mults=cfg["dim_mults"],
        adagn=cfg["unet_adagn"],
        zero=cfg["unet_zero"],
        unet_out_mult=cfg["out_mult"],
        xz_only=cfg["xz_only"],
        keyframe_conditioned=cfg["keyframe_conditioned"],
        keyframe_selection_scheme=cfg["keyframe_selection_scheme"],
        zero_keyframe_loss=cfg["zero_keyframe_loss"],
    )


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


def load_model_wo_clip(model, state_dict):
    state = dict(state_dict)
    state.pop("sequence_pos_encoder.pe", None)
    state.pop("embed_timestep.sequence_pos_encoder.pe", None)
    state = {k: v for k, v in state.items() if not k.startswith("clip_model.")}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected CondMDI checkpoint keys: {unexpected[:8]}")
    invalid_missing = [
        k for k in missing
        if not k.startswith("clip_model.") and "sequence_pos_encoder" not in k
    ]
    if invalid_missing:
        raise RuntimeError(f"missing CondMDI checkpoint keys: {invalid_missing[:8]}")


__all__ = [
    "DEFAULT_CONFIG",
    "build_diffusion",
    "build_model",
    "load_model_wo_clip",
    "normalize_config",
]
