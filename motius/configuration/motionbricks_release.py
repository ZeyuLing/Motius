"""Whitelisted parameters for the official SOMA and G1 release (no dynamic targets)."""

from types import SimpleNamespace
import torch
from motius.motion.representation.motionbricks.release import (
    Stats,
    GlobalRootGlobalJoints,
)
from motius.motion.conditioning.motionbricks_release import (
    extract_feature_from_motion_rep,
)

UPSTREAM_REVISION = "54fc563aeaaf819a753d72cc9fdc6713989cdd43"


def representation(model, mean, std):
    stats = Stats()
    stats.register_from_tensors(
        torch.as_tensor(mean).float(), torch.as_tensor(std).float()
    )
    return GlobalRootGlobalJoints(
        fps=30,
        skeleton=SimpleNamespace(nbjoints=30 if model == "soma" else 34),
        name="novaskel30_dual_root_global_joints"
        if model == "soma"
        else "g1skel34_dual_root_global_joints",
        stats=stats,
    )


def g1_options(rep):
    common = dict(
        min_tokens=6,
        max_tokens=16,
        down_t=2,
        local_pose_feature="joint_positions_and_rotations_and_hip_height",
    )
    pose_args = dict(
        common,
        pose_root_mode="pose",
        pose_vqvae_motion_rep="local",
        cond_root_feature="root_without_hip_height",
        cond_root_feature_is_from_motion_rep="global",
    )
    root_args = dict(
        common,
        pose_root_mode="root",
        local_root_feature="root",
        global_root_feature="root",
        use_hard_num_token_emb_for_root_prediction=True,
    )
    pose_network_args = dict(
        pose_args,
        pose_token_mlp_num_layers=2,
        root_feat_width=256,
        pose_feat_width=640,
        token_length_feat_width=128,
        text_emb_dim=4096,
        text_embeddings=None,
        n_embd=1024,
        n_head=16,
        n_layers=16,
        pose_vqvae=dict(
            nb_code=100000000, code_dim=256, num_heads=8, has_codebook=True
        ),
        root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
    )
    root_network_args = dict(
        root_args,
        pose_feat_dim=256,
        local_root_feat_dim=64,
        global_root_feat_dim=64,
        text_emb_dim=4096,
        text_embeddings=None,
        input_feat_mlp_num_layers=2,
        n_embd=512,
        n_head=16,
        n_layers_shared=3,
        n_layers_root_token=3,
        width=512,
        depth=4,
        dilation_growth_rate=3,
        activation="relu",
        norm="None",
        pose_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
        root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
    )
    local = rep.dual_rep.local_motion_rep
    global_rep = rep.dual_rep.global_motion_rep

    def dimension(representation, feature):
        return extract_feature_from_motion_rep(
            torch.zeros(1, 1, len(representation.indices["all"])),
            representation,
            feature,
        ).shape[-1]

    pose_network_args.update(
        local_root_dim=dimension(global_rep, pose_args["cond_root_feature"]),
        local_pose_dim=dimension(local, pose_args["local_pose_feature"]),
    )
    root_network_args.update(
        local_root_dim=4,
        global_root_dim=5,
        local_pose_dim=dimension(local, root_args["local_pose_feature"]),
    )
    return dict(
        pose=pose_network_args,
        root=root_network_args,
        pose_args=pose_args,
        root_args=root_args,
        encoder_dim=dimension(local, "joint_positions_and_rotations_and_hip_height"),
        decoder_dim=dimension(local, "pose"),
        target_dim=dimension(local, "joint_positions_and_rotations_and_hip_height"),
        external_dim=dimension(local, "root_without_hip_height_without_heading"),
    )


def soma_options(rep):
    common = dict(
        min_tokens=6,
        max_tokens=16,
        down_t=2,
        local_pose_feature="joint_positions_and_rotations_and_hip_height",
    )
    pose_args = dict(
        common,
        pose_root_mode="pose",
        pose_vqvae_motion_rep="local",
        cond_root_feature="root_without_hip_height",
        cond_root_feature_is_from_motion_rep="global",
    )
    root_args = dict(
        common,
        pose_root_mode="root",
        local_root_feature="root",
        global_root_feature="root",
        use_hard_num_token_emb_for_root_prediction=True,
    )
    pose_network_args = dict(
        pose_args,
        pose_token_mlp_num_layers=2,
        root_feat_width=256,
        pose_feat_width=640,
        token_length_feat_width=128,
        text_emb_dim=4096,
        text_embeddings=None,
        n_embd=1024,
        n_head=16,
        n_layers=16,
        pose_vqvae=dict(nb_code=1048576, code_dim=160, num_heads=5, has_codebook=True),
        root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
    )
    root_network_args = dict(
        root_args,
        pose_feat_dim=256,
        local_root_feat_dim=64,
        global_root_feat_dim=64,
        text_emb_dim=4096,
        text_embeddings=None,
        input_feat_mlp_num_layers=2,
        n_embd=512,
        n_head=16,
        n_layers_shared=6,
        n_layers_root_token=2,
        width=512,
        depth=4,
        dilation_growth_rate=3,
        activation="relu",
        norm="None",
        pose_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
        root_vqvae=dict(nb_code=1024, code_dim=60, num_heads=1),
    )
    local = rep.dual_rep.local_motion_rep
    global_rep = rep.dual_rep.global_motion_rep

    def dimension(representation, feature):
        return extract_feature_from_motion_rep(
            torch.zeros(1, 1, len(representation.indices["all"])),
            representation,
            feature,
        ).shape[-1]

    pose_network_args.update(
        local_root_dim=dimension(global_rep, pose_args["cond_root_feature"]),
        local_pose_dim=dimension(local, pose_args["local_pose_feature"]),
    )
    root_network_args.update(
        local_root_dim=4,
        global_root_dim=5,
        local_pose_dim=dimension(local, root_args["local_pose_feature"]),
    )
    return dict(
        pose=pose_network_args,
        root=root_network_args,
        pose_args=pose_args,
        root_args=root_args,
        encoder_dim=dimension(local, "joint_positions_and_rotations_and_hip_height"),
        decoder_dim=dimension(local, "pose"),
        target_dim=dimension(local, "joint_positions_and_rotations_and_hip_height"),
        external_dim=dimension(local, "root_without_hip_height_without_heading"),
    )
