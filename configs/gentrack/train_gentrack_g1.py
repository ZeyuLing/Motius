"""Paper-facing GenTrack G1 generator post-training recipe.

The round orchestrator writes a lagged physical-judge specification and points
``PHYSFLOW_JUDGE_SPEC`` at it.  ``MOTIUS_GENTRACK_G0_CHECKPOINT`` must identify
the immutable HYMotion-G1 initialization used both for warm start and the
same-noise counterfactual reference.
"""

import os


_base_ = "../hymotion_t2m/train_hymotion_g1.py"

custom_imports = dict(
    imports=[
        "motius.datasets.text_motion",
        "motius.models.hymotion_t2m",
        "motius.models.gentrack",
        "motius.trainers.gentrack",
    ],
    allow_failed_imports=False,
)

work_dir = os.environ.get(
    "MOTIUS_WORK_DIR",
    "outputs/training/gentrack_g1",
)
g0_checkpoint = os.environ.get(
    "MOTIUS_GENTRACK_G0_CHECKPOINT",
    "checkpoints/models/hymotion_g1/g0",
)

model = dict(
    type="GenTrackG1Bundle",
    sample_steps=30,
    sample_guidance=1.0,
    immutable_anchor_checkpoint=g0_checkpoint,
    require_immutable_anchor=True,
)

trainer = dict(
    _delete_=True,
    type="GenTrackFlowGRPOTrainer",
    num_samples=4,
    diffusion_steps=30,
    enable_reward=True,
    judge_backend="protomotions",
    grpo_eta=0.7,
    grpo_clip_range=0.2,
    grpo_replay_epochs=4,
    min_grpo_replay_epochs=2,
    require_effective_grpo_replay=True,
    grpo_reference_kl_weight=0.02,
    grpo_invalid_reference_penalty=2.0,
    gt_weight=1.0,
    gt_anchor_interval=0,
    gt_anchor_update_interval=1,
    gt_g0_distill_weight=1.0,
    frontier_mode=False,
    quality_judge="quality",
    trainee_judge="trainee",
    frontier_selection="relative_hard",
    frontier_topk_per_prompt=1,
    tracker_replay_selection="on_policy_all",
    tracker_replay_samples_per_prompt=4,
    tracker_replay_max_joint_vel=0.0,
    tracker_replay_max_root_vel=0.0,
    accept_min_completion=0.9,
    accept_require_no_fall=True,
    accept_min_joint_std=0.0,
    accept_max_root_disp_if_frozen=None,
    quality_max_joint_vel=0.0,
    quality_max_root_vel=0.0,
    tracker_pool_dir=None,
    tracker_qpos_pool_dir=(
        "outputs/training/gentrack_g1/tracker_qpos_pool"
    ),
    export_gt_to_pool=False,
    pool_max_motions=13337,
)

train_dataloader = dict(batch_size=2, num_workers=4)
optimizer = dict(type="AdamW", lr=5e-6, betas=[0.9, 0.99], weight_decay=0.0)
lr_scheduler = None
accelerator = dict(mixed_precision="no", gradient_accumulation_steps=1)
train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=3000,
    val_interval=999999,
    max_grad_norm=1.0,
)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=5),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=50,
        max_keep_ckpts=12,
        save_last=True,
    ),
)
load_from = dict(_delete_=True, path=g0_checkpoint, load_scope="model")
val_dataloader = None
val_evaluator = None
val_visualizer = None
