"""Matched online Flow-DPO objective for the GenTrack loop."""

_base_ = "train_gentrack_g1.py"

work_dir = "outputs/training/gentrack_ablation_dpo_g1"

trainer = dict(
    type="GenTrackFlowDPOTrainer",
    dpo_beta=100.0,
    dpo_min_reward_gap=0.2,
    dpo_timesteps_per_pair=8,
    dpo_replay_epochs=4,
    dpo_require_semantic_guard=False,
    dpo_pair_audit_dir="outputs/training/gentrack_ablation_dpo_g1/pairs",
    dpo_centralized_sonic=False,
    dpo_sonic_gpu_base=None,
    dpo_sonic_gpu_rank_stride=0,
)
