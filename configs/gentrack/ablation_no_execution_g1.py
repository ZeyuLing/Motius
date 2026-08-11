"""GenTrack ablation with tracker execution advantage disabled."""

_base_ = "train_gentrack_g1.py"

work_dir = "outputs/training/gentrack_ablation_no_execution_g1"

trainer = dict(
    execution_reward_mode="none",
    grpo_physical_advantage_weight=0.0,
)
