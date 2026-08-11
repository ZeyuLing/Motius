"""GenTrack ablation using only the binary fall-only execution outcome."""

_base_ = "train_gentrack_g1.py"

work_dir = "outputs/training/gentrack_ablation_success_only_g1"

trainer = dict(execution_reward_mode="success")
