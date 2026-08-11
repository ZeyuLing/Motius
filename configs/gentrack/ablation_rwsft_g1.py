"""Matched online reward-weighted SFT objective for the GenTrack loop."""

_base_ = "train_gentrack_g1.py"

work_dir = "outputs/training/gentrack_ablation_rwsft_g1"

trainer = dict(type="GenTrackRewardWeightedSFTTrainer")
