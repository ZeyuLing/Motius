"""Public HYMotion-G1 flow-matching training recipe."""

import os


_base_ = "./train_hymotion_t2m.py"

data_root = os.environ.get("MOTIUS_DATA_ROOT", "data/training/hymotion_g1")
manifest = os.environ.get("MOTIUS_TRAIN_MANIFEST", "train.json")
stats_dir = os.environ.get(
    "MOTIUS_MOTION_STATS",
    "checkpoints/models/hymotion_g1/stats",
)
pretrained_weights = os.environ.get("MOTIUS_PRETRAINED_WEIGHTS")

work_dir = os.environ.get("MOTIUS_WORK_DIR", "outputs/training/hymotion_g1")
motion_dim = 38

model = dict(
    motion_transformer=dict(
        input_dim=motion_dim,
        feat_dim=1024,
        output_dim=motion_dim,
        num_layers=18,
        num_heads=16,
    ),
    mean_std_dir=stats_dir,
    motion_type="g1_29dof",
    body_model_path=None,
    motion_weights_path=pretrained_weights,
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        manifest=manifest,
        motion_dim=motion_dim,
    ),
)
