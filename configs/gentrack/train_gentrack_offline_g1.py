"""Equal-budget frozen-pool GenTrack generator control.

Point ``MOTIUS_DATA_ROOT`` and ``MOTIUS_TRAIN_MANIFEST`` to a preprocessed
G1-38 replay pool.  The manifest uses the same
``ManifestTextMotionDataset`` contract as HYMotion-G1.
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
    "outputs/training/gentrack_offline_g1",
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
    type="GenTrackOfflineTrainer",
    num_samples=1,
    enable_reward=False,
    offline_anchor_weight=0.01,
    gt_anchor_interval=4,
)
train_dataloader = dict(batch_size=2, num_workers=4)
optimizer = dict(type="AdamW", lr=1e-6, betas=[0.9, 0.99], weight_decay=0.0)
lr_scheduler = None
train_cfg = dict(
    _delete_=True,
    by_epoch=False,
    max_iters=1440,
    val_interval=999999,
    max_grad_norm=1.0,
)
accelerator = dict(mixed_precision="no", gradient_accumulation_steps=1)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=5),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=False,
        interval=120,
        max_keep_ckpts=12,
        save_last=True,
    ),
)
load_from = dict(_delete_=True, path=g0_checkpoint, load_scope="model")
val_dataloader = None
val_evaluator = None
val_visualizer = None
