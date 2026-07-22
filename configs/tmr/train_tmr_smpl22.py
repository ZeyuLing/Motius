"""Public TMR recipe for 22-joint position features."""

import os


_base_ = "../_base_/default_runtime.py"

dataset_dir = os.environ.get("MOTIUS_DATA_ROOT", "data/training/tmr")
work_dir = os.environ.get("MOTIUS_WORK_DIR", "work_dirs/tmr_smpl22")

custom_imports = dict(
    imports=[
        "motius.datasets.motion.tmr_text_motion_dataset",
        "motius.models.tmr",
        "motius.trainers.tmr",
    ],
    allow_failed_imports=False,
)

model = dict(
    type="TMRBundle",
    motion_nfeats=66,
    text_nfeats=768,
    vae=True,
    arch=dict(
        latent_dim=256,
        ff_size=1024,
        num_layers=6,
        num_heads=4,
        dropout=0.1,
        activation="gelu",
    ),
    lmd=dict(recons=1.0, latent=1.0e-5, kl=1.0e-5, contrastive=0.1),
    temperature=0.1,
    threshold_selfsim=0.8,
    sample_mean=False,
)
trainer = dict(type="TMRTrainer")

train_dataloader = dict(
    batch_size=64,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    dataset=dict(
        type="TMRTextMotionDataset",
        dataset_dir=dataset_dir,
        split="train",
        fps=30.0,
        nfeats=66,
        token_modelname="distilbert-base-uncased",
        sentence_modelname="sentence-transformers/all-mpnet-base-v2",
        min_seconds=0.5,
        max_seconds=20.0,
    ),
)
val_dataloader = None

optimizer = dict(type="AdamW", lr=1.0e-4, betas=[0.9, 0.99], weight_decay=0.0)
lr_scheduler = None
accelerator = dict(
    mixed_precision="no",
    gradient_accumulation_steps=1,
    dataloader_device_placement=False,
)
train_cfg = dict(by_epoch=True, max_epochs=300, val_interval=100000, max_grad_norm=1.0)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=25),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        save_last=True,
    ),
)
