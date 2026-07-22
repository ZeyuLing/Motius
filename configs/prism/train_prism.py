"""Public PRISM flow-matching training recipe."""

import os


_base_ = "../_base_/default_runtime.py"

data_root = os.environ.get("MOTIUS_DATA_ROOT", "data/training")
manifest = os.environ.get("MOTIUS_TRAIN_MANIFEST", "train.json")
pretrained = os.environ.get(
    "MOTIUS_PRISM_PRETRAINED", "ZeyuLing/motius-prism-kt-humanml3d"
)
null_embedding = os.environ.get("MOTIUS_NULL_TEXT_FEATURE")

custom_imports = dict(
    imports=[
        "motius.datasets.text_motion",
        "motius.models.prism",
        "motius.trainers.prism",
    ],
    allow_failed_imports=False,
)

work_dir = os.environ.get("MOTIUS_WORK_DIR", "work_dirs/prism_training")

model = dict(
    type="PRISMBundle",
    checkpoint_path=pretrained,
    transformer_dtype="bf16",
    text_dtype="bf16",
    training=True,
    latent_sample_method="mode",
)

trainer = dict(
    type="PrismTrainer",
    condition_num_frames=[1, 5, 9],
    frame_condition_rate=0.1,
    prompt_drop_rate=0.1,
    max_text_length=128,
    translation_loss_weight=0.5,
    null_embedding_path=null_embedding,
)

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    pin_memory=True,
    shuffle=True,
    drop_last=True,
    dataset=dict(
        type="ManifestTextMotionDataset",
        data_root=data_root,
        manifest=manifest,
        motion_dim=138,
        max_frames=360,
        max_text_length=128,
        training=True,
        pad_mode="replicate",
    ),
)
val_dataloader = None

optimizer = dict(type="AdamW", lr=1.0e-4, betas=[0.9, 0.99], weight_decay=0.0)
lr_scheduler = None

accelerator = dict(
    mixed_precision="bf16",
    gradient_accumulation_steps=1,
    dataloader_device_placement=False,
)
train_cfg = dict(by_epoch=True, max_epochs=100, val_interval=10, max_grad_norm=1.0)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=10),
    checkpoint=dict(
        type="CheckpointHook",
        by_epoch=True,
        interval=1,
        max_keep_ckpts=5,
        save_last=True,
    ),
)
