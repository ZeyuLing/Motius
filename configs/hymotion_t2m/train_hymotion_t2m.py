"""Public HYMotion T2M flow-matching training recipe."""

import os


_base_ = "../_base_/default_runtime.py"

data_root = os.environ.get("MOTIUS_DATA_ROOT", "data/training")
manifest = os.environ.get("MOTIUS_TRAIN_MANIFEST", "train.json")
stats_dir = os.environ.get("MOTIUS_MOTION_STATS", "data/training/stats")
pretrained_weights = os.environ.get("MOTIUS_PRETRAINED_WEIGHTS")

custom_imports = dict(
    imports=[
        "motius.datasets.text_motion",
        "motius.models.hymotion_t2m",
        "motius.trainers.hymotion_t2m",
    ],
    allow_failed_imports=False,
)

work_dir = os.environ.get("MOTIUS_WORK_DIR", "work_dirs/hymotion_t2m_training")
motion_dim = 201

model = dict(
    type="HyMotionT2MBundle",
    motion_transformer=dict(
        type="HunyuanMotionT2MMMDiT",
        trainable=True,
        input_dim=motion_dim,
        feat_dim=1280,
        output_dim=motion_dim,
        ctxt_input_dim=4096,
        vtxt_input_dim=768,
        num_layers=27,
        num_heads=20,
        mlp_ratio=4.0,
        mlp_act_type="gelu_tanh",
        norm_type="layer",
        qk_norm_type="rms",
        qkv_bias=True,
        dropout=0.0,
        text_refiner_cfg=dict(num_layers=2),
        final_layer_cfg=dict(act_type="silu"),
        mask_mode="narrowband",
        apply_rope_to_single_branch=False,
        insert_start_token=False,
        with_long_skip_connection=False,
        time_factor=1000.0,
    ),
    text_encoder=None,
    mean_std_dir=stats_dir,
    motion_type="smpl_22",
    pred_type="velocity",
    uncondition_mode=False,
    losses_cfg=dict(loss_type="smooth_l1", velocity_weight=1.0),
    noise_scheduler_cfg=dict(method="euler"),
    infer_noise_scheduler_cfg=dict(validation_steps=50),
    cond_mask_prob=0.1,
    enable_special_game_feat=False,
    train_special_game_embeddings=False,
    vtxt_input_dim=768,
    ctxt_input_dim=4096,
    body_model_path=None,
    motion_weights_path=pretrained_weights,
)
trainer = dict(type="HyMotionT2MTrainer", val_num_steps=10, max_text_len=128)

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
        motion_dim=motion_dim,
        max_frames=360,
        max_text_length=128,
        training=True,
        pad_mode="replicate",
        require_text_features=True,
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
