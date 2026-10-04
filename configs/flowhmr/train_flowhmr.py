"""Official FlowHMR base recipe: 25 x 10,000 steps, FP32, 30 fps."""
_base_ = "../_base_/default_runtime.py"
import os
custom_imports = dict(imports=["motius.models.flowhmr", "motius.trainers.flowhmr",
                              "motius.datasets.flowhmr"], allow_failed_imports=False)
work_dir = os.environ.get("MOTIUS_WORK_DIR", "outputs/training/flowhmr")
body_model_path = os.environ.get("MOTIUS_SMPLH_MODEL", "outputs/checkpoints/flowhmr/body_models/smplh/neutral/model.npz")
j_regressor_path = os.environ.get("MOTIUS_J_REGRESSOR", "outputs/checkpoints/flowhmr/body_models/smpl_neutral_J_regressor.pt")
model = dict(type="FlowHMRBundle", body_model_path=body_model_path,
             j_regressor_path=j_regressor_path,
             ckpt_path=os.environ.get("MOTIUS_PRETRAINED_WEIGHTS"))
trainer = dict(type="FlowHMRTrainer")
train_dataloader = dict(batch_size=8, num_workers=4, shuffle=False, pin_memory=True,
    drop_last=True, persistent_workers=True,
    dataset=dict(type="FlowHMRWebDataset", body_model_path=body_model_path,
                 tar_urls=os.environ.get("MOTIUS_FLOWHMR_SHARDS", "data/flowhmr/train-{00000..00025}.tar"),
                 augmentation_type="original", cfg_dropout_prob=0.1,
                 max_len=360, shuffle_buffer=2000, resampled=True))
val_dataloader = None
optimizer = dict(type="AdamW", lr=1e-4, weight_decay=0.01, betas=(0.9, 0.999))
lr_scheduler = dict(type="CosineAnnealingLR", T_max=250000, eta_min=1e-5)
accelerator = dict(mixed_precision="no", gradient_accumulation_steps=1,
                   dataloader_device_placement=False)
train_cfg = dict(by_epoch=False, max_iters=250000, max_grad_norm=10.0)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=10),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=10000,
                    max_keep_ckpts=5, save_last=True))
