"""Native DART primitive training; original full-corpus curriculum is separate."""

_base_ = []
import os

stage = os.environ.get("MOTIUS_DART_STAGE", "denoiser")
custom_imports = dict(
    imports=[
        "motius.models.dart",
        "motius.trainers.dart",
        "motius.datasets.prepared_dart",
        "motius.hooks",
    ],
    allow_failed_imports=False,
)
model = dict(
    type="DARTBundle", artifact_dir=os.environ["MOTIUS_DART_ARTIFACT"], device="cpu"
)
trainer = dict(
    type="DARTTrainer",
    stage=stage,
    learning_rate=float(os.environ.get("MOTIUS_DART_LR", "0.0001")),
)
optimizer = dict(
    type="AdamW",
    lr=float(os.environ.get("MOTIUS_DART_LR", "0.0001")),
    weight_decay=0.01,
)
train_dataloader = dict(
    batch_size=int(os.environ.get("MOTIUS_DART_BATCH_SIZE", "32")),
    num_workers=int(os.environ.get("MOTIUS_DART_WORKERS", "0")),
    shuffle=True,
    drop_last=True,
    dataset=dict(
        type="PreparedDARTDataset", manifest=os.environ["MOTIUS_DART_MANIFEST"]
    ),
)
train_cfg = dict(
    by_epoch=False,
    max_iters=int(os.environ.get("MOTIUS_DART_STEPS", "100000")),
    max_grad_norm=1.0,
)
work_dir = os.environ.get("MOTIUS_DART_WORK_DIR", "work_dirs/dart_" + stage)
accelerator = dict(
    mixed_precision="no",
    gradient_accumulation_steps=1,
    step_scheduler_with_optimizer=False,
    ddp_kwargs=dict(find_unused_parameters=True),
    dataloader_config=dict(
        use_seedable_sampler=True,
        data_seed=int(os.environ.get("MOTIUS_DART_SEED", "37")),
    ),
)
cuda_precision = dict(allow_tf32=False, float32_matmul_precision="highest")
sdpa_backends = dict(flash=False, mem_efficient=False, math=True)
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=int(os.environ.get("MOTIUS_DART_SAVE_EVERY", "1000")),
        save_last=True,
        max_keep_ckpts=3,
    )
)
seed = int(os.environ.get("MOTIUS_DART_SEED", "37"))
