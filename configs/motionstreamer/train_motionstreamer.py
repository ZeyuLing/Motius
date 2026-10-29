"""Original TAE, T2M AR or streaming AR recipe through the shared native Runner."""

_base_ = "../_base_/default_runtime.py"
import os

custom_imports = dict(
    imports=[
        "motius.models.motionstreamer",
        "motius.trainers.motionstreamer",
        "motius.datasets.motionstreamer",
    ],
    allow_failed_imports=False,
)
stage = os.environ.get("MOTIUS_MS_STAGE", "ar")
if stage not in {"tae", "ar", "streaming"}:
    raise ValueError("MOTIUS_MS_STAGE must be tae, ar or streaming")
work_dir = os.environ.get("MOTIUS_WORK_DIR", "outputs/training/motionstreamer_" + stage)
artifact = os.environ.get(
    "MOTIUS_MS_ARTIFACT", "checkpoints/motionstreamer/t2m_humanml272"
)
hml_root = os.environ.get("MOTIUS_MS_HML_ROOT", "data/humanml3d_272")
babel_root = os.environ.get("MOTIUS_MS_BABEL_ROOT", "data/babel_272")
stream_root = os.environ.get("MOTIUS_MS_STREAM_ROOT", "data/babel_272_stream")
world_size = int(os.environ.get("WORLD_SIZE", "1"))
total_steps = int(
    os.environ.get("MOTIUS_MS_TOTAL_STEPS", "2000000" if stage == "tae" else "100000")
)
warmup_steps = int(
    os.environ.get(
        "MOTIUS_MS_WARMUP_STEPS", "1000" if stage == "tae" else str(total_steps // 10)
    )
)
model = dict(
    type="MotionStreamerBundle",
    tae_weights_path=os.path.join(artifact, "tae.safetensors"),
    ar_weights_path=os.path.join(artifact, "ar.safetensors"),
    mean_path=os.path.join(artifact, "Mean.npy"),
    std_path=os.path.join(artifact, "Std.npy"),
    text_model_name=os.path.join(artifact, "text_encoder/sentence-t5-xxl"),
    initialize_from_scratch=["tae" if stage == "tae" else "ar"],
    load_text_model=stage != "tae",
    device="cpu",
)

trainer = dict(
    type="MotionStreamerTrainer",
    stage=stage,
    total_steps=total_steps,
    batch_multiplier=4,
    root_weight=7.0,
)
if stage == "tae":
    dataset = dict(
        type="MotionStreamerTAEDataset",
        dataset_name=os.environ.get("MOTIUS_MS_DATASET", "t2m_272"),
        window_size=64,
        hml_root=hml_root,
        babel_root=babel_root,
        babel_stream_root=stream_root,
    )
    optimizer = dict(type="AdamW", lr=5e-5, betas=(0.9, 0.99), weight_decay=0.0)
    lr_scheduler = dict(
        type="motius.trainers.motionstreamer.schedule.MotionStreamerTAESchedule",
        warmup_iters=warmup_steps,
        milestones=[1900000],
        gamma=0.05,
    )
    max_iters = max(0, warmup_steps - 1) + total_steps
else:
    latent_dir = os.environ.get(
        "MOTIUS_MS_LATENTS",
        (
            os.path.join(hml_root, "t2m_latents")
            if stage == "ar"
            else os.path.join(stream_root, "t2m_babel_latents")
        ),
    )
    dataset = dict(
        type=(
            "MotionStreamerTextTokenDataset"
            if stage == "ar"
            else "MotionStreamerStreamingTokenDataset"
        ),
        dataset_name="t2m_272" if stage == "ar" else "t2m_babel_272",
        latent_dir=latent_dir,
        hml_root=hml_root,
        babel_root=babel_root,
        babel_stream_root=stream_root,
    )
    optimizer = dict(type="AdamW", lr=1e-4, betas=(0.9, 0.99), weight_decay=1e-6)
    lr_scheduler = dict(
        type="motius.trainers.motionstreamer.schedule.MotionStreamerWarmupCosine",
        warmup_iters=warmup_steps,
        total_iters=total_steps,
    )
    max_iters = total_steps + 1
train_dataloader = dict(
    batch_size=int(
        os.environ.get(
            "MOTIUS_MS_BATCH_SIZE", str((128 if stage == "tae" else 256) // world_size)
        )
    ),
    num_workers=int(os.environ.get("MOTIUS_MS_WORKERS", "8")),
    shuffle=True,
    drop_last=True,
    pin_memory=True,
    dataset=dataset,
)
val_dataloader = None
accelerator = dict(
    mixed_precision="no",
    gradient_accumulation_steps=1,
    step_scheduler_with_optimizer=False,
    dataloader_device_placement=False,
    ddp_kwargs=dict(static_graph=stage != "tae"),
)
# Match the FP32/math-attention runtime used for the source certificates.
cuda_precision = dict(allow_tf32=False, float32_matmul_precision="highest")
sdpa_backends = dict(flash=False, mem_efficient=False, math=True)
train_cfg = dict(by_epoch=False, max_iters=max_iters, max_grad_norm=None)
default_hooks = dict(
    logger=dict(type="LoggerHook", interval=1, iter_interval=100),
    checkpoint=dict(
        type="CheckpointHook", interval=10000, max_keep_ckpts=3, save_last=True
    ),
)
