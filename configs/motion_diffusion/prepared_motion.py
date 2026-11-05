"""Shared native prepared-motion training for MDM, CondMDI, OmniControl and ProjFlow.

Set MOTIUS_MOTION_METHOD, MOTIUS_MOTION_ARTIFACT and MOTIUS_MOTION_MANIFEST.
The manifest/NPZ format is documented by PreparedMotionDataset. Inputs already
use the selected checkpoint's normalization and observations; this recipe does
not claim the original raw-corpus augmentation or mixed-precision protocol.
"""

_base_ = []

import os
import json
from pathlib import Path

method = os.environ.get("MOTIUS_MOTION_METHOD", "mdm")
if method not in {
    "mdm",
    "condmdi",
    "omnicontrol",
    "projflow",
    "motionclr",
    "mld",
    "motionlcm",
}:
    raise ValueError("Unsupported MOTIUS_MOTION_METHOD: " + method)
artifact = Path(os.environ["MOTIUS_MOTION_ARTIFACT"]).expanduser().resolve()
manifest = os.environ["MOTIUS_MOTION_MANIFEST"]
metadata_file = artifact / (method + "_config.json")
metadata = json.loads(metadata_file.read_text()) if metadata_file.is_file() else {}
classes = {
    "motionlcm": ("MotionLCMBundle", "MotionLCMTrainer"),
    "mld": ("MLDBundle", "MLDTrainer"),
    "motionclr": ("MotionCLRBundle", "MotionCLRTrainer"),
    "mdm": ("MDMBundle", "MDMTrainer"),
    "condmdi": ("CondMDIBundle", "CondMDITrainer"),
    "omnicontrol": ("OmniControlBundle", "OmniControlTrainer"),
    "projflow": ("ProjFlowBundle", "ProjFlowTrainer"),
}
custom_imports = dict(
    imports=[
        "motius.models." + method,
        "motius.trainers." + method,
        "motius.datasets.prepared_motion",
        "motius.hooks",
    ],
    allow_failed_imports=False,
)
model = dict(
    type=classes[method][0],
    weights_path=str(artifact / "model.safetensors"),
    clip_weights_path=str(artifact / "clip.safetensors"),
    device="cpu",
)
if method in {"mld", "motionlcm"}:
    model = dict(
        type=classes[method][0],
        config=metadata["config"],
        vae_weights_path=str(artifact / "vae.safetensors"),
        denoiser_weights_path=str(artifact / "denoiser.safetensors"),
        mean_path=str(artifact / "Mean.npy"),
        std_path=str(artifact / "Std.npy"),
        text_encoder_name=str(artifact / metadata["text_encoder_name"]),
        device="cpu",
    )
elif method == "motionclr":
    model = dict(type="MotionCLRBundle", device="cpu", torch_dtype="float32")
    if metadata:
        model.update(
            network_config=metadata["network"],
            weights_path=str(artifact / metadata["weights"]),
            mean_path=str(artifact / "Mean.npy"),
            std_path=str(artifact / "Std.npy"),
            clip_path=str(artifact / metadata["text_encoder"]["path"]),
        )
    else:
        from motius.models.motionclr.bundle import (
            _network_config_from_options,
            _parse_official_options,
        )

        model.update(
            network_config=_network_config_from_options(
                _parse_official_options(artifact / "opt.txt")
            ),
            checkpoint_path=str(artifact / "latest.tar"),
            mean_path=str(artifact / "mean.npy"),
            std_path=str(artifact / "std.npy"),
        )
elif method == "projflow":
    model.update(
        model_id=metadata["model_id"],
        mean_path=str(artifact / "joints_mean.npy"),
        std_path=str(artifact / "joints_std.npy"),
    )
else:
    model["config"] = metadata["config"]
    if method == "condmdi":
        model.update(
            mean_abs_path=str(artifact / "Mean_abs_3d.npy"),
            std_abs_path=str(artifact / "Std_abs_3d.npy"),
        )
    else:
        model.update(
            mean_path=str(artifact / "Mean.npy"), std_path=str(artifact / "Std.npy")
        )
        if method == "omnicontrol":
            model.update(
                raw_mean_path=str(artifact / "Mean_raw.npy"),
                raw_std_path=str(artifact / "Std_raw.npy"),
            )
trainer = dict(type=classes[method][1])
if method == "mld":
    trainer["stage"] = os.environ.get("MOTIUS_MLD_STAGE", "diffusion")
elif method == "motionlcm":
    trainer["teacher_weights_path"] = os.environ["MOTIUS_MOTIONLCM_TEACHER"]
work_dir = os.environ.get("MOTIUS_MOTION_WORK_DIR", "work_dirs/" + method)
train_dataloader = dict(
    batch_size=int(os.environ.get("MOTIUS_MOTION_BATCH_SIZE", "8")),
    num_workers=int(os.environ.get("MOTIUS_MOTION_WORKERS", "4")),
    shuffle=True,
    drop_last=True,
    dataset=dict(type="PreparedMotionDataset", manifest=manifest, method=method),
)
optimizer = dict(
    type="AdamW",
    lr=float(os.environ.get("MOTIUS_MOTION_LR", "0.0001")),
    betas=(0.9, 0.99 if method == "projflow" else 0.999),
    weight_decay=0.01 if method == "motionclr" else 0.0,
)
train_cfg = dict(
    by_epoch=False, max_iters=int(os.environ.get("MOTIUS_MOTION_STEPS", "100000"))
)
if method == "motionclr":
    train_cfg["max_grad_norm"] = 1.0

accelerator = dict(
    mixed_precision="no",
    gradient_accumulation_steps=1,
    step_scheduler_with_optimizer=False,
    ddp_kwargs=dict(find_unused_parameters=True),
)
cuda_precision = dict(allow_tf32=False, float32_matmul_precision="highest")
sdpa_backends = dict(flash=False, mem_efficient=False, math=True)
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=int(os.environ.get("MOTIUS_MOTION_SAVE_EVERY", "1000")),
        max_keep_ckpts=3,
        save_last=True,
    )
)
seed = int(os.environ.get("MOTIUS_MOTION_SEED", "10"))
# Keep only serializable training configuration values.
if method == "motionclr" and not metadata:
    del _network_config_from_options, _parse_official_options
del Path, artifact, metadata, metadata_file, classes
