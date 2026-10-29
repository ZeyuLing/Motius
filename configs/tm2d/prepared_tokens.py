"""Joint TM2D audio/text token training through the shared Runner."""

_base_ = []
import json
import os
from pathlib import Path
import numpy as np

artifact = Path(os.environ["MOTIUS_TM2D_ARTIFACT"])
metadata = json.loads((artifact / "tm2d_config.json").read_text())
custom_imports = dict(
    imports=[
        "motius.models.tm2d",
        "motius.trainers.tm2d",
        "motius.datasets.prepared_tm2d",
        "motius.hooks",
    ],
    allow_failed_imports=False,
)
model = dict(
    type="TM2DBundle",
    config=metadata["config"],
    vocabulary_path=str(artifact / "vocab.json"),
    mean=np.load(artifact / "mean.npy").tolist(),
    std=np.load(artifact / "std.npy").tolist(),
    component_weights={
        name: str(artifact / (name + ".safetensors"))
        for name in [
            "audio_transformer",
            "text_transformer",
            "vq_encoder",
            "vq_decoder",
            "quantizer",
        ]
    },
    provenance=metadata.get("provenance"),
)
trainer = dict(
    type="TM2DTrainer",
    label_smoothing=os.environ.get("MOTIUS_TM2D_LABEL_SMOOTHING", "0") == "1",
)
optimizer = dict(type="Adam", lr=float(os.environ.get("MOTIUS_TM2D_LR", "0.0001")))
train_dataloader = dict(
    batch_size=int(os.environ.get("MOTIUS_TM2D_BATCH_SIZE", "32")),
    num_workers=0,
    shuffle=True,
    drop_last=True,
    dataset=dict(
        type="PreparedTM2DDataset", manifest=os.environ["MOTIUS_TM2D_MANIFEST"]
    ),
)
train_cfg = dict(
    by_epoch=False, max_iters=int(os.environ.get("MOTIUS_TM2D_STEPS", "100000"))
)
work_dir = os.environ.get("MOTIUS_TM2D_WORK_DIR", "work_dirs/tm2d_native")
seed = int(os.environ.get("MOTIUS_TM2D_SEED", "37"))
accelerator = dict(
    mixed_precision="no",
    gradient_accumulation_steps=1,
    step_scheduler_with_optimizer=False,
    ddp_kwargs=dict(find_unused_parameters=True),
    dataloader_config=dict(use_seedable_sampler=True, data_seed=seed),
)
cuda_precision = dict(allow_tf32=False, float32_matmul_precision="highest")
sdpa_backends = dict(flash=False, mem_efficient=False, math=True)
default_hooks = dict(
    checkpoint=dict(
        type="CheckpointHook",
        interval=int(os.environ.get("MOTIUS_TM2D_SAVE_EVERY", "1000")),
        save_last=True,
        max_keep_ckpts=3,
    )
)
del Path, artifact, metadata
