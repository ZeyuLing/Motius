"""Native original EDGE objectives, Adan and EMA on explicitly prepared windows."""

_base_ = []

import json
import os
from pathlib import Path
import numpy as np

artifact = Path(os.environ["MOTIUS_EDGE_ARTIFACT"]).expanduser().resolve()
metadata = json.loads((artifact / "edge_config.json").read_text())
with np.load(artifact / "normalizer.npz", allow_pickle=False) as statistics:
    scale, offset = statistics["scale"].tolist(), statistics["min"].tolist()
del statistics
custom_imports = dict(
    imports=[
        "motius.models.edge",
        "motius.trainers.edge",
        "motius.datasets.prepared_dance",
        "motius.hooks",
    ],
    allow_failed_imports=False,
)
model = dict(
    type="EDGEBundle",
    config=metadata["config"],
    weights_path=str(artifact / "model.safetensors"),
    normalizer_scale=scale,
    normalizer_min=offset,
    provenance=metadata.get("provenance"),
)
trainer = dict(type="EDGETrainer")
optimizer = dict(
    type="motius.optimizers.adan.Adan",
    lr=float(os.environ.get("MOTIUS_EDGE_LR", "0.0004")),
    weight_decay=0.02,
)
train_dataloader = dict(
    batch_size=int(os.environ.get("MOTIUS_EDGE_BATCH_SIZE", "32")),
    num_workers=int(os.environ.get("MOTIUS_EDGE_WORKERS", "4")),
    shuffle=True,
    drop_last=True,
    dataset=dict(
        type="PreparedEDGEDataset", manifest=os.environ["MOTIUS_EDGE_MANIFEST"]
    ),
)
work_dir = os.environ.get("MOTIUS_EDGE_WORK_DIR", "work_dirs/edge_native")
train_cfg = dict(
    by_epoch=False, max_iters=int(os.environ.get("MOTIUS_EDGE_STEPS", "100000"))
)
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
        interval=int(os.environ.get("MOTIUS_EDGE_SAVE_EVERY", "1000")),
        save_last=True,
        max_keep_ckpts=3,
    )
)
seed = int(os.environ.get("MOTIUS_EDGE_SEED", "37"))
del Path, artifact, metadata, scale, offset
