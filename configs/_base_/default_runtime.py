# configs/_base_/default_runtime.py
# Default runtime settings

work_dir = 'work_dirs/default'
auto_resume = False
load_from = None

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1000,
        max_keep_ckpts=3,
        save_last=True,
    ),
    logger=dict(
        type='LoggerHook',
        interval=1,
    ),
)

accelerator = dict(
    mixed_precision='no',
    gradient_accumulation_steps=1,
)
