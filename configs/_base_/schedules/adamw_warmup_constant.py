# configs/_base_/schedules/adamw_warmup_constant.py
optimizer = dict(type='AdamW', lr=2e-5, weight_decay=0.0)
lr_scheduler = dict(type='constant_with_warmup', num_warmup_steps=100)
