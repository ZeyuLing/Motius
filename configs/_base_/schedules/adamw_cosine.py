# configs/_base_/schedules/adamw_cosine.py
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-2)
lr_scheduler = dict(type='cosine_with_warmup', num_warmup_steps=100)
