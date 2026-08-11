"""MaskControl HumanML3D release configuration.

The official all-anchor checkpoint controls pelvis, feet, head, and wrists.
Body-part and sequential generation use the paper's zero-shot iterative
composition protocol; they are not separately trained checkpoints.
"""

model = dict(
    type="MaskControlBundle",
    motion_representation="HumanML3D-263",
    fps=20,
    max_frames=392,
    control_joint_ids=(0, 10, 11, 15, 20, 21),
    vq=dict(
        nb_code=512,
        code_dim=512,
        num_quantizers=6,
        down_t=2,
        stride_t=2,
    ),
    control=dict(
        latent_dim=384,
        ff_size=1024,
        n_layers=8,
        n_heads=6,
        dropout=0.2,
        cond_drop_prob=0.1,
    ),
    residual=dict(
        latent_dim=384,
        ff_size=1024,
        n_layers=8,
        n_heads=6,
        dropout=0.2,
        cond_drop_prob=0.2,
    ),
)

inference = dict(
    time_steps=10,
    cond_scale=4.0,
    residual_cond_scale=5.0,
    temperature=1.0,
    residual_temperature=1.0,
    control=dict(
        each_iterations=100,
        final_iterations=600,
        each_lr=0.06,
        final_lr=0.06,
    ),
    sequential=dict(
        transition_padding=5,
        each_iterations=300,
        final_iterations=300,
    ),
)
