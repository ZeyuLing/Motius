"""Frozen source-only D3 MotionRepair configuration used in the paper."""

custom_imports = dict(
    imports=["motius.pipelines.motionrepair"],
    allow_failed_imports=False,
)

pipeline = dict(
    type="MotionRepairPipeline",
    config=dict(
        regularization=3.0,
        ground_quantile=0.98,
        prefix_frames=10,
        transition_frames=30,
        penetration_margin_m=0.049,
        support_tolerance=1.0e-12,
    ),
)

motion = dict(
    representation="motion135",
    skeleton="smpl22",
    up_axis="+Y",
    coordinate_unit="meter",
)

provenance = dict(
    policy="lambda=3, mode=both, ground=prefix_q98",
    learned_artifacts_used=False,
    clean_target_required=False,
    corruption_label_required=False,
)
