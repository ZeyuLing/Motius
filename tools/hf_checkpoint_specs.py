"""Canonical Hugging Face artifact specifications for released Motius methods."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CheckpointSpec:
    source_repo: str
    target_repo: str
    pipeline_class: str
    bundle_class: str
    tasks: tuple[str, ...]

    @property
    def class_name(self) -> str:
        return self.pipeline_class.rsplit(".", 1)[-1]


CHECKPOINT_EXEMPTIONS = {
    "beyondmimic": (
        "The upstream release exports one policy per reference trajectory and "
        "does not publish a redistributable general-purpose checkpoint."
    ),
    "motion_clip": "Internal evaluator package rather than a method Pipeline.",
    "motionrepair": "Deterministic, training-free source Pipeline.",
    "prompthmr": (
        "The upstream license does not permit a standalone checkpoint release."
    ),
}


def _spec(
    repo: str,
    pipeline_class: str,
    bundle_class: str,
    tasks: tuple[str, ...],
    target_repo: str | None = None,
) -> CheckpointSpec:
    return CheckpointSpec(
        source_repo=repo,
        target_repo=target_repo or repo,
        pipeline_class=pipeline_class,
        bundle_class=bundle_class,
        tasks=tasks,
    )


CHECKPOINT_SPECS = (
    _spec(
        "ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2",
        "motius.pipelines.any2track.Any2TrackPipeline",
        "motius.models.any2track.Any2TrackBundle",
        ("motion_tracking",),
    ),
    _spec(
        "ZeyuLing/motius-ardy-330-horizon40",
        "motius.pipelines.ardy.ARDYPipeline",
        "motius.models.ardy.ARDYBundle",
        (
            "text_to_motion",
            "sequential_text_to_motion",
            "kinematic_motion_control",
        ),
    ),
    _spec(
        "ZeyuLing/Motius-Bailando-AISTPP",
        "motius.pipelines.bailando.BailandoPipeline",
        "motius.models.bailando.BailandoBundle",
        ("music_to_dance",),
    ),
    _spec(
        "ZeyuLing/motius-condmdi-humanml3d",
        "motius.pipelines.condmdi.CondMDIPipeline",
        "motius.models.condmdi.CondMDIBundle",
        ("text_to_motion", "kinematic_motion_control"),
    ),
    _spec(
        "ZeyuLing/motius-dart-humanml3d",
        "motius.pipelines.dart.DARTPipeline",
        "motius.models.dart.DARTBundle",
        ("text_to_motion",),
    ),
    _spec(
        "ZeyuLing/Motius-EDGE-AISTPP",
        "motius.pipelines.edge.EDGEPipeline",
        "motius.models.edge.EDGEBundle",
        ("music_to_dance",),
    ),
    _spec(
        "ZeyuLing/hftrainer-flowmdm-humanml3d",
        "motius.pipelines.flowmdm.FlowMDMPipeline",
        "motius.models.flowmdm.FlowMDMBundle",
        (
            "text_to_motion",
            "temporal_motion_completion",
            "sequential_text_to_motion",
        ),
        "ZeyuLing/Motius-FlowMDM-HumanML3D",
    ),
    _spec(
        "ZeyuLing/motius-flowmdm-babel",
        "motius.pipelines.flowmdm.FlowMDMPipeline",
        "motius.models.flowmdm.FlowMDMBundle",
        ("sequential_text_to_motion",),
    ),
    _spec(
        "ZeyuLing/Motius-GEM-SMPL",
        "motius.pipelines.gem_smpl.GemSmplPipeline",
        "motius.models.gem_smpl.GemSmplBundle",
        ("monocular_motion_capture",),
    ),
    _spec(
        "ZeyuLing/Motius-GEM-X",
        "motius.pipelines.gem_x.GemXPipeline",
        "motius.models.gem_x.GemXBundle",
        ("monocular_motion_capture",),
    ),
    _spec(
        "ZeyuLing/Motius-GVHMR",
        "motius.pipelines.gvhmr.GVHMRPipeline",
        "motius.models.gvhmr.GVHMRBundle",
        ("monocular_motion_capture",),
    ),
    _spec(
        "ZeyuLing/hftrainer-hymotion-t2m-1.0",
        "motius.pipelines.hymotion_t2m.HyMotionT2MPipeline",
        "motius.models.hymotion_t2m.HyMotionT2MBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-HYMotion-T2M-1.0",
    ),
    _spec(
        "ZeyuLing/hftrainer-hymotion-t2m-1.0-lite",
        "motius.pipelines.hymotion_t2m.HyMotionT2MPipeline",
        "motius.models.hymotion_t2m.HyMotionT2MBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-HYMotion-T2M-1.0-Lite",
    ),
    _spec(
        "ZeyuLing/Motius-HYMotion-G1",
        "motius.pipelines.hymotion_t2m.HyMotionT2MPipeline",
        "motius.models.hymotion_t2m.HyMotionT2MBundle",
        ("text_to_motion",),
    ),
    _spec(
        "ZeyuLing/Motius-HumanoidGPT-G1",
        "motius.pipelines.humanoid_gpt.HumanoidGPTPipeline",
        "motius.models.humanoid_gpt.HumanoidGPTBundle",
        ("motion_tracking",),
    ),
    _spec(
        "ZeyuLing/motius-intergen-interhuman",
        "motius.pipelines.intergen.InterGenPipeline",
        "motius.models.intergen.InterGenBundle",
        ("text_to_multi_person_motion",),
    ),
    _spec(
        "ZeyuLing/motius-intermask-interhuman",
        "motius.pipelines.intermask.InterMaskPipeline",
        "motius.models.intermask.InterMaskBundle",
        ("text_to_multi_person_motion",),
    ),
    _spec(
        "ZeyuLing/hftrainer-kimodo-soma-rp",
        "motius.pipelines.kimodo.KIMODOPipeline",
        "motius.models.kimodo.KIMODOBundle",
        (
            "text_to_motion",
            "temporal_motion_completion",
            "sequential_text_to_motion",
            "kinematic_motion_control",
        ),
        "ZeyuLing/Motius-KIMODO-SOMA-RP",
    ),
    _spec(
        "ZeyuLing/hftrainer-kimodo-g1-rp",
        "motius.pipelines.kimodo.KIMODOPipeline",
        "motius.models.kimodo.KIMODOBundle",
        ("text_to_motion", "kinematic_motion_control"),
        "ZeyuLing/Motius-KIMODO-G1-RP",
    ),
    _spec(
        "ZeyuLing/hftrainer-kimodo-g1-seed",
        "motius.pipelines.kimodo.KIMODOPipeline",
        "motius.models.kimodo.KIMODOBundle",
        ("text_to_motion", "kinematic_motion_control"),
        "ZeyuLing/Motius-KIMODO-G1-SEED",
    ),
    _spec(
        "ZeyuLing/hftrainer-kimodo-smplx-rp",
        "motius.pipelines.kimodo.KIMODOPipeline",
        "motius.models.kimodo.KIMODOBundle",
        ("text_to_motion", "kinematic_motion_control"),
        "ZeyuLing/Motius-KIMODO-SMPLX-RP",
    ),
    _spec(
        "ZeyuLing/motius-maskcontrol-humanml3d",
        "motius.pipelines.maskcontrol.MaskControlPipeline",
        "motius.models.maskcontrol.MaskControlBundle",
        (
            "text_to_motion",
            "temporal_motion_completion",
            "part_level_motion_control",
            "sequential_text_to_motion",
        ),
    ),
    _spec(
        "ZeyuLing/hftrainer-mdm-humanml3d",
        "motius.pipelines.mdm.MDMPipeline",
        "motius.models.mdm.MDMBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-MDM-HumanML3D",
    ),
    _spec(
        "ZeyuLing/hftrainer-mld-humanml3d",
        "motius.pipelines.mld.MLDPipeline",
        "motius.models.mld.MLDBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-MLD-HumanML3D",
    ),
    _spec(
        "ZeyuLing/hftrainer-mogents-humanml3d",
        "motius.pipelines.mogents.MoGenTSPipeline",
        "motius.models.mogents.MoGenTSBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-MoGenTS-HumanML3D",
    ),
    _spec(
        "ZeyuLing/hftrainer-momask-humanml3d",
        "motius.pipelines.momask.MoMaskPipeline",
        "motius.models.momask.MoMaskBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-MoMask-HumanML3D",
    ),
    _spec(
        "ZeyuLing/motius-motionbricks-g1",
        "motius.pipelines.motionbricks.MotionBricksPipeline",
        "motius.models.motionbricks.MotionBricksBundle",
        ("g1_realtime_navigation", "g1_qpos_generation"),
    ),
    _spec(
        "ZeyuLing/Motius-MotionCanvas-0.46B",
        "motius.pipelines.motioncanvas.MotionCanvasPipeline",
        "motius.models.motioncanvas.MotionCanvasBundle",
        (
            "temporal_motion_completion",
            "kinematic_motion_control",
            "motion_editing",
        ),
    ),
    _spec(
        "ZeyuLing/motius-motionclr-humanml3d",
        "motius.pipelines.motionclr.MotionCLRPipeline",
        "motius.models.motionclr.MotionCLRBundle",
        ("text_to_motion",),
    ),
    _spec(
        "ZeyuLing/motius-omnicontrol-humanml3d",
        "motius.pipelines.omnicontrol.OmniControlPipeline",
        "motius.models.omnicontrol.OmniControlBundle",
        (
            "text_to_motion",
            "temporal_motion_completion",
            "kinematic_motion_control",
        ),
    ),
    _spec(
        "ZeyuLing/motius-projflow-humanml3d",
        "motius.pipelines.projflow.ProjFlowPipeline",
        "motius.models.projflow.ProjFlowBundle",
        (
            "temporal_motion_completion",
            "kinematic_motion_control",
            "part_level_motion_control",
        ),
    ),
    _spec(
        "ZeyuLing/Motius-MotionGPT-HumanML3D",
        "motius.pipelines.motiongpt.MotionGPTPipeline",
        "motius.models.motiongpt.MotionGPTBundle",
        ("text_to_motion", "motion_to_text"),
    ),
    _spec(
        "ZeyuLing/Motius-MotionGPT3-HumanML3D",
        "motius.pipelines.motiongpt3.MotionGPT3Pipeline",
        "motius.models.motiongpt3.MotionGPT3Bundle",
        ("text_to_motion", "motion_to_text"),
    ),
    _spec(
        "ZeyuLing/hftrainer-motionlcm-humanml3d",
        "motius.pipelines.motionlcm.MotionLCMPipeline",
        "motius.models.motionlcm.MotionLCMBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-MotionLCM-HumanML3D",
    ),
    _spec(
        "ZeyuLing/hftrainer-gotozero-7b-train-humanml272",
        "motius.pipelines.motionmillion.MotionMillionPipeline",
        "motius.models.motionmillion.MotionMillionBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-MotionMillion-7B-HumanML272",
    ),
    _spec(
        "ZeyuLing/hftrainer-gotozero-3b-train-humanml272",
        "motius.pipelines.motionmillion.MotionMillionPipeline",
        "motius.models.motionmillion.MotionMillionBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-MotionMillion-3B-HumanML272",
    ),
    _spec(
        "ZeyuLing/hftrainer-motionstreamer-humanml272",
        "motius.pipelines.motionstreamer.MotionStreamerPipeline",
        "motius.models.motionstreamer.MotionStreamerBundle",
        (
            "text_to_motion",
            "temporal_motion_completion",
            "sequential_text_to_motion",
        ),
        "ZeyuLing/Motius-MotionStreamer-HumanML272",
    ),
    _spec(
        "ZeyuLing/motius-prism-1.0-humanml3d",
        "motius.pipelines.prism.PRISMPipeline",
        "motius.models.prism.PRISMBundle",
        ("text_to_motion",),
    ),
    _spec(
        "ZeyuLing/Motius-ProtoMotions-G1-BONES-SEED",
        "motius.pipelines.protomotions.ProtoMotionsPipeline",
        "motius.models.protomotions.ProtoMotionsBundle",
        ("motion_tracking",),
    ),
    _spec(
        "ZeyuLing/motius-prism-kt-humanml3d",
        "motius.pipelines.prism.PRISMPipeline",
        "motius.models.prism.PRISMBundle",
        (
            "text_to_motion",
            "temporal_motion_completion",
            "sequential_text_to_motion",
        ),
    ),
    _spec(
        "ZeyuLing/hftrainer-t2mgpt-humanml3d",
        "motius.pipelines.t2mgpt.T2MGPTPipeline",
        "motius.models.t2mgpt.T2MGPTBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-T2M-GPT-HumanML3D",
    ),
    _spec(
        "ZeyuLing/Motius-TM2D-HumanML3D-AISTPP",
        "motius.pipelines.tm2d.TM2DPipeline",
        "motius.models.tm2d.TM2DBundle",
        ("text_to_motion", "music_to_dance"),
    ),
    _spec(
        "ZeyuLing/Motius-SONIC-G1",
        "motius.pipelines.sonic.SONICPipeline",
        "motius.models.sonic.SONICBundle",
        ("motion_tracking",),
    ),
    _spec(
        "ZeyuLing/Motius-TM2T-HumanML3D",
        "motius.pipelines.tm2t.TM2TPipeline",
        "motius.models.tm2t.TM2TBundle",
        ("motion_to_text",),
    ),
    _spec(
        "ZeyuLing/Motius-UniMuMo",
        "motius.pipelines.unimumo.UniMuMoPipeline",
        "motius.models.unimumo.UniMuMoBundle",
        (
            "text_to_music_motion",
            "text_to_motion",
            "text_to_music",
            "music_to_dance",
            "dance_to_music",
            "music_to_text",
            "motion_to_text",
        ),
    ),
    _spec(
        "ZeyuLing/Motius-VerMo-HumanML3D",
        "motius.pipelines.vermo.VermoPipeline",
        "motius.models.vermo.VermoBundle",
        ("motion_to_text",),
    ),
    _spec(
        "ZeyuLing/hftrainer-vimogen-1.3b-humanml3d",
        "motius.pipelines.vimogen.ViMoGenPipeline",
        "motius.models.vimogen.ViMoGenBundle",
        ("text_to_motion",),
        "ZeyuLing/Motius-ViMoGen-1.3B-HumanML3D",
    ),
)

SPEC_BY_SOURCE = {spec.source_repo: spec for spec in CHECKPOINT_SPECS}
SPEC_BY_TARGET = {spec.target_repo: spec for spec in CHECKPOINT_SPECS}
