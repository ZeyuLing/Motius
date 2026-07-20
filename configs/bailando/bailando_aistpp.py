"""Official Bailando AIST++ music-to-dance inference and evaluation."""

custom_imports = dict(
    imports=[
        "motius.models.bailando",
        "motius.pipelines.bailando",
        "motius.evaluation.music_to_dance",
    ],
    allow_failed_imports=False,
)

source_repository = "https://github.com/lisiyao21/Bailando"
source_revision = "cc90b98bff81c9709570db413c9610c2562e27ca"
pretrained_model_name_or_path = "ZeyuLing/Motius-Bailando-AISTPP"

model = dict(type="BailandoBundle")
pipeline = dict(type="BailandoPipeline")
evaluator = dict(
    type="AISTPPMusicDanceEvaluator",
    max_frames=1200,
    physical=True,
)

data = dict(
    dataset="AIST++",
    split="crossmodal_test+crossmodal_val",
    motion_representation="AIST++ SMPL-24 joint positions",
    motion_fps=60.0,
    music_feature_dim=438,
    music_feature_fps=7.5,
    official_eval_music_fps=60.0,
    initial_motion_protocol="first_gt_vq_token",
)
