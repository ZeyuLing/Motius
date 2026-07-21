"""Official EDGE AIST++ music-to-dance inference and evaluation."""

custom_imports = dict(
    imports=[
        "motius.models.edge",
        "motius.pipelines.edge",
        "motius.evaluation.music_to_dance",
    ],
    allow_failed_imports=False,
)

source_repository = "https://github.com/Stanford-TML/EDGE"
source_revision = "17c3428669ed6733edd9d8c66f7dc62060b8e46d"
pretrained_model_name_or_path = "ZeyuLing/Motius-EDGE-AISTPP"

model = dict(type="EDGEBundle")
pipeline = dict(type="EDGEPipeline")
evaluator = dict(type="AISTPPMusicDanceEvaluator", max_frames=1200, physical=True)

data = dict(
    dataset="AIST++",
    split="crossmodal_test",
    model_motion_representation="EDGE-151 contacts/root/SMPL24 local rot6d",
    public_motion_representation="AIST++ SMPL-24 joint positions",
    motion_fps=30.0,
    music_representation="Jukebox layer 66",
    music_feature_dim=4800,
    music_feature_fps=30.0,
    window_frames=150,
    overlap_frames=75,
)
