"""Official MotionCLR HumanML3D-263 inference configuration."""

custom_imports = dict(
    imports=["motius.models.motionclr", "motius.pipelines.motionclr"],
    allow_failed_imports=False,
)

source_repository = "https://github.com/IDEA-Research/MotionCLR"
source_revision = "a6f44a791940682fe335c82f1b436bae05a1cebb"
pretrained_model_name_or_path = "EvanTHU/MotionCLR"

official_files = dict(
    checkpoint_sha256="5852e139bbe45f5ca45b67b72cc54ab02b7da7ae18b42f27ea630a715c5c2b5f",
    mean_sha256="0bdb5ba69a3a9e34d71990db15bc535ebc024c8d95ddb5574196f96058faa7d3",
    std_sha256="487855309295f986d08e96d65e415fb6b2a94211ac34ce444007e84cba8f33bb",
)

network = dict(
    input_feats=263,
    base_dim=512,
    dim_mults=[2, 2, 2, 2],
    adagn=True,
    zero=True,
    dropout=0.1,
    no_eff=True,
    time_dim=512,
    latent_dim=512,
    cond_mask_prob=0.1,
    clip_dim=512,
    clip_version="ViT-B/32",
    text_latent_dim=256,
    text_ff_size=2048,
    text_num_heads=4,
    activation="gelu",
    num_text_layers=4,
    self_attention=True,
    vis_attn=False,
)

bundle_kwargs = dict(
    use_ema=True,
    torch_dtype="fp16",
    diffuser_name="dpmsolver",
    num_inference_steps=10,
    guidance_scale=2.5,
)

pipeline = dict(
    type="MotionCLRPipeline",
    diffuser_name="dpmsolver",
    num_inference_steps=10,
    guidance_scale=2.5,
)

motion = dict(representation="humanml3d_263", fps=20, max_frames=196)
