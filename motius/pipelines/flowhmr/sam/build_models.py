"""Construct the native SAM bundle for the video feature pipeline."""
from motius.models.vision.sam3d_body.bundle import SAM3DBodyBundle


def load_sam_3d_body(checkpoint_path="", device="cuda", mhr_path="", config_path="", *, precision="source"):
    bundle = SAM3DBodyBundle(checkpoint=checkpoint_path, mhr_path=mhr_path,
                            config_path=config_path or None, precision=precision).to(device).eval()
    return bundle.model, bundle.config
