"""Released SAM3D RGB scaling, normalization and fixed-width crop preparation."""


def prepare_sam3d_images(model, batch, *, unit_rgb=False):
    inputs = batch["img"]
    if inputs.ndim == 5:
        inputs = inputs.view(inputs.shape[0] * inputs.shape[1], *inputs.shape[2:])
    image_mean, image_std = model.image_mean, model.image_std
    if unit_rgb:
        # GPU preprocessing has already established uint8 / 255 range. Avoid
        # repeated device-to-host scalar synchronizations for a known invariant.
        if max(model.cfg.MODEL.IMAGE_MEAN) > 1:
            inputs = inputs * 255.0
    elif inputs.max() > 1 and image_mean.max() <= 1.0:
        inputs = inputs / 255.0
    elif inputs.max() <= 1.0 and image_mean.max() > 1:
        inputs = inputs * 255.0
    prepared = (inputs - image_mean) / image_std
    backbone = model.cfg.MODEL.BACKBONE.TYPE
    if backbone in ["vit_hmr", "vit", "vit_b", "vit_l", "vit_hmr_512_384"]:
        if backbone in ["vit_hmr", "vit"]:
            prepared = prepared[:, :, :, 32:-32]
        elif backbone == "vit_hmr_512_384":
            prepared = prepared[:, :, :, 64:-64]
        else:
            # Preserve the original forward's unsupported crop-width branch.
            raise ValueError("Unsupported SAM3D crop-width backbone: " + backbone)
    batch["network_images"] = prepared
    return batch
