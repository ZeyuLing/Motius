"""Build inference composition around the Bundle's actual neural networks."""

from copy import deepcopy
from pathlib import Path
from motius.configuration.ardy.loading import instantiate_from_dict
from motius.motion.representation.ardy.token_codec import FSQVAETransformer
from .two_stage import AutoLatentTwostageDenoiser
from .runtime import Ardy


def compose_runtime(bundle, text_encoder=None):
    config = deepcopy(bundle._model_config)
    denoiser_config = config.pop("denoiser")
    autoencoder_config = config.pop("autoencoder")
    denoiser_config.pop("_target_")
    autoencoder_config.pop("_target_")
    denoiser_config["motion_rep"] = instantiate_from_dict(denoiser_config["motion_rep"])
    autoencoder_config["motion_rep"] = instantiate_from_dict(
        autoencoder_config["motion_rep"]
    )
    autoencoder_config["statistics_root"] = (
        Path(autoencoder_config["ckpt_path"]).parent / "stats"
    )
    autoencoder_config["ckpt_path"] = None
    denoiser_config["ckpt_path"] = None
    denoiser = AutoLatentTwostageDenoiser(**denoiser_config, network=bundle.denoiser)
    autoencoder = FSQVAETransformer(**autoencoder_config, network=bundle.autoencoder)
    config.pop("_target_")
    config.pop("text_encoder", None)
    config.pop("device", None)
    return Ardy(
        denoiser=denoiser,
        autoencoder=autoencoder,
        text_encoder=text_encoder,
        device=bundle.device,
        **config,
    ).eval()
