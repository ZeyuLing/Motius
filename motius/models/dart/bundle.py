"""DART neural denoiser, motion VAE and frozen CLIP encoder."""

from pathlib import Path
import json
import shutil
import torch
from motius.configuration.dart import read_arguments
from motius.models.base_model_bundle import ModelBundle
from motius.models.clip_artifact import load_openai_clip
from motius.registry import MODEL_BUNDLES
from .denoiser import DenoiserMLP, DenoiserTransformer
from .vae import AutoMldVae

_DEFAULT_DENOISER = "mld_denoiser/smplh_hml3d_2_8_4/checkpoint_300000.pt"


def unwrap(module):
    return module.module if hasattr(module, "module") else module


@MODEL_BUNDLES.register_module()
class DARTBundle(ModelBundle):
    motion_dim = 276

    def __init__(
        self,
        artifact_dir,
        denoiser_checkpoint=_DEFAULT_DENOISER,
        clip_weights_path=None,
        device="cuda",
        **inference_options,
    ):
        super().__init__()
        self.artifact_dir = Path(artifact_dir).resolve()
        self.denoiser_checkpoint = denoiser_checkpoint
        self.inference_options = inference_options
        checkpoint = self.artifact_dir / denoiser_checkpoint
        arguments = read_arguments(checkpoint.parent / "args.yaml")
        self.denoiser_args = arguments.denoiser_args
        model_class = (
            DenoiserMLP
            if self.denoiser_args.model_type == "mlp"
            else DenoiserTransformer
        )
        self.denoiser_model = model_class(**vars(self.denoiser_args.model_args))
        self.denoiser_model.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=False)[
                "model_state_dict"
            ],
            strict=True,
        )
        vae_checkpoint = self.artifact_dir / self.denoiser_args.mvae_path
        self.vae_args = read_arguments(vae_checkpoint.parent / "args.yaml")
        self.vae_model = AutoMldVae(**vars(self.vae_args.model_args))
        state = torch.load(vae_checkpoint, map_location="cpu", weights_only=False)[
            "model_state_dict"
        ]
        state.setdefault("latent_mean", torch.tensor(0))
        state.setdefault("latent_std", torch.tensor(1))
        self.vae_model.load_state_dict(state, strict=True)
        # Legacy source buffers are integer scalars until the released float
        # statistics replace them. Preserve their actual dtype and values.
        self.vae_model.latent_mean = state["latent_mean"]
        self.vae_model.latent_std = state["latent_std"]
        self.clip_weights_path = Path(
            clip_weights_path or self.artifact_dir / "clip.safetensors"
        )
        self.clip_model = None
        resolved = torch.device(
            device if device != "cuda" or torch.cuda.is_available() else "cpu"
        )
        self.requires_grad_(False).to(resolved).eval()

    @property
    def device(self):
        return next(self.denoiser_model.parameters()).device

    def load_text_encoder(self):
        if self.clip_model is None:
            self.clip_model = load_openai_clip(
                self.clip_weights_path, device=self.device
            )
            import clip

            clip.model.convert_weights(self.clip_model)
            self.clip_model.requires_grad_(False)
        return self.clip_model

    def predict_latent(self, x_t, timesteps, y):
        return self.denoiser_model(x_t, timesteps, y)

    def encode_motion(self, future_motion, history_motion, scale_latent=False):
        return unwrap(self.vae_model).encode(
            future_motion, history_motion, scale_latent
        )

    def decode_motion(self, z, history_motion, nfuture, scale_latent=False):
        return self.vae_model(
            z=z,
            history_motion=history_motion,
            nfuture=nfuture,
            scale_latent=scale_latent,
        )

    def reconstruct_motion(self, future_motion, history_motion, scale_latent=False):
        return self.vae_model(
            future_motion=future_motion,
            history_motion=history_motion,
            scale_latent=scale_latent,
        )

    def forward(self, x_t, timesteps, y):
        return self.predict_latent(x_t, timesteps, y)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        path = Path(pretrained_model_name_or_path).expanduser()
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(snapshot_download(pretrained_model_name_or_path))
        index = path / "model_index.json"
        config = json.loads(index.read_text()) if index.exists() else {}
        from motius.configuration.dart import inference_options

        config = inference_options(config)
        config.update(kwargs)
        return cls(artifact_dir=path, **config)

    def save_pretrained(self, save_directory, **kwargs):
        use_ema = kwargs.pop("use_ema", True)
        target = Path(save_directory)
        target.mkdir(parents=True, exist_ok=True)
        # Export only inference dependencies, never whole experiment/data trees.
        paths = [
            self.denoiser_checkpoint,
            self.denoiser_args.mvae_path,
            str(Path(self.denoiser_checkpoint).parent / "args.yaml"),
            str(Path(self.denoiser_args.mvae_path).parent / "args.yaml"),
            self.vae_args.data_args.cfg_path,
            self.inference_options.get("seed_sequence", "data/stand_20fps.pkl"),
        ]
        data = self.artifact_dir / self.vae_args.data_args.data_dir
        paths.extend(
            str(f.relative_to(self.artifact_dir)) for f in data.glob("mean_std*.pkl")
        )
        for relative in paths:
            source = self.artifact_dir / relative
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if source.resolve() != destination.resolve():
                shutil.copy2(source, destination)
        denoiser = (
            getattr(self, "denoiser_model_ema", self.denoiser_model)
            if use_ema
            else self.denoiser_model
        )
        vae = (
            getattr(self, "vae_model_ema", self.vae_model)
            if use_ema
            else self.vae_model
        )
        torch.save(
            {"model_state_dict": unwrap(denoiser).state_dict()},
            target / self.denoiser_checkpoint,
        )
        torch.save(
            {"model_state_dict": unwrap(vae).state_dict()},
            target / self.denoiser_args.mvae_path,
        )
        if self.clip_model is not None:
            from safetensors.torch import save_file

            save_file(
                {
                    k: v.detach().cpu().contiguous()
                    for k, v in self.clip_model.state_dict().items()
                },
                str(target / "clip.safetensors"),
            )
        elif (
            self.clip_weights_path.resolve() != (target / "clip.safetensors").resolve()
        ):
            shutil.copy2(self.clip_weights_path, target / "clip.safetensors")
        from motius.configuration.dart import artifact_metadata

        config = artifact_metadata(self.inference_options, self.denoiser_checkpoint)
        (target / "model_index.json").write_text(json.dumps(config, indent=2) + "\n")
        return str(target)
