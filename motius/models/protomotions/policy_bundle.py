"""Neural roots for native ProtoMotions PPO/AMP training and inference."""

from pathlib import Path
import json
import torch
from motius.models.base_model_bundle import ModelBundle
from motius.models.onnx_policy import resolve_pretrained_directory
from motius.registry import MODEL_BUNDLES
from motius.utils.torch_checkpoint import load_torch_checkpoint
from .policy import GaussianActor, ObservationMLP


SOURCE_NEURAL_PREFIXES = {
    "_actor.mu.norm.running_obs_norm.": "actor.mean.norm.",
    "_actor.mu.mlp.": "actor.mean.mlp.",
    "_actor.logstd": "actor.logstd",
    "_critic.norm.running_obs_norm.": "critic.norm.",
    "_critic.mlp.": "critic.mlp.",
    "_discriminator.models.0.norm.running_obs_norm.": "discriminator.norm.",
    "_discriminator.models.0.mlp.": "discriminator.mlp.",
    "_disc_critic.models.0.norm.running_obs_norm.": "discriminator_critic.norm.",
    "_disc_critic.models.0.mlp.": "discriminator_critic.mlp.",
}


@MODEL_BUNDLES.register_module()
class ProtoMotionsPolicyBundle(ModelBundle):
    METHOD_NAME = "ProtoMotions"

    def __init__(self, network_options, *, checkpoint=None, device="cpu"):
        super().__init__()
        self.network_options = network_options
        self.artifact_root = (
            Path(checkpoint).expanduser().resolve().parent
            if checkpoint is not None
            else None
        )
        self.actor = GaussianActor(**network_options["actor"])
        self._save_ckpt_modules = ["actor"]
        for name in ["critic", "discriminator", "discriminator_critic"]:
            options = network_options.get(name)
            if options is not None:
                setattr(self, name, ObservationMLP(**options))
                self._save_ckpt_modules.append(name)
        self._trainable_modules = list(self._save_ckpt_modules)
        if checkpoint is not None:
            state = load_torch_checkpoint(checkpoint, weights_only=False)
            if "model" in state:
                state = self.convert_original_state(state["model"])
            self.load_state_dict(state, strict=True)
        self.to(device)

    @staticmethod
    def convert_original_state(state):
        prefixes = SOURCE_NEURAL_PREFIXES
        converted = {}
        for key, value in state.items():
            for source, target in prefixes.items():
                if key.startswith(source):
                    destination = target + key[len(source) :]
                    if destination in converted:
                        raise ValueError("Duplicate neural state: " + destination)
                    converted[destination] = value
                    break
            else:
                raise ValueError("Unsupported original neural state: " + key)
        return converted

    @property
    def device(self):
        return next(self.actor.parameters()).device

    def predict_policy(self, observations):
        return self.actor(observations)

    def predict_value(self, observations):
        return self.critic(observations)

    def predict_discriminator(self, observations, *, return_normalized=False):
        return self.discriminator(observations, return_normalized=return_normalized)

    def predict_discriminator_value(self, observations):
        return self.discriminator_critic(observations)

    def forward(self, observations):
        output = self.predict_policy(observations)
        if hasattr(self, "critic"):
            output["value"] = self.predict_value(observations)
        if hasattr(self, "discriminator"):
            output["disc_logits"] = self.predict_discriminator(observations)
        if hasattr(self, "discriminator_critic"):
            output["disc_value"] = self.predict_discriminator_value(observations)
        return output

    @classmethod
    def from_original_checkpoint(cls, checkpoint, *, configuration=None, device="cpu"):
        from motius.configuration.protomotions import original_neural_options

        checkpoint = Path(checkpoint).expanduser().resolve()
        state = load_torch_checkpoint(checkpoint, weights_only=False)
        configuration = configuration or checkpoint.parent / "resolved_configs.yaml"
        options = original_neural_options(state["model"], configuration)
        return cls(options, checkpoint=checkpoint, device=device)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        hub = {
            name: kwargs.pop(name)
            for name in ["revision", "cache_dir", "token", "local_files_only"]
            if name in kwargs
        }
        root = resolve_pretrained_directory(pretrained_model_name_or_path, **hub)
        config = json.loads((root / "protomotions_neural.json").read_text())
        config.update(kwargs)
        return cls(**config, checkpoint=root / "protomotions_networks.pt").eval()

    def save_pretrained(self, save_directory, **kwargs):
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        state = {
            name + "." + key: value.detach().cpu()
            for name in self._save_ckpt_modules
            for key, value in getattr(
                getattr(self, name), "module", getattr(self, name)
            )
            .state_dict()
            .items()
        }
        torch.save(state, root / "protomotions_networks.pt")
        (root / "protomotions_neural.json").write_text(
            json.dumps(dict(network_options=self.network_options), indent=2) + "\n"
        )
        (root / "model_index.json").write_text(
            json.dumps(
                dict(
                    model_type="protomotions",
                    artifact_format="motius-protomotions-neural-v2",
                    bundle_class="motius.models.protomotions.ProtoMotionsPolicyBundle",
                    pipeline_class="motius.pipelines.protomotions.ProtoMotionsPipeline",
                    tasks=["motion_tracking"],
                    required_files=[
                        "protomotions_neural.json",
                        "protomotions_networks.pt",
                    ],
                ),
                indent=2,
            )
            + "\n"
        )
        return str(root)
