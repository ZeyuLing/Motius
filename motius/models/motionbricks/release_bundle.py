"""Native five-root PyTorch bundle for MotionBricks SOMA/G1 released weights."""

from pathlib import Path
import hashlib
import json

import numpy as np
from safetensors.torch import load_file, save_file

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES
from motius.configuration.motionbricks_release import (
    representation,
    soma_options,
    g1_options,
    UPSTREAM_REVISION,
)
from .encdec import Encoder
from .encdec_double_cond import DoubleCondDecoder
from .release_decoder import SOMADecoder
from .release_codebook import QuantizeEMAResetMultiHead
from .release_g1 import G1PoseNetwork, G1RootNetwork
from .release_soma import SOMAPoseNetwork, SOMARootNetwork


@MODEL_BUNDLES.register_module()
class MotionBricksReleaseBundle(ModelBundle):
    """Own only learned roots. Geometry and inference policy belong to the pipeline."""

    def __init__(self, artifact_root, *, model=None, device="cpu", verify_source=True):
        super().__init__()
        self.artifact_root = Path(artifact_root).expanduser().resolve()
        config_path = self.artifact_root / "motionbricks_release.json"
        native = config_path.is_file()
        config = json.loads(config_path.read_text()) if native else {}
        self.model = model or config.get("rig")
        if self.model not in ("g1", "soma"):
            raise ValueError(
                'Specify model="g1" or model="soma" for official source weights'
            )
        if native and self.model != config["rig"]:
            raise ValueError("Requested rig differs from artifact rig")
        if native:
            with np.load(
                self.artifact_root / "normalization.npz", allow_pickle=False
            ) as data:
                self.mean, self.std = data["mean"].copy(), data["std"].copy()
        else:
            metadata = (
                Path(__file__).resolve().parents[2]
                / "configuration/motionbricks_assets/release_checksums.json"
            )
            release = json.loads(metadata.read_text())[self.model]
            source = self.artifact_root
            if not (source / release["files"][0]["path"]).is_file():
                source = source / release["root_directory"]
            sources = {}
            for entry in release["files"]:
                path = source / entry["path"]
                if verify_source:
                    with path.open("rb") as stream:
                        checksum = hashlib.sha256()
                        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                            checksum.update(chunk)
                        digest = checksum.hexdigest()
                    if (
                        path.stat().st_size != entry["bytes"]
                        or digest != entry["sha256"]
                    ):
                        raise ValueError(
                            f"Official checkpoint checksum mismatch: {path}"
                        )
                sources[entry["path"]] = (
                    load_file(path)
                    if path.suffix == ".safetensors"
                    else np.load(path, allow_pickle=False)
                )
            self.mean = next(v for k, v in sources.items() if k.endswith("mean.npy"))
            self.std = next(v for k, v in sources.items() if k.endswith("std.npy"))
        rep = representation(self.model, self.mean, self.std)
        self.options = (soma_options if self.model == "soma" else g1_options)(rep)
        cfg = self.options
        soma = self.model == "soma"
        code_dim, heads, codes = (160, 5, 1048576) if soma else (256, 8, 100000000)
        self.pose = (SOMAPoseNetwork if soma else G1PoseNetwork)(cfg["pose"])
        self.root = (SOMARootNetwork if soma else G1RootNetwork)(cfg["root"])
        self.encoder = Encoder(
            cfg["encoder_dim"],
            code_dim,
            2,
            2,
            512,
            4,
            3,
            activation="relu",
            norm="None",
        )
        self.decoder = (SOMADecoder if soma else DoubleCondDecoder)(
            cfg["decoder_dim"],
            code_dim,
            2,
            512,
            4,
            3,
            activation="relu",
            norm="None",
            target_cond_dim=cfg["target_dim"],
            external_cond_dim=cfg["external_dim"],
            cond_fusion_last_layer=False,
        )
        self.quantizer = QuantizeEMAResetMultiHead(codes, code_dim, heads)
        self._save_ckpt_modules = ["pose", "root", "encoder", "decoder", "quantizer"]
        if native:
            self.load_state_dict(
                load_file(self.artifact_root / "model.safetensors"), strict=True
            )
        else:
            for stage in ("pose", "root", "vqvae"):
                matches = [
                    v
                    for k, v in sources.items()
                    if k.endswith(".safetensors")
                    and (
                        f"_{stage}/" in k
                        or f"-{stage}-" in k
                        or (stage == "vqvae" and k.startswith("vqvae-"))
                    )
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {stage} checkpoint")
                state = matches[0]
                if stage == "vqvae":
                    consumed = set()
                    for name in ("encoder", "decoder", "quantizer"):
                        prefix = "pose_net." + name + "."
                        weights = {
                            k[len(prefix) :]: v
                            for k, v in state.items()
                            if k.startswith(prefix)
                        }
                        consumed.update(prefix + k for k in weights)
                        getattr(self, name).load_state_dict(weights, strict=True)
                else:
                    prefix = "mmm_net." if soma else "backbone_net."
                    weights = {
                        k[len(prefix) :]: v
                        for k, v in state.items()
                        if k.startswith(prefix)
                    }
                    consumed = {prefix + k for k in weights}
                    getattr(self, stage).load_state_dict(weights, strict=True)
                if consumed != set(state):
                    raise ValueError(
                        f"Unconsumed checkpoint keys for {stage}: {set(state) - consumed}"
                    )
        self.to(device).eval()

    @property
    def device(self):
        return next(self.pose.parameters()).device

    def predict_pose(self, *args, **kwargs):
        return self.pose(*args, **kwargs)

    def encode_root_conditions(self, **kwargs):
        return self.root.encode_conditions(**kwargs)

    def decode_root_motion(self, context, chosen_token):
        return self.root.decode_motion(context, chosen_token)

    def encode_features(self, features):
        return self.encoder(features.transpose(1, 2))

    def decode_tokens(self, indices, *, use_overall_indices=False, **kwargs):
        features = (
            self.quantizer.dequantize(indices, use_overall_indices)
            .transpose(1, 2)
            .contiguous()
        )
        return self.decoder(features, **kwargs).transpose(1, 2)

    @classmethod
    def from_pretrained(
        cls,
        source,
        *,
        revision=None,
        cache_dir=None,
        token=None,
        local_files_only=False,
        **kwargs,
    ):
        path = Path(source).expanduser()
        if not path.is_dir():
            from huggingface_hub import snapshot_download

            path = Path(
                snapshot_download(
                    str(source),
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                )
            )
        return cls(path, **kwargs)

    def save_pretrained(self, save_directory, **kwargs):
        destination = Path(save_directory)
        destination.mkdir(parents=True, exist_ok=True)
        save_file(
            {k: v.detach().cpu().contiguous() for k, v in self.state_dict().items()},
            destination / "model.safetensors",
        )
        np.savez(destination / "normalization.npz", mean=self.mean, std=self.std)
        (destination / "motionbricks_release.json").write_text(
            json.dumps(
                dict(
                    rig=self.model,
                    source_revision=UPSTREAM_REVISION,
                    format="motius-motionbricks-native-v3",
                    runtime_minimum_motius_version="0.3.23",
                    runtime_extra="motionbricks-runtime",
                    runtime_capabilities=[
                        "world_alignment",
                        "endpoint_correction",
                        "foot_contacts",
                        "headless_scene",
                        "smart_primitives",
                    ]
                    + (["soma_ik"] if self.model == "soma" else []),
                ),
                indent=2,
            )
            + "\n"
        )
        index = dict(
            model_type="motionbricks",
            bundle_class="motius.models.motionbricks.release_bundle.MotionBricksReleaseBundle",
            pipeline_class="motius.pipelines.motionbricks.MotionBricksReleasePipeline",
            required_files=[
                "model.safetensors",
                "normalization.npz",
                "motionbricks_release.json",
            ],
        )
        (destination / "model_index.json").write_text(
            json.dumps(index, indent=2) + "\n"
        )
        return str(destination)
