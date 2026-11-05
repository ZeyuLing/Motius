"""Read DART's serialized argument data without importing its trainer classes."""

from pathlib import Path
from types import SimpleNamespace
import yaml


def inference_options(metadata):
    """Read native inference settings and correct the legacy framewise floor default."""
    names = {
        "denoiser_checkpoint",
        "seed_sequence",
        "guidance_param",
        "respacing",
        "zero_noise",
        "use_predicted_joints",
        "fix_floor",
        "coord_conversion",
        "translation_source",
        "initial_transform",
        "rollout_body_type",
    }
    result = {key: value for key, value in metadata.items() if key in names}
    if (
        metadata.get("format_version", 1) < 2
        and result.get("translation_source") == "floor_aligned_smpl_transl"
    ):
        result["translation_source"] = "smpl_transl"
    return result


def artifact_metadata(options, denoiser_checkpoint):
    """Describe a current DART artifact for public automatic cold loading."""
    return dict(
        options,
        model_type="dart",
        format_version=2,
        denoiser_checkpoint=denoiser_checkpoint,
        _class_name="DARTPipeline",
        _library_name="motius",
        pipeline_class="motius.pipelines.dart.DARTPipeline",
        bundle_class="motius.models.dart.DARTBundle",
        tasks=["text_to_motion"],
        requires_body_model=True,
    )


class _DARTLoader(yaml.SafeLoader):
    pass


def _dataclass(loader, suffix, node):
    allowed = {
        "MLDArgs",
        "Args",
        "DataArgs",
        "TrainArgs",
        "DenoiserArgs",
        "DiffusionArgs",
        "DenoiserMLPArgs",
        "DenoiserTransformerArgs",
        "VAEArgs",
    }
    if suffix not in allowed:
        raise ValueError(f"Unknown DART argument type: {suffix}")
    return loader.construct_mapping(node, deep=True)


_DARTLoader.add_multi_constructor("!dataclass:", _dataclass)
_DARTLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple",
    lambda loader, node: tuple(loader.construct_sequence(node, deep=True)),
)
_DARTLoader.add_constructor(
    "tag:yaml.org,2002:python/object/apply:pathlib.PosixPath",
    lambda loader, node: str(Path(*loader.construct_sequence(node, deep=True))),
)


def read_arguments(path):
    data = yaml.safe_load(Path(path).read_text())
    if isinstance(data, str):
        data = yaml.load(data, Loader=_DARTLoader)

    def namespace(value):
        return (
            SimpleNamespace(**{k: namespace(v) for k, v in value.items()})
            if isinstance(value, dict)
            else value
        )

    return namespace(data)
