"""Build complete Motius GEM-SMPL or GEM-X Hugging Face artifacts."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download

from motius.models.gem_smpl import GemSmplBundle
from motius.configuration.gem_smpl_provenance import HF_REVISION as GEM_SMPL_HF_REVISION
from motius.models.gem_x import GemXBundle
from motius.configuration.gem_x_provenance import HF_REVISION as GEM_X_HF_REVISION


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_REPO = "nvidia/GEM-X"
YOLOX_URL = (
    "https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/"
    "onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip"
)
YOLOX_NAME = "yolox_x_8xb8-300e_humanart-a39d44ed.onnx"
RELEASE_FILES = {
    "gem-smpl": {
        "card": ROOT / "docs/model_zoo/gem_smpl.md",
        "demo": ROOT / "assets/model_zoo/gem_smpl",
        "licenses": (
            ROOT / "motius/licenses/gem_smpl/ATTRIBUTIONS.md",
            ROOT / "motius/licenses/gem_smpl/GENMO_LICENSE",
            ROOT / "motius/licenses/gem_smpl/GENMO_ATTRIBUTIONS.md",
            ROOT / "motius/licenses/gem_smpl/GVHMR_LICENSE",
        ),
    },
    "gem-x": {
        "card": ROOT / "docs/model_zoo/gem_x.md",
        "demo": ROOT / "assets/model_zoo/gem_x",
        "licenses": (
            ROOT / "motius/licenses/gem_x/ATTRIBUTIONS.md",
            ROOT / "motius/licenses/gem_x/GEM_X_LICENSE",
            ROOT / "motius/licenses/gem_x/GEM_X_ATTRIBUTIONS.md",
            ROOT / "motius/licenses/flowhmr/DINOV3_LICENSE.md",
            ROOT
            / "motius/licenses/flowhmr/SAM3D_LICENSE",
            ROOT / "motius/licenses/soma/SOMA_LICENSE",
            ROOT / "motius/licenses/soma/SOMA_ATTRIBUTIONS.md",
        ),
    },
}


def _attach_release_files(method: str, output: Path) -> None:
    release = RELEASE_FILES[method]
    shutil.copy2(release["card"], output / "README.md")
    shutil.copytree(release["demo"], output / "demos", dirs_exist_ok=True)
    license_output = output / "licenses"
    license_output.mkdir(parents=True, exist_ok=True)
    for source in release["licenses"]:
        source = Path(source)
        shutil.copy2(source, license_output / source.name)


def _download_hf(filename: str, *, revision: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=UPSTREAM_REPO,
            filename=filename,
            revision=revision,
            cache_dir=str(cache_dir),
        )
    ).resolve()


def _download_yolox(cache_dir: Path) -> Path:
    target = cache_dir / YOLOX_NAME
    if target.is_file():
        return target
    archive = cache_dir / Path(YOLOX_URL).name
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not archive.is_file():
        urllib.request.urlretrieve(YOLOX_URL, archive)
    with zipfile.ZipFile(archive) as handle:
        member = next(
            name for name in handle.namelist() if name.endswith("end2end.onnx")
        )
        with handle.open(member) as source, target.open("wb") as destination:
            shutil.copyfileobj(source, destination)
    archive.unlink()
    return target


def _assemble_assets(output,sources,variant):
    import json
    from motius.utils.hashing import sha256_file
    import importlib
    provenance=importlib.import_module('motius.configuration.'+variant+'_provenance')
    output.mkdir(parents=True,exist_ok=True)
    for relative,source in sources.items():
        source=Path(source)
        if not source.is_file():raise FileNotFoundError(source)
        destination=output/relative;destination.parent.mkdir(parents=True,exist_ok=True)
        if source.resolve()!=destination.resolve():shutil.copy2(source,destination)
    checkpoint='inputs/pretrained/'+provenance.CHECKPOINT_FILENAME
    provenance.verify_checkpoint(output/checkpoint)
    metadata=dict(artifact_format='motius-'+variant.replace('_','-')+'-v2',model_type=variant,
        source_revision=provenance.SOURCE_REVISION,source_repository=provenance.SOURCE_REPOSITORY,
        checkpoint=checkpoint,assets=list(sources),sha256={key:sha256_file(output/key) for key in sources})
    (output/(variant+'_config.json')).write_text(json.dumps(metadata,indent=2)+'\n')


def export_gem_smpl(args: argparse.Namespace) -> None:
    sources = {
        "inputs/pretrained/gem_smpl.ckpt": _download_hf(
            "gem_smpl.ckpt",
            revision=GEM_SMPL_HF_REVISION,
            cache_dir=args.cache_dir,
        ),
        "inputs/checkpoints/hmr2/epoch=10-step=25000.ckpt": args.hmr2,
        "inputs/checkpoints/vitpose/vitpose-h-multi-coco.pth": args.vitpose_coco17,
        "yolov8x.pt": args.yolov8x,
    }
    _assemble_assets(args.output,sources,"gem_smpl")
    bundle = GemSmplBundle(
        artifact_root=args.output,
        body_models_root=args.body_models,
        device="cpu",
    )
    bundle.save_pretrained(str(args.output))
    _attach_release_files("gem-smpl", args.output)


def export_gem_x(args: argparse.Namespace) -> None:
    soma_assets = args.soma_assets
    upstream = {
        "inputs/pretrained/gem_soma.ckpt": "gem_soma.ckpt",
        "inputs/checkpoints/vitpose/vitpose.pth": "vitpose.pth",
        "inputs/checkpoints/sam-3d-body-dinov3/sam3d_body.ckpt": (
            "sam3d_body.ckpt"
        ),
        "inputs/checkpoints/sam-3d-body-dinov3/model_config.yaml": (
            "model_config.yaml"
        ),
        "inputs/mhr_data/mhr_model.pt": "mhr_model.pt",
        "inputs/soma_data/scale_mean.pth": "scale_mean.pth",
        "inputs/soma_data/scale_comps.pth": "scale_comps.pth",
    }
    sources = {
        relative: _download_hf(
            filename,
            revision=GEM_X_HF_REVISION,
            cache_dir=args.cache_dir,
        )
        for relative, filename in upstream.items()
    }
    sources["inputs/checkpoints/yolox/" + YOLOX_NAME] = _download_yolox(
        args.cache_dir / "yolox"
    )
    for relative in (
        "SOMA_neutral.npz",
        "correctives_model.pt",
        "MHR/mhr_model_lod6.pt",
        "MHR/base_body_lod6.obj",
        "MHR/SOMA_wrap_lod1.obj",
    ):
        sources[f"inputs/soma_assets/{relative}"] = soma_assets / relative
    _assemble_assets(args.output,sources,"gem_x")
    bundle = GemXBundle(artifact_root=args.output,device="cpu")
    bundle.save_pretrained(str(args.output))
    _attach_release_files("gem-x", args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("gem-smpl", "gem-x"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("outputs/cache/huggingface/gem"),
    )
    parser.add_argument("--hmr2", type=Path)
    parser.add_argument("--vitpose-coco17", type=Path)
    parser.add_argument("--yolov8x", type=Path)
    parser.add_argument("--body-models", type=Path)
    parser.add_argument("--soma-assets", type=Path)
    args = parser.parse_args()
    args.output = args.output.expanduser().resolve()
    args.cache_dir = args.cache_dir.expanduser().resolve()
    if args.method == "gem-smpl":
        required = ("hmr2", "vitpose_coco17", "yolov8x")
    else:
        required = ("soma_assets",)
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(missing))
    for name in required:
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.body_models is not None:
        args.body_models = args.body_models.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.method == "gem-smpl":
        export_gem_smpl(args)
    else:
        export_gem_x(args)


if __name__ == "__main__":
    main()
