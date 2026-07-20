#!/usr/bin/env python3
"""Publish self-contained Motius M2T artifacts to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


ROOT = Path(__file__).resolve().parents[1]

SPECS = {
    "motiongpt": {
        "repo_id": "ZeyuLing/Motius-MotionGPT-HumanML3D",
        "source_repo": "ZeyuLing/hftrainer-motiongpt-humanml3d",
        "card": ROOT / "docs/model_zoo/motiongpt.md",
        "required": ["motiongpt_s3_h3d.tar", "assets/meta/mean.npy"],
        "index": {
            "_class_name": "MotionGPTPipeline",
            "_library_name": "motius",
            "model_type": "motiongpt",
            "format": "motius-motiongpt-artifact-v1",
            "bundle_class": "motius.models.motiongpt.MotionGPTBundle",
            "pipeline_class": "motius.pipelines.motiongpt.MotionGPTPipeline",
            "api": "motius.pipelines.motiongpt.MotionGPTPipeline.from_pretrained",
        },
    },
    "motiongpt3": {
        "repo_id": "ZeyuLing/Motius-MotionGPT3-HumanML3D",
        "source_repo": "ZeyuLing/hftrainer-motiongpt3-humanml3d",
        "card": ROOT / "docs/model_zoo/motiongpt3.md",
        "required": ["motiongpt3.ckpt", "configs/test.yaml"],
        "index": {
            "_class_name": "MotionGPT3Pipeline",
            "_library_name": "motius",
            "model_type": "motiongpt3",
            "format": "motius-motiongpt3-artifact-v1",
            "bundle_class": "motius.models.motiongpt3.MotionGPT3Bundle",
            "pipeline_class": "motius.pipelines.motiongpt3.MotionGPT3Pipeline",
            "api": "motius.pipelines.motiongpt3.MotionGPT3Pipeline.from_pretrained",
        },
    },
    "tm2t": {
        "repo_id": "ZeyuLing/Motius-TM2T-HumanML3D",
        "source_path": ROOT / "checkpoints/tm2t",
        "card": ROOT / "docs/model_zoo/tm2t.md",
        "required": [
            "glove/our_vab_data.npy",
            "t2m/M2T_EL4_DL4_NH8_PS/model/finest.tar",
            "t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/model/finest.tar",
        ],
        "allow_prefixes": (
            "glove/",
            "t2m/M2T_EL4_DL4_NH8_PS/",
            "t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/",
        ),
        "index": {
            "_class_name": "TM2TPipeline",
            "_library_name": "motius",
            "model_type": "tm2t",
            "format": "motius-tm2t-artifact-v1",
            "bundle_class": "motius.models.tm2t.TM2TBundle",
            "pipeline_class": "motius.pipelines.tm2t.TM2TPipeline",
            "api": "motius.pipelines.tm2t.TM2TPipeline.from_pretrained",
        },
    },
    "vermo": {
        "repo_id": "ZeyuLing/Motius-VerMo-HumanML3D",
        "source_path": ROOT / "checkpoints/vermo/motius_humanml3d",
        "card": ROOT / "docs/model_zoo/vermo.md",
        "required": [
            "bundle_config.json",
            "lm/model.safetensors",
            "motion_tokenizer/diffusion_pytorch_model.safetensors",
        ],
        "index": {
            "_class_name": "VermoPipeline",
            "_library_name": "motius",
            "model_type": "vermo",
            "format": "motius-vermo-artifact-v1",
            "bundle_class": "motius.models.vermo.VermoBundle",
            "pipeline_class": "motius.pipelines.vermo.VermoPipeline",
            "api": "motius.pipelines.vermo.VermoPipeline.from_pretrained",
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True, choices=tuple(SPECS))
    parser.add_argument(
        "--staging-root",
        default=str(ROOT / "outputs/hf_artifacts/m2t"),
    )
    parser.add_argument("--source-path", default="")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Update README and model_index.json without restaging model weights.",
    )
    parser.add_argument("--private", action="store_true")
    return parser.parse_args()


def link_tree(source: Path, target: Path, allow_prefixes=None) -> None:
    for path in source.glob("**/*"):
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        if relative in {"README.md", "model_index.json"}:
            continue
        if relative.startswith(".cache/"):
            continue
        if allow_prefixes and not relative.startswith(tuple(allow_prefixes)):
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(path.resolve())


def write_metadata(method: str, staging: Path, spec: dict) -> None:
    card = spec["card"].read_text(encoding="utf-8")
    frontmatter = (
        "---\n"
        "library_name: motius\n"
        "pipeline_tag: other\n"
        "tags:\n"
        "- motion-to-text\n"
        "- motion-captioning\n"
        "- humanml3d\n"
        f"- {method}\n"
        "---\n\n"
    )
    (staging / "README.md").write_text(frontmatter + card, encoding="utf-8")
    (staging / "model_index.json").write_text(
        json.dumps(spec["index"], indent=2) + "\n",
        encoding="utf-8",
    )
    if method == "vermo":
        manifest = {
            "format": "motius_vermo_bundle_v1",
            "task": "M2T",
            "motion_representation": "vermo_smpl22_abs_rel_rot6d_column_138",
            "vocab_size": 147743,
            "torch_dtype": "bf16",
        }
        path = staging / "artifact_manifest.json"
        if path.exists() or path.is_symlink():
            path.unlink()
        path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    method = args.method
    spec = SPECS[method]
    staging = Path(args.staging_root).expanduser().resolve() / method
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    if not args.metadata_only:
        if args.source_path:
            source = Path(args.source_path).expanduser().resolve()
        elif "source_repo" in spec:
            source = Path(snapshot_download(spec["source_repo"]))
        else:
            source = Path(spec["source_path"]).resolve()
        link_tree(source, staging, spec.get("allow_prefixes"))
    write_metadata(method, staging, spec)

    if not args.metadata_only:
        for required in spec["required"]:
            if not (staging / required).is_file():
                raise FileNotFoundError(f"{method} staging is missing {required}")

    api = HfApi()
    api.create_repo(
        repo_id=spec["repo_id"],
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )
    if args.metadata_only:
        api.upload_folder(
            repo_id=spec["repo_id"],
            folder_path=staging,
            repo_type="model",
            commit_message="Update M2T model card and Motius metadata",
        )
    else:
        api.upload_large_folder(
            repo_id=spec["repo_id"],
            folder_path=staging,
            repo_type="model",
            num_workers=args.num_workers,
            print_report=True,
            print_report_every=60,
        )
    files = {item.rfilename for item in api.repo_info(spec["repo_id"]).siblings}
    missing = [path for path in spec["required"] if path not in files]
    if missing:
        raise RuntimeError(f"Hub upload is incomplete: {missing}")
    print(json.dumps({"method": method, "repo_id": spec["repo_id"], "files": len(files)}))


if __name__ == "__main__":
    main()
