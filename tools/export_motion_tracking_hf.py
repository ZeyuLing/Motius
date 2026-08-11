#!/usr/bin/env python3
"""Package official motion-tracking ONNX releases as Motius artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from motius.models.any2track import Any2TrackBundle
from motius.models.beyondmimic import BeyondMimicBundle
from motius.models.humanoid_gpt import HumanoidGPTBundle
from motius.models.protomotions import ProtoMotionsBundle
from motius.models.sonic import SONICBundle


METHODS = {
    "humanoid-gpt": {
        "bundle": HumanoidGPTBundle,
        "files": {
            "policy": "storage/ckpts/pns_wo_priv216.onnx",
            "scene": "storage/assets/unitree_g1_5010/scene_mjx_track.xml",
            "robot_mjcf": "storage/assets/unitree_g1_5010/g1_mjx_track.xml",
            "robot_license": "storage/assets/unitree_g1_5010/LICENSE",
        },
        "title": "Motius HumanoidGPT G1",
        "checkpoint": "ZeyuLing/Motius-HumanoidGPT-G1",
        "license": "apache-2.0",
    },
    "sonic": {
        "bundle": SONICBundle,
        "files": {
            "encoder": "model_encoder.onnx",
            "decoder": "model_decoder.onnx",
            "observation_config": "observation_config.yaml",
        },
        "title": "Motius SONIC G1",
        "checkpoint": "ZeyuLing/Motius-SONIC-G1",
        "license": "other",
        "license_name": "nvidia-open-model-license",
    },
    "protomotions": {
        "bundle": ProtoMotionsBundle,
        "files": {
            "policy": "compiled_models/unified_pipeline.onnx",
            "deployment_config": "compiled_models/unified_pipeline.yaml",
        },
        "title": "Motius ProtoMotions G1 BONES-SEED",
        "checkpoint": "ZeyuLing/Motius-ProtoMotions-G1-BONES-SEED",
        "license": "apache-2.0",
    },
    "any2track": {
        "bundle": Any2TrackBundle,
        "files": {"policy": "model.onnx", "training_config": "config.json"},
        "title": "Motius Any2Track G1 LAFAN1 v2",
        "checkpoint": "ZeyuLing/Motius-Any2Track-G1-LAFAN1-v2",
        "license": "apache-2.0",
    },
    "beyondmimic": {
        "bundle": BeyondMimicBundle,
        "files": {"policy": "policy.onnx"},
        "title": "Motius BeyondMimic G1 Runtime",
        "checkpoint": "<your-local-artifact>",
        "license": "bsd-3-clause",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readme(method: str, spec: dict) -> str:
    frontmatter = [
        "---",
        "library_name: motius",
        f"license: {spec['license']}",
    ]
    if spec.get("license_name"):
        frontmatter.extend(
            [
                f"license_name: {spec['license_name']}",
                "license_link: LICENSE",
            ]
        )
    frontmatter.extend(
        [
            "pipeline_tag: reinforcement-learning",
            "tags:",
            "- motion-tracking",
            "- robotics",
            "- unitree-g1",
            "- onnx",
            "---",
        ]
    )
    if method == "beyondmimic":
        note = (
            "This local artifact wraps a policy exported from your own BeyondMimic "
            "run. The upstream project does not publish a named pretrained policy."
        )
        call = "out = pipe.infer_motion_tracking(observation, time_step=0)"
    elif method == "humanoid-gpt":
        note = (
            "This artifact contains the official non-privileged Humanoid-GPT "
            "216 ONNX policy and the complete Unitree G1-5010 MuJoCo asset tree."
        )
        call = "out = pipe.infer_motion_tracking(observation)  # [B, 136]"
    elif method == "sonic":
        note = "This artifact contains the official default SONIC encoder and decoder."
        call = (
            "out = pipe.infer_motion_tracking(\n"
            "    encoder_observation,   # [1, 1762]\n"
            "    decoder_observation,   # [1, 930], excluding the 64D token\n"
            ")"
        )
    elif method == "protomotions":
        note = (
            "This artifact contains the official G1 BONES-SEED unified deployment "
            "pipeline and its exact input/output YAML contract."
        )
        call = "out = pipe.infer_motion_tracking(observations)"
    else:
        note = (
            "This artifact contains the official OpenTrack Any2Track LAFAN1 "
            "generalist v2 policy and config."
        )
        call = "out = pipe.infer_motion_tracking(observation)  # [B, 156]"
    return "\n".join(frontmatter) + f"""

# {spec['title']}

{note}

```python
from motius import Pipeline

pipe = Pipeline.from_pretrained("{spec['checkpoint']}")
{call}
```

`infer_motion_tracking` runs one controller step. A complete physical rollout
must supply simulator/robot state observations at 50 Hz and apply the returned
29-DOF action under the gains and ordering declared by the artifact.
"""


def export(args: argparse.Namespace) -> Path:
    spec = METHODS[args.method]
    source = args.source.expanduser().resolve()
    file_paths = {
        role: source / relative
        for role, relative in spec["files"].items()
    }
    if args.method == "beyondmimic" and source.is_file():
        file_paths = {"policy": source}
    bundle = spec["bundle"](
        file_paths=file_paths,
        load_model=False,
    )
    output = args.output.expanduser().resolve()
    bundle.save_pretrained(
        output,
        copy_mode=args.copy_mode,
        readme=_readme(args.method, spec),
    )

    required = []
    if args.license_file is not None:
        source_license = args.license_file.expanduser().resolve()
        if not source_license.is_file():
            raise FileNotFoundError(source_license)
        shutil.copy2(source_license, output / "LICENSE")
        required.append("LICENSE")

    required.append("artifact_inventory.json")
    model_index_path = output / "model_index.json"
    model_index = json.loads(model_index_path.read_text(encoding="utf-8"))
    model_index["required_files"] = [
        *model_index.get("required_files", []),
        *required,
    ]
    model_index_path.write_text(
        json.dumps(model_index, indent=2) + "\n",
        encoding="utf-8",
    )
    inventory = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "artifact_inventory.json":
            relative = str(path.relative_to(output))
            inventory[relative] = {
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    (output / "artifact_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": args.method,
                "files": inventory,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=sorted(METHODS))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--license-file", type=Path)
    parser.add_argument("--copy-mode", choices=("copy", "hardlink"), default="copy")
    return parser.parse_args()


def main() -> int:
    output = export(parse_args())
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
