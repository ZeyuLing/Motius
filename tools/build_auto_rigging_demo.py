#!/usr/bin/env python3
"""Reproduce the public real-mesh auto-rigging and animation demo.

The orchestration layer intentionally uses only the standard library. Blender
performs all geometry, skinning, FBX, animation, and rendering work. Pillow is
needed only to assemble the README GIF from rendered PNG frames.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = (
    "https://download.blender.org/demo/bundles/bundles-3.6/"
    "human-base-meshes-bundle-v1.0.0.zip"
)
SOURCE_SHA256 = "46a912c0524072ac3b78c35d5d2471df7b8df102394a050ca8cd7184e3393648"
SOURCE_LICENSE = "CC0 1.0"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--blender",
        type=Path,
        default=os.environ.get("MOTIUS_BLENDER"),
        required=os.environ.get("MOTIUS_BLENDER") is None,
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("outputs/auto_rigging_demo/blender_cc0_male_004822"),
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=Path("outputs/auto_rigging_template_regression/published"),
    )
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--resolution", type=int, default=640)
    parser.add_argument("--gif-resolution", type=int, default=420)
    parser.add_argument("--gif-step", type=int, default=3)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def _blender(blender: Path, script: Path, *values: object, blend=None) -> None:
    command = [str(blender)]
    if blend is not None:
        command.append(str(Path(blend).resolve()))
    command.extend(
        [
            "--background",
            "--python-exit-code",
            "1",
            "--python",
            str(script.resolve()),
            "--",
            *(str(value) for value in values),
        ]
    )
    _run(command)


def _source_bundle(args, work: Path) -> tuple[Path, Path]:
    archive = (
        args.source_archive.expanduser().resolve()
        if args.source_archive
        else work / "human-base-meshes-bundle-v1.0.0.zip"
    )
    if not archive.is_file():
        print(f"Downloading {SOURCE_URL}")
        urllib.request.urlretrieve(SOURCE_URL, archive)
    actual = _sha256(archive)
    if actual != SOURCE_SHA256:
        raise ValueError(
            f"Human Base Meshes SHA-256 mismatch: expected {SOURCE_SHA256}, got {actual}."
        )
    extracted = work / "source_bundle"
    blend = extracted / "human_base_meshes_bundle.blend"
    if not blend.is_file():
        extracted.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(extracted)
    if not blend.is_file():
        candidates = sorted(extracted.rglob("human_base_meshes_bundle.blend"))
        if len(candidates) != 1:
            raise FileNotFoundError(
                "Could not identify human_base_meshes_bundle.blend."
            )
        blend = candidates[0]
    return archive, blend


def _assemble_gif(frames_dir: Path, output: Path, duration_ms: int) -> int:
    try:
        from PIL import Image
    except ImportError as error:
        raise ImportError(
            "Pillow is required to assemble the README GIF: python -m pip install Pillow"
        ) from error
    paths = sorted(frames_dir.glob("frame_*.png"))
    if not paths:
        raise FileNotFoundError(f"No rendered GIF frames found under {frames_dir}.")
    images = [
        Image.open(path).convert("P", palette=Image.Palette.ADAPTIVE) for path in paths
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
        optimize=True,
    )
    for image in images:
        image.close()
    return len(paths)


def _assemble_visual_qa(diagnostics: Path, output: Path, *, worst_frame: int) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError as error:
        raise ImportError(
            "Pillow is required to assemble the visual QA contact sheet."
        ) from error
    pose_fronts = sorted(diagnostics.glob("pose_*_front.png"))
    if len(pose_fronts) < 3:
        raise FileNotFoundError(
            f"Expected at least three pose diagnostics under {diagnostics}."
        )
    worst = diagnostics / f"pose_{worst_frame:04d}_front.png"
    frame_numbers = {path: int(path.stem.split("_")[1]) for path in pose_fronts}
    midpoint = (frame_numbers[pose_fronts[0]] + frame_numbers[pose_fronts[-1]]) / 2
    middle = min(pose_fronts, key=lambda path: abs(frame_numbers[path] - midpoint))
    selected = list(
        dict.fromkeys(
            [
                pose_fronts[0],
                middle,
                worst,
                pose_fronts[-1],
            ]
        )
    )
    panels = [
        ("rest_skeleton_front.png", "Rest skeleton / front"),
        ("rest_skeleton_side.png", "Rest skeleton / side"),
        ("dominant_weights_front.png", "Dominant weights / front"),
        ("dominant_weights_back.png", "Dominant weights / back"),
    ]
    for front in selected:
        frame = int(front.stem.split("_")[1])
        panels.extend(
            [
                (front.name, f"Frame {frame} / front"),
                (
                    front.name.replace("_front.png", "_side.png"),
                    f"Frame {frame} / side",
                ),
            ]
        )
    tile = 384
    label = 30
    columns = 2
    rows = (len(panels) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * tile, rows * (tile + label)), "#11141a")
    draw = ImageDraw.Draw(sheet)
    for index, (name, caption) in enumerate(panels):
        path = diagnostics / name
        if not path.is_file():
            raise FileNotFoundError(f"Missing visual QA panel: {path}")
        with Image.open(path) as image:
            panel = image.convert("RGB")
            panel.thumbnail((tile, tile))
            x = (index % columns) * tile + (tile - panel.width) // 2
            y = (index // columns) * (tile + label)
            sheet.paste(panel, (x, y))
        draw.text((index % columns * tile + 10, y + tile + 7), caption, fill="#f2f4f8")
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, optimize=True)


def _portable_report(value, *, work: Path):
    if isinstance(value, dict):
        return {key: _portable_report(item, work=work) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_report(item, work=work) for item in value]
    if not isinstance(value, str):
        return value
    try:
        path = Path(value)
        if not path.is_absolute():
            return value
        try:
            relative = path.relative_to(ROOT)
            return relative.as_posix()
        except ValueError:
            relative = path.relative_to(work)
            return (
                Path("outputs/auto_rigging_demo/blender_cc0_male_004822") / relative
            ).as_posix()
    except (OSError, ValueError):
        return value


def build(args: argparse.Namespace) -> Path:
    blender = args.blender.expanduser().resolve()
    if not blender.is_file():
        raise FileNotFoundError(f"Blender executable does not exist: {blender}.")
    work = args.work_dir.expanduser().resolve()
    publish = args.publish_dir.expanduser().resolve()
    work.mkdir(parents=True, exist_ok=True)
    publish.mkdir(parents=True, exist_ok=True)
    archive, source_blend = _source_bundle(args, work)

    source_glb = work / "blender_cc0_male_unrigged.glb"
    source_export = work / "source_export.json"
    source_validation = work / "unrigged_validation.json"
    rigged_fbx = work / "blender_cc0_male_rigged.fbx"
    rigging_report = Path(f"{rigged_fbx}.json")
    rig_validation = work / "rigged_validation.json"
    animated_fbx = work / "blender_cc0_male_004822.fbx"
    animation_report = work / "animation_report.json"
    animation_validation = work / "animation_validation.json"
    mp4 = work / "blender_cc0_male_autorig_004822_640_30fps.mp4"
    frames_dir = work / "gif_frames"
    render_report = work / "render_report.json"
    diagnostics_dir = work / "diagnostics"
    diagnostics_report = work / "diagnostics.json"

    _blender(
        blender,
        ROOT / "tools/blender_extract_human_base_mesh.py",
        "--output",
        source_glb,
        "--report",
        source_export,
        "--object-pattern",
        "body_male_realistic",
        blend=source_blend,
    )
    _blender(
        blender,
        ROOT / "tools/blender_validate_unrigged_asset.py",
        "--asset",
        source_glb,
        "--report",
        source_validation,
    )

    rig_job = work / "rig_job.json"
    rig_job.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "method": "template",
                "character_path": str(source_glb),
                "output_path": str(rigged_fbx),
                "manifest_path": str(rigging_report),
                "template_module": str(
                    (ROOT / "motius/motion/rigging/template.py").resolve()
                ),
                "up_axis": "auto",
                "replace_existing_rig": False,
                "weight_method": "capsules",
                "config": {
                    "top_k": 4,
                    "weight_falloff": 1.75,
                    "side_penalty": 0.025,
                    "chunk_size": 16384,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _blender(
        blender,
        ROOT / "motius/motion/rigging/_blender.py",
        "--job",
        rig_job,
    )
    _blender(
        blender,
        ROOT / "tests/blender_validate_rigged_asset.py",
        "--asset",
        rigged_fbx,
        "--report",
        rig_validation,
    )
    _blender(
        blender,
        ROOT / "tools/blender_retarget_smpl22_joints.py",
        "--input",
        rigged_fbx,
        "--motion",
        ROOT / "assets/motion/representation_demo/data.json",
        "--output",
        animated_fbx,
        "--report",
        animation_report,
        "--frames",
        args.frames,
        "--start-frame",
        0,
        "--fps",
        args.fps,
    )
    _blender(
        blender,
        ROOT / "tests/blender_validate_rigged_asset.py",
        "--asset",
        animated_fbx,
        "--report",
        animation_validation,
        "--require-animation",
        "--require-deformation",
    )
    if diagnostics_dir.exists():
        shutil.rmtree(diagnostics_dir)
    _blender(
        blender,
        ROOT / "tools/blender_render_rigging_diagnostics.py",
        "--rigged",
        rigged_fbx,
        "--animated",
        animated_fbx,
        "--output-dir",
        diagnostics_dir,
        "--report",
        diagnostics_report,
        "--resolution",
        512,
    )
    diagnostics_payload = json.loads(diagnostics_report.read_text(encoding="utf-8"))
    animation_payload = json.loads(animation_report.read_text(encoding="utf-8"))
    stretch = diagnostics_payload["edge_stretch"]
    deformation_limits = {
        "combined_p99_ratio_from_one": 1.65,
        "combined_p999_ratio_from_one": 2.60,
        "combined_fraction_beyond_1_5x": 0.02,
        "maximum_frame_p99_ratio_from_one": 1.65,
        "maximum_frame_fraction_beyond_1_5x": 0.02,
    }
    violations = {
        key: (float(stretch[key]), limit)
        for key, limit in deformation_limits.items()
        if float(stretch[key]) > limit
    }
    if violations:
        raise AssertionError(f"Rigging deformation QA failed: {violations}")
    direction = animation_payload["bone_direction_error"]
    direction_limits = {
        "mean_degrees": 1.0,
        "p95_degrees": 7.5,
        "max_degrees": 8.0,
        "maximum_frame_mean_degrees": 1.0,
        "maximum_frame_p95_degrees": 4.0,
    }
    violations = {
        key: (float(direction[key]), limit)
        for key, limit in direction_limits.items()
        if float(direction[key]) > limit
    }
    if violations:
        raise AssertionError(f"Animation direction QA failed: {violations}")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    _blender(
        blender,
        ROOT / "tools/blender_render_auto_rigging_demo.py",
        "--input",
        animated_fbx,
        "--output",
        mp4,
        "--frames-dir",
        frames_dir,
        "--report",
        render_report,
        "--frames",
        args.frames,
        "--fps",
        args.fps,
        "--resolution",
        args.resolution,
        "--gif-resolution",
        args.gif_resolution,
        "--gif-step",
        args.gif_step,
    )

    published_mp4 = publish / mp4.name
    published_gif = publish / "blender_cc0_male_autorig_004822_readme.gif"
    published_visual_qa = publish / "blender_cc0_male_autorig_visual_qa.png"
    shutil.copy2(mp4, published_mp4)
    gif_frames = _assemble_gif(
        frames_dir,
        published_gif,
        duration_ms=max(1, round(args.gif_step / args.fps * 1000)),
    )
    _assemble_visual_qa(
        diagnostics_dir,
        published_visual_qa,
        worst_frame=int(stretch["worst_frame_p999"]),
    )
    reports = {
        "source_export": source_export,
        "unrigged_validation": source_validation,
        "rigging": rigging_report,
        "rigged_validation": rig_validation,
        "animation": animation_report,
        "animation_validation": animation_validation,
        "render": render_report,
        "diagnostics": diagnostics_report,
    }
    copied_reports = {}
    for name, path in reports.items():
        target = publish / f"{name}.json"
        portable = _portable_report(
            json.loads(path.read_text(encoding="utf-8")),
            work=work,
        )
        target.write_text(json.dumps(portable, indent=2) + "\n", encoding="utf-8")
        copied_reports[name] = target.name
    manifest = {
        "schema_version": 1,
        "source": {
            "name": "Blender Human Base Meshes v1.0.0 / realistic male",
            "url": SOURCE_URL,
            "license": SOURCE_LICENSE,
            "archive_sha256": _sha256(archive),
            "source_object": "GEO-body_male_realistic",
            "input_asset_committed": False,
        },
        "motion": {
            "case_id": "004822",
            "source": "assets/motion/representation_demo/data.json",
            "representation": "persisted HumanML3D SMPL22 joint trajectory",
            "frames": args.frames,
            "source_start_frame": 0,
            "fps": args.fps,
        },
        "pipeline": [
            "download and SHA-256 verify public CC0 source",
            "export static GLB and prove no armature, skin, weights, or action",
            "Motius template auto-rig to canonical SMPL22 FBX",
            "drive the target from persisted HumanML3D motion 004822",
            "validate animation action and measured vertex deformation",
            "render rest-skeleton, dominant-weight, and multi-pose diagnostics",
            "enforce bone-direction and robust local edge-stretch limits before publication",
            "render MP4 and README GIF with Blender EEVEE Next",
        ],
        "artifacts": {
            "gif": published_gif.name,
            "gif_sha256": _sha256(published_gif),
            "gif_frames": gif_frames,
            "mp4": published_mp4.name,
            "mp4_sha256": _sha256(published_mp4),
            "visual_qa": published_visual_qa.name,
            "visual_qa_sha256": _sha256(published_visual_qa),
            "deformation_quality_limits": deformation_limits,
            "animation_direction_quality_limits": direction_limits,
            "reports": copied_reports,
        },
    }
    manifest_path = publish / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


if __name__ == "__main__":
    print(build(_parser().parse_args()))
