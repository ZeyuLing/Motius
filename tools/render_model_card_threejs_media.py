#!/usr/bin/env python3
"""Render Model Card videos from the canonical Three.js Leaderboard scenes."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import imageio_ffmpeg
from pathlib import Path
import re
import subprocess
import sys
from typing import Optional, Union
from urllib.parse import parse_qs, urlsplit

try:
    from tools.audit_model_card_media import _audit_one
except ModuleNotFoundError:
    from audit_model_card_media import _audit_one


ROOT = Path(__file__).resolve().parents[1]
ATTACHMENTS = ROOT / "docs/model_zoo/video_attachments.json"
T2M_MANIFEST = (
    ROOT / "docs/leaderboards/hf_space_t2m_humanml3d/cases/manifest.json"
)
T2M_VIEWER = (
    ROOT / "docs/leaderboards/hf_space_t2m_humanml3d/cases/index.html"
)
M2D_VIEWER = (
    ROOT / "docs/leaderboards/hf_space_music_to_dance/cases/index.html"
)
CAPTURE_TOOL = ROOT / "tools/capture_leaderboard_method_gif.py"

T2M_PACKAGES = {
    "condmdi": "condmdi",
    "dart": "dart",
    "flowmdm": "flowmdm",
    "hymotion_t2m": "hymotion1b",
    "kimodo": "kimodo",
    "maskcontrol": "maskcontrol",
    "mdm": "mdm",
    "mld": "mld",
    "mogents": "mogents",
    "momask": "momask",
    "motionclr": "motionclr",
    "motiongpt": "motiongpt",
    "motiongpt3": "motiongpt3",
    "motionlcm": "motionlcm",
    "motioncanvas": "motioncanvas",
    "motionmillion": "gotozero7b",
    "motionstreamer": "motionstreamer",
    "prism_1_0": "prism1",
    "prism_kt": "prismkafs",
    "t2mgpt": "t2mgpt",
    "tm2d": "tm2d",
    "unimumo": "unimumo",
    "vimogen": "vimogenrewrite",
}

T2M_NAME = re.compile(
    r"_humanml3d_(?P<case>.+?)_smpl_mesh_512_30fps\.gif$"
)


@dataclass(frozen=True)
class RenderJob:
    source: str
    method: str
    label: str
    case_id: str
    viewer: Union[str, Path] = T2M_VIEWER
    include_audio: bool = False
    audio_track: Optional[str] = None
    representation: str = "smpl"
    layout: str = "tile"
    fps: int = 30

    def output(self, root: Path) -> Path:
        return root / Path(self.source).with_suffix(".mp4")


def _t2m_jobs() -> list[RenderJob]:
    attachments = json.loads(ATTACHMENTS.read_text(encoding="utf-8"))
    manifest = json.loads(T2M_MANIFEST.read_text(encoding="utf-8"))
    labels = {
        entry["key"]: entry["label"]
        for entry in manifest["motion_methods"]
    }
    jobs = []
    for source in sorted(attachments["videos"]):
        path = Path(source)
        if len(path.parts) < 4:
            continue
        package = path.parts[2]
        method = T2M_PACKAGES.get(package)
        match = T2M_NAME.search(path.name)
        if method is None or match is None:
            continue
        jobs.append(
            RenderJob(
                source=source,
                method=method,
                label=labels[method],
                case_id=match.group("case"),
            )
        )
    special = (
        "assets/model_zoo/flowmdm/"
        "flowmdm_text_to_motion_512_30fps.gif"
    )
    if special in attachments["videos"]:
        jobs.append(
            RenderJob(
                source=special,
                method="flowmdm",
                label=labels["flowmdm"],
                case_id="001840",
            )
        )
    shared = (
        "assets/model_zoo/shared/"
        "motion_to_text_input_smpl_512_30fps.gif"
    )
    if shared in attachments["videos"]:
        jobs.append(
            RenderJob(
                source=shared,
                method="gt",
                label=labels["gt"],
                case_id="001840",
            )
        )
    return sorted(jobs, key=lambda job: job.source)


def _sidecar_jobs() -> list[RenderJob]:
    attachments = json.loads(ATTACHMENTS.read_text(encoding="utf-8"))
    jobs = []
    for source in sorted(attachments["videos"]):
        sidecar = (ROOT / source).with_suffix(".json")
        if not sidecar.is_file():
            continue
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        viewer_url = payload.get("viewer_url")
        if not viewer_url:
            continue
        parsed = urlsplit(viewer_url)
        query = parse_qs(parsed.query)
        method = query.get("method", [None])[0]
        case_id = query.get("case", [None])[0]
        if not method or not case_id:
            raise ValueError(f"Incomplete viewer URL in {sidecar}")
        jobs.append(
            RenderJob(
                source=source,
                method=method,
                label=str(payload["method"]),
                case_id=case_id,
                viewer=_local_viewer(viewer_url.split("?", 1)[0]),
            )
        )
    return jobs


def _music_to_dance_jobs() -> list[RenderJob]:
    specs = (
        (
            "assets/model_zoo/bailando/"
            "bailando_aistpp_break_gBR_mBR0_smpl_mesh_512_30fps.gif",
            "bailando",
            "Bailando",
            "gBR_sBM_cAll_d04_mBR0_ch01",
        ),
        (
            "assets/model_zoo/bailando/"
            "bailando_aistpp_krump_gKR_mKR2_smpl_mesh_512_30fps.gif",
            "bailando",
            "Bailando",
            "gKR_sBM_cAll_d28_mKR2_ch01",
        ),
        (
            "assets/model_zoo/bailando/"
            "bailando_aistpp_waacking_gWA_mWA0_smpl_mesh_512_30fps.gif",
            "bailando",
            "Bailando",
            "gWA_sBM_cAll_d25_mWA0_ch01",
        ),
        (
            "assets/model_zoo/edge/"
            "edge_aistpp_gBR_mBR0_smpl_mesh_512_30fps.gif",
            "edge",
            "EDGE",
            "gBR_sBM_cAll_d04_mBR0_ch01",
        ),
        (
            "assets/model_zoo/tm2d/"
            "tm2d_aistpp_gBR_mBR0_smpl_mesh_512_30fps.gif",
            "tm2d",
            "TM2D",
            "gBR_sBM_cAll_d04_mBR0_ch01",
        ),
        (
            "assets/model_zoo/unimumo/"
            "unimumo_aistpp_gBR_mBR0_smpl_mesh_512_30fps.gif",
            "unimumo",
            "UniMuMo",
            "gBR_sBM_cAll_d04_mBR0_ch01",
        ),
    )
    attachments = json.loads(ATTACHMENTS.read_text(encoding="utf-8"))
    return [
        RenderJob(
            source=source,
            method=method,
            label=label,
            case_id=case_id,
            viewer=M2D_VIEWER,
            include_audio=True,
            representation="smpl-plus-native-skeleton",
            layout="stage",
        )
        for source, method, label, case_id in specs
        if source in attachments["videos"]
    ]


def _motioncanvas_jobs() -> list[RenderJob]:
    specs = (
        RenderJob(
            source=(
                "assets/model_zoo/motioncanvas/"
                "motioncanvas_humanml3d_004822_smpl_mesh_512_30fps.gif"
            ),
            method="motioncanvas",
            label="MotionCanvas 0.46B",
            case_id="004822",
        ),
        RenderJob(
            source=(
                "assets/model_zoo/motioncanvas/"
                "motioncanvas_motion_repair_512_20fps.gif"
            ),
            method="motioncanvas",
            label="MotionCanvas",
            case_id="repair_000",
            viewer=(
                ROOT
                / "docs/leaderboards/hf_space_motion_repair/cases/index.html"
            ),
            fps=20,
        ),
    )
    return [
        job for job in specs if (ROOT / job.source).is_file()
    ]


def _native_jobs() -> list[RenderJob]:
    specs = (
        ("assets/model_zoo/condmdi/condmdi_kinematic_motion_control_512_30fps.gif", "condmdi", "CondMDI", "condmdi_kinematic_motion_control", "condmdi/kinematic_motion_control", "humanml3d-263-native-skeleton", 30),
        ("assets/model_zoo/ardy/ardy_kinematic_motion_control_512_30fps.gif", "ardy", "ARDY", "ardy_kinematic_native", "ardy/kinematic_motion_control", "ardy-330-native-mesh", 20),
        ("assets/model_zoo/ardy/ardy_sequential_text_to_motion_512_20fps.gif", "ardy", "ARDY", "ardy_sequential_native", "ardy/sequential_text_to_motion", "ardy-330-native-mesh", 20),
        ("assets/model_zoo/ardy/ardy_text_to_motion_512_20fps.gif", "ardy", "ARDY", "ardy_text_native", "ardy/text_to_motion", "ardy-330-native-mesh", 20),
        ("assets/model_zoo/gem_smpl/gem_smpl_tennis_world.webp", "gem_smpl", "GEM-SMPL", "gem_smpl_tennis", "gem_smpl/monocular_motion_capture", "smpl-native-mesh", 30),
        ("assets/model_zoo/gem_x/gem_x_tennis_world.webp", "gem_x", "GEM-X", "gem_x_tennis", "gem_x/monocular_motion_capture", "soma-30-native-mesh", 30),
        ("assets/model_zoo/gvhmr/case_01_global.webp", "gvhmr", "GVHMR", "gvhmr_case_01", "gvhmr/case_01", "smpl-native-mesh", 30),
        ("assets/model_zoo/gvhmr/case_02_global.webp", "gvhmr", "GVHMR", "gvhmr_case_02", "gvhmr/case_02", "smpl-native-mesh", 30),
        ("assets/model_zoo/gvhmr/case_03_global.webp", "gvhmr", "GVHMR", "gvhmr_case_03", "gvhmr/case_03", "smpl-native-mesh", 30),
        ("assets/model_zoo/intergen/intergen_interhuman_handshake_smpl_pair_512_30fps.gif", "intergen", "InterGen", "handshake", "intergen/handshake", "interhuman-262-native-skeleton", 30),
        ("assets/model_zoo/intergen/intergen_interhuman_help_stand_smpl_pair_512_30fps.gif", "intergen", "InterGen", "help_stand", "intergen/help_stand", "interhuman-262-native-skeleton", 30),
        ("assets/model_zoo/intermask/intermask_interhuman_gentle_push_smpl_pair_512_30fps.gif", "intermask", "InterMask", "gentle_push", "intermask/gentle_push", "interhuman-262-native-skeleton", 30),
        ("assets/model_zoo/intermask/intermask_interhuman_hug_smpl_pair_512_30fps.gif", "intermask", "InterMask", "hug", "intermask/hug", "interhuman-262-native-skeleton", 30),
        ("assets/model_zoo/kimodo/kimodo_kinematic_motion_control_512_30fps.gif", "kimodo", "KIMODO", "kimodo_kinematic_motion_control", "kimodo/kinematic_motion_control", "soma-30-native-skeleton", 30),
        ("assets/model_zoo/kimodo/kimodo_sequential_text_to_motion_512_30fps.gif", "kimodo", "KIMODO", "kimodo_sequential_text_to_motion", "kimodo/sequential_text_to_motion", "soma-30-native-skeleton", 30),
        ("assets/model_zoo/kimodo/kimodo_temporal_motion_completion_512_30fps.gif", "kimodo", "KIMODO", "kimodo_temporal_motion_completion", "kimodo/temporal_motion_completion", "soma-30-native-skeleton", 30),
        ("assets/model_zoo/maskcontrol/maskcontrol_kinematic_motion_control_512_30fps.gif", "maskcontrol", "MaskControl", "maskcontrol_body_control", "maskcontrol/body_control", "smpl-native-mesh", 30),
        ("assets/model_zoo/maskcontrol/maskcontrol_part_level_motion_control_512_30fps.gif", "maskcontrol", "MaskControl", "maskcontrol_body_part", "maskcontrol/body_part", "smpl-native-mesh", 30),
        ("assets/model_zoo/maskcontrol/maskcontrol_temporal_motion_completion_512_30fps.gif", "maskcontrol", "MaskControl", "maskcontrol_temporal_motion_completion", "maskcontrol/temporal_motion_completion", "humanml3d-263-native-skeleton", 30),
        ("assets/model_zoo/motionbricks/motionbricks_g1_random_rollout.gif", "motionbricks", "MotionBricks", "motionbricks_random_rollout", "motionbricks/random_rollout", "motionbricks-g1-414-native-mesh", 30),
        ("assets/model_zoo/motionstreamer/motionstreamer_temporal_motion_completion_512_30fps.gif", "motionstreamer", "MotionStreamer", "motionstreamer_temporal_motion_completion", "motionstreamer/temporal_motion_completion", "motionstreamer-272-native-skeleton", 30),
        ("assets/model_zoo/omnicontrol/omnicontrol_kinematic_motion_control_512_30fps.gif", "omnicontrol", "OmniControl", "omnicontrol_kinematic_motion_control", "omnicontrol/kinematic_motion_control", "humanml3d-263-native-skeleton", 30),
        ("assets/model_zoo/omnicontrol/omnicontrol_temporal_motion_completion_512_30fps.gif", "omnicontrol", "OmniControl", "omnicontrol_temporal_motion_completion", "omnicontrol/temporal_motion_completion", "humanml3d-263-native-skeleton", 30),
        ("assets/model_zoo/omnicontrol/omnicontrol_text_to_motion_512_30fps.gif", "omnicontrol", "OmniControl", "omnicontrol_text_to_motion", "omnicontrol/text_to_motion", "humanml3d-263-native-skeleton", 30),
        ("assets/model_zoo/prism_kt/prism_temporal_motion_completion_512_30fps.gif", "prism", "PRISM-KT", "prism_temporal_motion_completion", "prism/temporal_motion_completion", "smpl-native-mesh", 30),
        ("assets/model_zoo/projflow/projflow_temporal_motion_completion_512_20fps.gif", "projflow", "ProjFlow", "projflow_temporal_motion_completion", "projflow/temporal_motion_completion", "humanml3d-smpl22-native-joints", 20),
        ("assets/model_zoo/projflow/projflow_kinematic_motion_control_512_20fps.gif", "projflow", "ProjFlow", "projflow_kinematic_motion_control", "projflow/kinematic_motion_control", "humanml3d-smpl22-native-joints", 20),
        ("assets/model_zoo/projflow/projflow_part_level_motion_control_512_20fps.gif", "projflow", "ProjFlow", "projflow_part_level_motion_control", "projflow/part_level_motion_control", "humanml3d-smpl22-native-joints", 20),
    )
    attachments = json.loads(ATTACHMENTS.read_text(encoding="utf-8"))
    jobs = []
    for source, method, label, case_id, viewer, representation, fps in specs:
        viewer_path = (
            ROOT / "outputs/model_card_native_viewers" / viewer / "index.html"
        )
        if source not in attachments["videos"] or not viewer_path.is_file():
            continue
        jobs.append(
            RenderJob(
                source=source,
                method=method,
                label=label,
                case_id=case_id,
                viewer=viewer_path,
                representation=representation,
                fps=fps,
            )
        )
    return jobs


def _jobs() -> list[RenderJob]:
    by_source = {
        job.source: job
        for job in [
            *_t2m_jobs(),
            *_sidecar_jobs(),
            *_music_to_dance_jobs(),
            *_motioncanvas_jobs(),
            *_native_jobs(),
        ]
    }
    return [by_source[source] for source in sorted(by_source)]


def _local_viewer(viewer_url: str) -> Union[str, Path]:
    routes = {
        "t2m-humanml3d-leaderboard": (
            ROOT / "docs/leaderboards/hf_space_t2m_humanml3d/cases/index.html"
        ),
        "music-to-dance-aistpp-leaderboard": (
            ROOT / "docs/leaderboards/hf_space_music_to_dance/cases/index.html"
        ),
        "babel-sequential-generation-leaderboard": (
            ROOT / "docs/leaderboards/hf_space_babel_sequential/cases/index.html"
        ),
        "temporal-condition-leaderboard": (
            ROOT / "docs/leaderboards/hf_space_temporal_condition"
        ),
        "motion-repair-brokenamass-leaderboard": (
            ROOT / "docs/leaderboards/hf_space_motion_repair/cases/index.html"
        ),
    }
    for marker, local in routes.items():
        if marker not in viewer_url:
            continue
        if local.is_file():
            return local
        suffix = urlsplit(viewer_url).path.split("/cases/", 1)[-1]
        return local / "cases" / suffix
    return viewer_url


def _command(job: RenderJob, output_root: Path) -> list[str]:
    command = [
        sys.executable,
        str(CAPTURE_TOOL),
        "--viewer" if isinstance(job.viewer, Path) else "--url",
        str(job.viewer),
        "--method",
        job.method,
        "--label",
        job.label,
        "--case",
        job.case_id,
        "--layout",
        job.layout,
        "--output",
        str(job.output(output_root)),
        "--width",
        "512",
        "--height",
        "512",
        "--frames",
        "10000",
        "--frame-step",
        "1",
        "--fps",
        str(job.fps),
        "--representation",
        job.representation,
        "--show-input-condition",
        "--sync-canvas-buffer",
    ]
    if job.include_audio:
        command.append("--include-audio")
    if job.audio_track:
        command.extend(["--audio-track", job.audio_track])
    return command


def _render(
    job: RenderJob,
    output_root: Path,
    overwrite: bool,
    adopt_existing: bool,
) -> tuple[str, str]:
    output = job.output(output_root)
    metadata = output.with_suffix(".render.json")
    if output.is_file() and metadata.is_file() and not overwrite:
        try:
            _audit_one(job.source, output_root)
        except Exception:
            pass
        else:
            return job.source, "cached"
    if output.is_file() and adopt_existing and not overwrite:
        frames, duration = imageio_ffmpeg.count_frames_and_secs(str(output))
        metadata.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "render_backend": "threejs",
                    "render_profile": "motius-threejs-floor-v1",
                    "method": job.method,
                    "label": job.label,
                    "case_id": job.case_id,
                    "source_url": str(job.viewer),
                    "frames": frames,
                    "fps": round(frames / duration),
                    "frame_step": 1,
                    "width": 512,
                    "height": 512,
                    "representation": job.representation,
                    "floor": "matte light floor with neutral gray grid",
                    "audio": (
                        {"embedded": True}
                        if job.include_audio
                        else None
                    ),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return job.source, "adopted"
    output.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        _command(job, output_root),
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    return job.source, "rendered"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/model_card_video_uploads",
    )
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--adopt-existing", action="store_true")
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--match")
    args = parser.parse_args()

    jobs = _native_jobs() if args.native_only else _jobs()
    if args.match:
        jobs = [job for job in jobs if args.match in job.source]
    if args.limit is not None:
        jobs = jobs[: args.limit]
    print(f"Rendering {len(jobs)} Three.js Model Card video(s)", flush=True)
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as executor:
        pending = {
            executor.submit(
                _render,
                job,
                args.output_root,
                args.overwrite,
                args.adopt_existing,
            ): job
            for job in jobs
        }
        for future in as_completed(pending):
            job = pending[future]
            try:
                source, status = future.result()
                print(f"{status}: {source}", flush=True)
            except Exception as exc:
                failures.append(f"{job.source}: {exc}")
                print(f"failed: {job.source}: {exc}", flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} render(s) failed:\n" + "\n".join(failures)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
