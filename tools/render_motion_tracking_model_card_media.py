#!/usr/bin/env python3
"""Render unified Three.js Model Card videos from tracking Leaderboards."""

from __future__ import annotations

import argparse
import base64
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright

try:
    from tools.capture_leaderboard_method_gif import _serve
except ModuleNotFoundError:
    from capture_leaderboard_method_gif import _serve


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs/model_card_video_uploads"
SPECS = {
    "any2track": ("mujoco", "Any2Track"),
    "protomotions": ("mujoco", "ProtoMotions"),
    "humanoid_gpt": ("mujoco", "HumanoidGPT"),
    "sonic": ("isaaclab", "SONIC"),
    "beyondmimic": ("isaaclab", "BeyondMimic"),
}


def _paths(package: str) -> tuple[Path, Path, Path]:
    stem = f"{package}_motion_tracking_512_30fps"
    source = ROOT / "assets/model_zoo" / package / f"{stem}.gif"
    video = OUTPUT_ROOT / source.relative_to(ROOT).with_suffix(".mp4")
    metadata = video.with_suffix(".render.json")
    return source, video, metadata


def _write_gif(video: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video),
        "-filter_complex",
        (
            "fps=30,scale=1024:-1:flags=lanczos,split[s0][s1];"
            "[s0]palettegen=max_colors=192[p];"
            "[s1][p]paletteuse=dither=bayer:bayer_scale=3"
        ),
        str(output),
    ]
    subprocess.run(command, check=True)


def _local_viewer(
    engine: str,
    label: str,
    temporary: Path,
) -> Path:
    source = (
        ROOT
        / "docs/leaderboards"
        / f"hf_space_motion_tracking_{engine}"
        / "cases"
    )
    shutil.copy2(source / "index.html", temporary / "index.html")
    shutil.copy2(source / "robot.json", temporary / "robot.json")
    shutil.copytree(source / "three", temporary / "three")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    manifest["columns"] = [
        column
        for column in manifest["columns"]
        if column["label"] in {"GT reference", label}
    ]
    manifest["asset_base_url"] = "assets/"
    manifest["mesh_base_url"] = "meshes/"
    (temporary / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    (temporary / "assets").symlink_to(
        ROOT / "outputs/publication/motion_tracking_g1_v1" / engine,
        target_is_directory=True,
    )
    mesh_source = (
        ROOT / "motius/models/kimodo/network/assets/skeletons/"
        "g1skel34/meshes/g1"
    )
    mesh_output = temporary / "meshes"
    mesh_output.mkdir()
    for path in mesh_source.glob("*"):
        if path.is_file():
            (mesh_output / path.name.lower()).symlink_to(path)
    return temporary / "index.html"


def _canvas_image(data_url: str, label: str) -> np.ndarray:
    payload = base64.b64decode(data_url.split(",", 1)[1])
    canvas = Image.open(io.BytesIO(payload)).convert("RGB").resize(
        (512, 430),
        Image.Resampling.LANCZOS,
    )
    panel = Image.new("RGB", (512, 472), "white")
    panel.paste(canvas, (0, 42))
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default(size=18)
    draw.text((16, 12), label, fill=(23, 35, 31), font=font)
    return np.asarray(panel)


def _render(package: str, engine: str, label: str, frames: int) -> None:
    source, video, metadata = _paths(package)
    video.parent.mkdir(parents=True, exist_ok=True)
    images = []
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        viewer = _local_viewer(engine, label, temporary)
        server = _serve(viewer.parent)
        url = (
            f"http://127.0.0.1:{server.server_address[1]}/{viewer.name}"
        )
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                page = browser.new_page(
                    viewport={"width": 1072, "height": 620},
                    device_scale_factor=1,
                )
                page.on(
                    "pageerror",
                    lambda error: print(
                        f"[browser error] {error}",
                        file=sys.stderr,
                        flush=True,
                    ),
                )
                page.on(
                    "requestfailed",
                    lambda request: print(
                        f"[request failed] {request.url}: "
                        f"{request.failure}",
                        file=sys.stderr,
                        flush=True,
                    ),
                )

                def preserve_canvas(route):
                    response = route.fetch()
                    body = response.text().replace(
                        (
                            "new THREE.WebGLRenderer({canvas:this.canvas, "
                            'antialias:true, powerPreference:"high-performance"})'
                        ),
                        (
                            "new THREE.WebGLRenderer({canvas:this.canvas, "
                            "antialias:true, powerPreference:"
                            '"high-performance",preserveDrawingBuffer:true})'
                        ),
                    )
                    body = body.replace(
                        (
                            "      views.forEach((view) => view.render());\n"
                            "    }\n\n"
                            "    Promise.all"
                        ),
                        (
                            "      if (playing) "
                            "views.forEach((view) => view.render());\n"
                            "    }\n\n"
                            "    Promise.all"
                        ),
                    )
                    route.fulfill(response=response, body=body)

                page.route("**/index.html", preserve_canvas)
                page.goto(url, wait_until="domcontentloaded", timeout=120_000)
                try:
                    page.wait_for_function(
                        """
                        label => {
                          const views = [...document.querySelectorAll(".view")];
                          const target = views.find(view =>
                            view.querySelector(".view-head strong")
                              ?.textContent.trim() === label
                          );
                          const reference = views.find(view =>
                            view.querySelector(".view-head strong")
                              ?.textContent.trim() === "GT reference"
                          );
                          return target && reference &&
                            !target.querySelector(".status")?.textContent
                              .includes("Preparing") &&
                            !target.querySelector(".status")?.textContent
                              .includes("Loading") &&
                            !reference.querySelector(".status")?.textContent
                              .includes("Preparing") &&
                            !reference.querySelector(".status")?.textContent
                              .includes("Loading");
                        }
                        """,
                        arg=label,
                        timeout=15_000,
                    )
                except PlaywrightTimeoutError:
                    state = page.evaluate(
                        """
                        () => ({
                          caseLabel: document.querySelector("#case-label")
                            ?.textContent,
                          statuses: [...document.querySelectorAll(".status")]
                            .map(node => node.textContent),
                          views: [...document.querySelectorAll(
                            ".view-head strong"
                          )].map(node => node.textContent),
                        })
                        """
                    )
                    raise RuntimeError(
                        f"Tracking viewer did not become ready: {state}"
                    )
                page.locator("#play").click()
                page.add_style_tag(
                    content="""
                .toolbar,.timeline,.case-meta,.status{display:none!important}
                .viewer{padding:0!important}
                .views{width:1024px!important;grid-template-columns:repeat(2,512px)!important;gap:0!important}
                .view{display:none;width:512px!important;border-radius:0!important}
                .view.keep{display:block!important}
                canvas{width:512px!important;height:430px!important;aspect-ratio:auto!important;min-height:0!important}
                """
                )
                page.evaluate(
                    """
                label => {
                  for (const view of document.querySelectorAll(".view")) {
                    const name = view.querySelector(".view-head strong")
                      ?.textContent.trim();
                    view.classList.toggle(
                      "keep",
                      name === "GT reference" || name === label
                    );
                  }
                }
                """,
                    label,
                )
                for output_frame in range(frames):
                    source_frame = round(output_frame * 50 / 30)
                    data_urls = page.evaluate(
                        """
                        ({value, label}) => {
                          const element = document.querySelector("#scrub");
                          element.value = String(value);
                          element.dispatchEvent(
                            new Event("input", {bubbles: true})
                          );
                          const selected = [...document.querySelectorAll(".view")]
                            .filter(view => {
                              const name = view.querySelector(".view-head strong")
                                ?.textContent.trim();
                              return name === "GT reference" || name === label;
                            });
                          return selected.map(view =>
                            view.querySelector("canvas").toDataURL("image/png")
                          );
                        }
                        """,
                        {"value": source_frame, "label": label},
                    )
                    panels = [
                        _canvas_image(data_urls[0], "GT reference"),
                        _canvas_image(data_urls[1], label),
                    ]
                    images.append(np.concatenate(panels, axis=1))
                browser.close()
        finally:
            server.shutdown()
            server.server_close()

    writer = imageio.get_writer(
        video,
        fps=30,
        codec="libx264",
        quality=8,
        pixelformat="yuv420p",
        macro_block_size=2,
    )
    try:
        for frame in images:
            writer.append_data(frame)
    finally:
        writer.close()
    metadata.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "render_backend": "threejs",
                "render_profile": "motius-threejs-floor-v1",
                "method": package,
                "label": label,
                "case_id": "dance1_subject1",
                "frames": len(images),
                "fps": 30,
                "frame_step": 1,
                "representation": "Unitree G1 mesh",
                "floor": "neutral gray grid",
                "source_clock_hz": 50,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_gif(video, source)
    print(source)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "packages",
        nargs="*",
        choices=sorted(SPECS),
        default=sorted(SPECS),
    )
    parser.add_argument("--frames", type=int, default=121)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    for package in args.packages:
        _render(package, *SPECS[package], frames=args.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
