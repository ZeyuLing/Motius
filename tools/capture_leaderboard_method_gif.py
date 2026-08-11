#!/usr/bin/env python3
"""Capture one Motius Three.js method scene as a full-frame MP4 or GIF."""

from __future__ import annotations

import argparse
import base64
import http.server
from io import BytesIO
import json
import os
from pathlib import Path
import socketserver
import subprocess
import tempfile
import threading
from typing import Optional
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import imageio.v2 as imageio
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[1]


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return


def _serve(directory: Path):
    handler = lambda *args, **kwargs: _QuietHandler(
        *args,
        directory=str(directory),
        **kwargs,
    )
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def _duration_ms(frames: int, fps: int) -> list[int]:
    # GIF delays are stored in centiseconds. Quantize cumulative timestamps so
    # the 30/40 ms cadence averages to the requested 30 fps instead of every
    # 33 ms frame being rounded down to 30 ms (33.3 fps).
    boundaries = [
        round(index * 100 / fps) * 10
        for index in range(frames + 1)
    ]
    return [
        max(10, boundaries[index + 1] - boundaries[index])
        for index in range(frames)
    ]


def _sample_durations_ms(
    source_frames: list[int],
    source_fps: int,
    maximum: int,
) -> list[int]:
    """Preserve source time when the capture skips viewer frames."""
    if source_fps < 1:
        raise ValueError("fps must be >= 1")
    if not source_frames:
        return []
    if len(source_frames) > 1:
        sample_step = source_frames[-1] - source_frames[-2]
    else:
        sample_step = 1
    terminal_frame = min(maximum + 1, source_frames[-1] + sample_step)
    start_frame = source_frames[0]
    boundaries = [
        round((frame - start_frame) * 100 / source_fps) * 10
        for frame in [*source_frames, terminal_frame]
    ]
    return [
        max(10, boundaries[index + 1] - boundaries[index])
        for index in range(len(source_frames))
    ]


def _source_frame_indices(
    maximum: int,
    start_frame: int,
    frame_step: int,
    limit: int,
) -> list[int]:
    if frame_step < 1:
        raise ValueError("frame_step must be >= 1")
    if limit < 1:
        raise ValueError("frames must be >= 1")
    start = max(0, int(start_frame))
    if start > maximum:
        return []
    available = (maximum - start) // frame_step + 1
    return [
        start + offset * frame_step
        for offset in range(min(limit, available))
    ]


def _crop_frame(image: Image.Image, width: int, height: int) -> Image.Image:
    if image.width < width or image.height < height:
        raise ValueError(
            f"Captured frame {image.size} is smaller than {(width, height)}"
        )
    return image.crop((0, 0, width, height))


def _font(size: int, bold: bool = False):
    name = (
        "/usr/share/fonts/truetype/dejavu/"
        f"DejaVuSans-{'Bold' if bold else 'Book'}.ttf"
    )
    try:
        return ImageFont.truetype(name, size)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    value: str,
    font,
    width: int,
    maximum_lines: int,
) -> list[str]:
    words = str(value).split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == maximum_lines:
            break
    if current and len(lines) < maximum_lines:
        lines.append(current)
    return lines


def _paint_overlays(
    image: Image.Image,
    label: str,
    overlay: dict | None,
) -> Image.Image:
    draw = ImageDraw.Draw(image, "RGBA")
    label_font = _font(12, bold=True)
    label_box = draw.textbbox((0, 0), label, font=label_font)
    label_width = label_box[2] - label_box[0]
    draw.rounded_rectangle(
        (14, 14, 30 + label_width, 38),
        radius=4,
        fill=(255, 255, 255, 232),
        outline=(255, 255, 255, 245),
    )
    draw.text((22, 19), label, font=label_font, fill=(23, 35, 30, 255))
    if not overlay:
        return image

    primary = str(overlay.get("primary", "")).strip()
    secondary = str(overlay.get("secondary", "")).strip()
    body_font = _font(12, bold=True)
    secondary_font = _font(11)
    label_small = _font(10, bold=True)
    lines = _wrap_text(draw, primary, body_font, image.width - 40, 3)
    body_height = max(18, len(lines) * 16)
    has_timeline = bool(overlay.get("segments"))
    height = 48 + body_height
    if secondary:
        height += 17
    if has_timeline:
        height += 13
    top = image.height - height - 10
    draw.rounded_rectangle(
        (10, top, image.width - 10, image.height - 10),
        radius=5,
        fill=(255, 255, 255, 238),
        outline=(255, 255, 255, 248),
    )
    draw.text(
        (20, top + 10),
        "INPUT CONDITION",
        font=label_small,
        fill=(102, 115, 109, 255),
    )
    cursor = top + 28
    for line in lines:
        draw.text(
            (20, cursor),
            line,
            font=body_font,
            fill=(23, 35, 30, 255),
        )
        cursor += 16
    if secondary:
        draw.text(
            (20, cursor + 1),
            secondary,
            font=secondary_font,
            fill=(88, 101, 95, 255),
        )
        cursor += 18
    if has_timeline:
        left = 20
        right = image.width - 20
        width = right - left
        draw.rounded_rectangle(
            (left, cursor + 3, right, cursor + 10),
            radius=2,
            fill=(220, 228, 224, 255),
        )
        for segment in overlay["segments"]:
            start = left + width * float(segment.get("start", 0))
            end = left + width * float(segment.get("end", 0))
            color = segment.get("color") or "#315f9d"
            try:
                fill = (*ImageColor.getrgb(color), 255)
            except ValueError:
                fill = (49, 95, 157, 255)
            draw.rectangle(
                (start, cursor + 3, max(start + 1, end), cursor + 10),
                fill=fill,
            )
        playhead = left + width * float(overlay.get("playhead", 0))
        draw.rectangle(
            (playhead - 1, cursor + 1, playhead + 1, cursor + 12),
            fill=(23, 35, 30, 255),
        )
    return image


def _canvas_image(
    page,
    target,
    width: int,
    height: int,
    label: str,
    canvas_buffer: bool,
    data_url: Optional[str] = None,
):
    canvas = target.locator("canvas").first
    shared_canvas = canvas.count() == 0
    if shared_canvas:
        canvas = page.locator("#canvas").first
    if data_url is not None or canvas_buffer:
        if data_url is None:
            data_url = canvas.evaluate("canvas => canvas.toDataURL('image/png')")
        _, payload = data_url.split(",", 1)
        encoded = base64.b64decode(payload)
    else:
        encoded = canvas.screenshot(type="png")
    image = Image.open(BytesIO(encoded)).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.LANCZOS)
    if shared_canvas:
        ImageDraw.Draw(image).rectangle(
            (0, 0, image.width, 38),
            fill=(244, 246, 248),
        )
    overlay = page.evaluate(
        """
        () => {
          const root = document.querySelector(".motius-demo-input");
          if (!root) return null;
          const timeline = root.querySelector(".motius-demo-timeline");
          const segments = timeline
            ? [...timeline.querySelectorAll(
                ".motius-demo-timeline-segment"
              )].map(element => {
                const style = getComputedStyle(element);
                const left = parseFloat(element.style.left || "0") / 100;
                const width = parseFloat(element.style.width || "100") / 100;
                return {
                  start: left,
                  end: Math.min(1, left + width),
                  color: style.backgroundColor,
                };
              })
            : [];
          const playhead = root.querySelector(".motius-demo-playhead");
          return {
            primary: root.querySelector(
              ".motius-demo-input-primary"
            )?.textContent || "",
            secondary: root.querySelector(
              ".motius-demo-input-secondary"
            )?.textContent || "",
            segments,
            playhead: parseFloat(playhead?.style.left || "0") / 100,
          };
        }
        """
    )
    return _paint_overlays(image, label, overlay)


def _validate_output(output: Path, frame_step: int, include_audio: bool) -> None:
    suffix = output.suffix.lower()
    if suffix not in {".gif", ".mp4"}:
        raise ValueError("output must use a .gif or .mp4 suffix")
    if suffix == ".mp4" and frame_step != 1:
        raise ValueError(
            "MP4 capture requires --frame-step 1 so every source frame is "
            "rendered instead of duplicated"
        )
    if include_audio and suffix != ".mp4":
        raise ValueError("audio is only supported for MP4 output")


def _audio_spec(page, track_key: str | None) -> dict | None:
    return page.evaluate(
        """
        async trackKey => {
          const manifestUrl = new URL("manifest.json", location.href);
          const manifest = await fetch(manifestUrl).then(response => {
            if (!response.ok) throw new Error(
              `Unable to load demo manifest: ${response.status}`
            );
            return response.json();
          });
          const caseId = document.querySelector("#case-id")
            ?.textContent.trim();
          const item = manifest.cases.find(candidate =>
            [candidate.case_id, candidate.case_key, candidate.sample_id]
              .filter(Boolean)
              .map(String)
              .includes(caseId)
          );
          if (!item) throw new Error(`Unable to resolve demo case ${caseId}`);
          const base = manifest.asset_base_url
            ? new URL(manifest.asset_base_url, manifestUrl)
            : manifestUrl;
          if (trackKey) {
            const track = (item.audio_tracks || []).find(
              candidate => candidate.key === trackKey
            );
            if (!track) return null;
            return {
              url: new URL(track.asset, base).href,
              start_seconds: Number(track.start_seconds || 0),
              end_seconds: Number(track.end_seconds || 0),
            };
          }
          if (!item.audio) return null;
          return {
            url: new URL(item.audio, base).href,
            start_seconds: Number(item.audio_start_seconds || 0),
            end_seconds: Number(item.audio_end_seconds || 0),
          };
        }
        """,
        track_key,
    )


def _download(url: str, output: Path) -> None:
    request = Request(
        url,
        headers={"User-Agent": "Motius-model-card-media-renderer"},
    )
    with urlopen(request, timeout=120) as response:
        output.write_bytes(response.read())


def _write_mp4(
    output: Path,
    images: list[np.ndarray],
    fps: int,
    audio: dict | None,
) -> None:
    silent = output.with_name(f"{output.stem}.silent.mp4")
    imageio.mimwrite(
        silent,
        images,
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=1,
        output_params=["-movflags", "+faststart"],
    )
    if audio is None:
        silent.replace(output)
        return
    suffix = Path(str(audio["url"]).split("?", 1)[0]).suffix or ".audio"
    source = output.with_name(f"{output.stem}{suffix}")
    try:
        _download(str(audio["url"]), source)
        video_duration = len(images) / fps
        start = max(0.0, float(audio.get("start_seconds", 0)))
        declared_end = float(audio.get("end_seconds", 0))
        available = (
            max(0.0, declared_end - start)
            if declared_end > start
            else video_duration
        )
        duration = min(video_duration, available)
        command = [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(silent),
            "-ss",
            f"{start:.6f}",
            "-i",
            str(source),
            "-t",
            f"{duration:.6f}",
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(output),
        ]
        subprocess.run(command, check=True)
    finally:
        silent.unlink(missing_ok=True)
        source.unlink(missing_ok=True)


def _install_input_overlay(page, tile) -> None:
    page.add_style_tag(
        content="""
        .motius-demo-input {
          position: absolute;
          left: 10px;
          right: 10px;
          bottom: 10px;
          z-index: 4;
          display: grid;
          gap: 6px;
          padding: 9px 10px 10px;
          border: 1px solid rgba(255, 255, 255, .82);
          border-radius: 5px;
          color: #17231e;
          background: rgba(255, 255, 255, .92);
          box-shadow: 0 4px 18px rgba(20, 36, 30, .13);
          backdrop-filter: blur(9px);
          font: 12px/1.35 ui-sans-serif, system-ui, sans-serif;
          visibility: hidden;
        }
        .motius-demo-input-label {
          color: #66736d;
          font: 700 10px/1 ui-monospace, monospace;
          text-transform: uppercase;
        }
        .motius-demo-input-primary {
          overflow: hidden;
          min-height: 32px;
          max-height: 48px;
          font-weight: 650;
        }
        .motius-demo-input-secondary {
          overflow: hidden;
          color: #58655f;
          font-size: 11px;
          white-space: nowrap;
          text-overflow: ellipsis;
        }
        .motius-demo-timeline {
          position: relative;
          display: flex;
          width: 100%;
          height: 7px;
          overflow: hidden;
          border-radius: 2px;
          background: #dce4e0;
        }
        .motius-demo-timeline-segment {
          height: 100%;
        }
        .motius-demo-playhead {
          position: absolute;
          top: -2px;
          bottom: -2px;
          width: 2px;
          border-radius: 1px;
          background: #17231e;
          transform: translateX(-1px);
        }
        """
    )
    tile.evaluate(
        """
        async tile => {
          const manifestUrl = new URL("manifest.json", location.href);
          const manifest = await fetch(manifestUrl).then(response => {
            if (!response.ok) throw new Error(
              `Unable to load demo manifest: ${response.status}`
            );
            return response.json();
          });
          const caseId = document.querySelector("#case-id")
            ?.textContent.trim();
          const item = manifest.cases.find(candidate =>
            [candidate.case_id, candidate.case_key, candidate.sample_id]
              .filter(Boolean)
              .map(String)
              .includes(caseId)
          );
          if (!item) throw new Error(`Unable to resolve demo case ${caseId}`);

          const totalFrames = Number(
            document.querySelector("#timeline")?.max || 0
          ) + 1;
          const references = item.references || [];
          const segments = item.segments || [];
          const intervals = item.condition_intervals || [];
          const colors = [
            "#315f9d",
            "#d95f02",
            "#287147",
            "#6d4ea2",
            "#a5412e",
            "#087d72",
          ];

          const overlay = document.createElement("div");
          overlay.className = "motius-demo-input";
          const label = document.createElement("div");
          label.className = "motius-demo-input-label";
          label.textContent = "Input condition";
          const primary = document.createElement("div");
          primary.className = "motius-demo-input-primary";
          const secondary = document.createElement("div");
          secondary.className = "motius-demo-input-secondary";
          const timeline = document.createElement("div");
          timeline.className = "motius-demo-timeline";
          const playhead = document.createElement("div");
          playhead.className = "motius-demo-playhead";

          if (segments.length) {
            for (const [index, segment] of segments.entries()) {
              const bar = document.createElement("div");
              bar.className = "motius-demo-timeline-segment";
              bar.style.width = `${
                100 * (Number(segment.end_frame) -
                  Number(segment.start_frame)) / totalFrames
              }%`;
              bar.style.background = segment.color || colors[index % colors.length];
              timeline.append(bar);
            }
          } else {
            const base = document.createElement("div");
            base.className = "motius-demo-timeline-segment";
            base.style.width = "100%";
            base.style.background = "#a5412e";
            timeline.append(base);
            for (const interval of intervals) {
              const bar = document.createElement("div");
              bar.className = "motius-demo-timeline-segment";
              bar.style.position = "absolute";
              bar.style.left = `${100 * Number(interval[0]) / totalFrames}%`;
              bar.style.width = `${
                100 * (Number(interval[1]) - Number(interval[0])) / totalFrames
              }%`;
              bar.style.background =
                manifest.condition_legend?.conditioned?.color || "#d95f02";
              timeline.append(bar);
            }
          }
          timeline.append(playhead);
          overlay.append(label, primary);
          if (segments.length || references.length > 1 || intervals.length) {
            overlay.append(secondary, timeline);
          }
          tile.append(overlay);

          const textOf = value => typeof value === "string"
            ? value
            : value?.caption || "";
          const update = frame => {
            const value = Math.max(0, Math.min(Number(frame), totalFrames - 1));
            playhead.style.left = `${
              100 * value / Math.max(1, totalFrames - 1)
            }%`;
            if (segments.length) {
              let index = segments.findIndex(segment =>
                value >= Number(segment.start_frame) &&
                value < Number(segment.end_frame)
              );
              if (index < 0) index = segments.length - 1;
              primary.textContent = segments[index]?.caption || "";
              secondary.textContent =
                `Segment ${index + 1} / ${segments.length}`;
            } else {
              primary.textContent = textOf(references[0]) || "Motion input";
              if (references.length > 1) {
                secondary.textContent = references.slice(1)
                  .map(textOf)
                  .filter(Boolean)
                  .join(" · ");
              } else if (intervals.length) {
                const conditioned = intervals.some(interval =>
                  value >= Number(interval[0]) &&
                  value < Number(interval[1])
                );
                secondary.textContent = conditioned
                  ? "Condition frame"
                  : "Generated frame";
              }
            }
          };
          globalThis.__MOTIUS_CAPTURE_OVERLAY__ = {update};
          update(0);
        }
        """
    )


def _install_render_route(page, unify_scene: bool) -> None:
    replacements = {
        "new THREE.WebGLRenderer({canvas,antialias:true,alpha:false})": (
            "new THREE.WebGLRenderer({canvas,antialias:true,"
            "alpha:false,preserveDrawingBuffer:true})"
        ),
        "new THREE.WebGLRenderer({canvas,antialias:true})": (
            "new THREE.WebGLRenderer({canvas,antialias:true,"
            "preserveDrawingBuffer:true})"
        ),
    }
    if unify_scene:
        replacements.update(
            {
                "0xe8eeeb": "0xf4f6f8",
                "0xf7faf8": "0xe9edf2",
                "0xa9b7b0": "0xb7c1ce",
                "0xd7dfdb": "0xd8dee7",
            }
        )

    def handle(route):
        response = route.fetch()
        body = response.text()
        for source, target in replacements.items():
            body = body.replace(source, target)
        route.fulfill(response=response, body=body)

    page.route("**/cases/index.html*", handle)


def capture(args: argparse.Namespace) -> None:
    server = None
    if args.url:
        base = args.url
    else:
        viewer = args.viewer.expanduser().resolve()
        server = _serve(viewer.parent)
        base = (
            f"http://127.0.0.1:{server.server_address[1]}/"
            f"{viewer.name}"
        )
    query = {"method": args.method}
    if args.case:
        query["case"] = args.case
    url = base + ("&" if "?" in base else "?") + urlencode(query)

    output = args.output.expanduser().resolve()
    output.relative_to((ROOT / "outputs").resolve())
    _validate_output(output, args.frame_step, args.include_audio)
    output.parent.mkdir(parents=True, exist_ok=True)
    images: list[np.ndarray] = []
    audio = None
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            with sync_playwright() as playwright:
                launch_options = {"headless": True}
                chromium = os.environ.get("MOTIUS_CHROMIUM_EXECUTABLE")
                if chromium:
                    launch_options["executable_path"] = chromium
                browser = playwright.chromium.launch(**launch_options)
                page = browser.new_page(
                    viewport={
                        "width": args.width + 48,
                        "height": args.height + 240,
                    },
                    device_scale_factor=1,
                )
                _install_render_route(
                    page,
                    unify_scene=args.layout == "stage",
                )
                page.goto(
                    url,
                    wait_until="domcontentloaded",
                    timeout=args.timeout_ms,
                )
                if args.layout == "tile":
                    target = page.locator(
                        ".tile",
                        has=page.locator(".method", has_text=args.label),
                    ).first
                    target.scroll_into_view_if_needed(timeout=args.timeout_ms)
                    page.wait_for_function(
                        """
                        label => [...document.querySelectorAll('.tile')].some(
                          tile =>
                            tile.dataset.loadState === 'ready' &&
                            tile.querySelector('.method')
                              ?.textContent.includes(label)
                        )
                        """,
                        arg=args.label,
                        timeout=args.timeout_ms,
                    )
                    target.evaluate(
                        """
                        tile => {
                          const gallery = tile.closest(".gallery");
                          for (const sibling of gallery?.querySelectorAll(
                            ":scope > .tile"
                          ) || []) {
                            if (sibling !== tile) sibling.style.display = "none";
                          }
                        }
                        """
                    )
                    page.add_style_tag(
                        content=f"""
                        header, aside, .player, .case-copy, .loading {{
                          display: none !important;
                        }}
                        body, main {{ overflow: visible !important; }}
                        .gallery-wrap {{ padding: 0 !important; }}
                        .gallery {{
                          display: block !important;
                          width: {args.width}px !important;
                        }}
                        .tile {{
                          width: {args.width}px !important;
                          height: {args.height}px !important;
                        }}
                        .tile:not(:has(.method)) {{
                          display: none !important;
                        }}
                        .tile-actions {{ display: none !important; }}
                        """
                    )
                else:
                    target = page.locator(".stage").first
                    page.wait_for_function(
                        """
                        () => {
                          const loading = document.querySelector("#loading");
                          return loading && loading.hidden;
                        }
                        """,
                        timeout=args.timeout_ms,
                    )
                    page.add_style_tag(
                        content=f"""
                        .topbar, .panel, .loading, .stage-label {{
                          display: none !important;
                        }}
                        body, main, .workspace {{
                          width: {args.width}px !important;
                          height: {args.height}px !important;
                          overflow: visible !important;
                        }}
                        .workspace {{
                          display: block !important;
                        }}
                        .stage {{
                          position: relative !important;
                          width: {args.width}px !important;
                          height: {args.height}px !important;
                        }}
                        """
                    )
                    target.evaluate(
                        """
                        (element, label) => {
                          const badge = document.createElement("div");
                          badge.className = "method";
                          badge.textContent = label;
                          badge.style.cssText = [
                            "position:absolute",
                            "z-index:4",
                            "top:14px",
                            "left:14px",
                            "padding:5px 8px",
                            "border:1px solid rgba(255,255,255,.82)",
                            "border-radius:4px",
                            "background:rgba(255,255,255,.9)",
                            "color:#17231e",
                            "font:750 12px/1 system-ui,sans-serif",
                          ].join(";");
                          element.append(badge);
                        }
                        """,
                        args.label,
                    )
                target.scroll_into_view_if_needed()
                if args.show_input_condition:
                    _install_input_overlay(page, target)
                if args.include_audio:
                    audio = _audio_spec(page, args.audio_track)
                    if audio is None:
                        target = (
                            f" track {args.audio_track!r}"
                            if args.audio_track
                            else ""
                        )
                        raise RuntimeError(
                            f"Viewer case exposes no requested audio{target}"
                        )
                timeline = page.locator("#timeline")
                maximum = int(timeline.get_attribute("max") or "0")
                source_frames = _source_frame_indices(
                    maximum,
                    args.start_frame,
                    args.frame_step,
                    args.frames,
                )
                if len(source_frames) < 2:
                    raise RuntimeError(
                        f"Viewer exposes only {len(source_frames)} frame(s)"
                    )
                for source_frame in source_frames:
                    timeline.evaluate(
                        """
                        (element, frame) => {
                          element.value = String(frame);
                          element.dispatchEvent(
                            new Event('input', {bubbles: true})
                          );
                        }
                        """,
                        source_frame,
                    )
                    if args.show_input_condition:
                        page.evaluate(
                            """
                            frame => globalThis
                              .__MOTIUS_CAPTURE_OVERLAY__
                              ?.update(frame)
                            """,
                            source_frame,
                        )
                    data_url = None
                    if args.sync_canvas_buffer:
                        data_url = target.evaluate(
                            """
                            (tile, frame) => {
                              const timeline =
                                document.querySelector("#timeline");
                              timeline.value = String(frame);
                              timeline.dispatchEvent(
                                new Event("input", {bubbles: true})
                              );
                              globalThis.__MOTIUS_CAPTURE_OVERLAY__
                                ?.update(frame);
                              const canvas =
                                tile.querySelector("canvas") ||
                                document.querySelector("#canvas");
                              return canvas.toDataURL("image/png");
                            }
                            """,
                            source_frame,
                        )
                    else:
                        page.evaluate(
                            "() => new Promise(resolve => "
                            "requestAnimationFrame(() => "
                            "requestAnimationFrame(resolve)))"
                        )
                    images.append(
                        np.asarray(
                            _canvas_image(
                                page,
                                target,
                            args.width,
                            args.height,
                            args.label,
                            args.canvas_buffer,
                            data_url,
                        )
                        )
                    )
                browser.close()
    finally:
        if server is not None:
            server.shutdown()
            server.server_close()

    if output.suffix.lower() == ".mp4":
        _write_mp4(output, images, args.fps, audio)
        render_metadata = {
            "schema_version": 1,
            "render_backend": "threejs",
            "render_profile": "motius-threejs-floor-v1",
            "method": args.method,
            "label": args.label,
            "case_id": args.case,
            "source_url": url,
            "frames": len(images),
            "fps": args.fps,
            "frame_step": args.frame_step,
            "width": args.width,
            "height": args.height,
            "representation": args.representation,
            "capture_mode": (
                "synchronous-webgl-buffer"
                if args.sync_canvas_buffer
                else "webgl-buffer"
                if args.canvas_buffer
                else "browser-compositor"
            ),
            "floor": "matte light floor with neutral gray grid",
            "audio": audio,
        }
        output.with_suffix(".render.json").write_text(
            json.dumps(render_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        imageio.mimsave(
            output,
            images,
            duration=_sample_durations_ms(
                source_frames,
                args.fps,
                maximum,
            ),
            loop=0,
            subrectangles=True,
        )
    print(output.relative_to(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--viewer", type=Path)
    source.add_argument("--url")
    parser.add_argument("--method", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--case")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout",
        choices=("tile", "stage"),
        default="tile",
        help="Leaderboard scene layout to capture.",
    )
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--frames", type=int, default=90)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help=(
            "Source viewer timeline FPS. Output timestamps preserve this "
            "clock even when --frame-step skips rendered frames."
        ),
    )
    parser.add_argument("--timeout-ms", type=int, default=300000)
    parser.add_argument(
        "--include-audio",
        action="store_true",
        help="Mux the case audio declared by the viewer manifest into MP4.",
    )
    parser.add_argument(
        "--audio-track",
        help=(
            "For multi-track manifests, select the audio track key. "
            "Without this option, use the case-level input audio."
        ),
    )
    parser.add_argument(
        "--representation",
        default="smpl",
        help="Representation shown by the captured Three.js scene.",
    )
    parser.add_argument(
        "--show-input-condition",
        action="store_true",
        help=(
            "Overlay the exact text/temporal/segment condition from the "
            "leaderboard manifest."
        ),
    )
    parser.add_argument(
        "--canvas-buffer",
        action="store_true",
        help=(
            "Read an explicitly preserved WebGL buffer. Use only for local "
            "viewers that create the renderer with preserveDrawingBuffer."
        ),
    )
    parser.add_argument(
        "--sync-canvas-buffer",
        action="store_true",
        help=(
            "Dispatch the timeline update and read the WebGL buffer in one "
            "browser task. Intended for canonical Leaderboard viewers whose "
            "input handler renders synchronously."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    capture(parse_args())
