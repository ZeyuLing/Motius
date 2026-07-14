#!/usr/bin/env python3
"""Capture a Motius Three.js demo page into a README-friendly GIF."""

from __future__ import annotations

import argparse
import http.server
import socketserver
import tempfile
import threading
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):  # noqa: A002
        return


def _serve(directory: Path):
    handler = lambda *args, **kwargs: _QuietHandler(*args, directory=str(directory), **kwargs)
    server = socketserver.TCPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


def capture(args: argparse.Namespace) -> None:
    demo_dir = args.viewer.resolve().parent
    server = _serve(demo_dir)
    url = f"http://127.0.0.1:{server.server_address[1]}/{args.viewer.name}"
    frames = []
    try:
        with tempfile.TemporaryDirectory() as tmpdir, sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": args.width, "height": args.height}, device_scale_factor=1)
            page.goto(url, wait_until="networkidle")
            page.wait_for_function("window.__MOTIUS_READY__ === true", timeout=30000)
            page.evaluate(
                """
                () => {
                  document.querySelector('.controls')?.remove();
                  window.__MOTIUS_DEMO__?.setFrame(0);
                }
                """
            )
            for frame in range(args.frames):
                page.evaluate("frame => window.__MOTIUS_DEMO__.setFrame(frame)", frame)
                page.evaluate("() => new Promise(requestAnimationFrame)")
                png = Path(tmpdir) / f"{frame:04d}.png"
                page.screenshot(path=str(png), type="png")
                image = Image.open(png).convert("RGB")
                frames.append(np.asarray(image))
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.output, frames, fps=args.fps, loop=0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("viewer", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--frames", type=int, default=72)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    capture(parse_args())
