#!/usr/bin/env python3
"""Serve a local monocular gallery with HTTP byte-range support."""

from __future__ import annotations

import argparse
import functools
import http.server
import re
import shutil
from pathlib import Path


class RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler with standards-compliant single byte ranges."""

    _range: tuple[int, int] | None = None

    def send_head(self):
        path = Path(self.translate_path(self.path))
        if path.is_dir():
            return super().send_head()
        try:
            source = path.open("rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        stat = path.stat()
        size = stat.st_size
        start, end = 0, max(0, size - 1)
        status = 200
        header = self.headers.get("Range")
        if header:
            match = re.fullmatch(r"bytes=(\d*)-(\d*)", header.strip())
            if not match:
                source.close()
                self.send_error(416, "Invalid byte range")
                return None
            raw_start, raw_end = match.groups()
            if raw_start:
                start = int(raw_start)
                end = int(raw_end) if raw_end else end
            elif raw_end:
                length = int(raw_end)
                start = max(0, size - length)
            if start >= size or start > end:
                source.close()
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return None
            end = min(end, size - 1)
            status = 206

        self._range = (start, end) if status == 206 else None
        self.send_response(status)
        self.send_header("Content-Type", self.guess_type(str(path)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Last-Modified", self.date_time_string(stat.st_mtime))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if path.suffix.lower() in {".html", ".json", ".joints"}:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        return source

    def copyfile(self, source, outputfile):
        if self._range is None:
            shutil.copyfileobj(source, outputfile)
            return
        start, end = self._range
        source.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = source.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            try:
                outputfile.write(chunk)
            except (BrokenPipeError, ConnectionResetError):
                break
            remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    handler = functools.partial(RangeRequestHandler, directory=str(directory))
    server = http.server.ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Serving {directory} at http://{args.host}:{args.port}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
