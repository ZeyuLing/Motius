#!/usr/bin/env python3
"""Apply Motius' deterministic device fix to a pinned PromptHMR runtime."""

from __future__ import annotations

import argparse
from pathlib import Path


_DETECTRON2_WRAPPER = Path("pipeline/utils_detectron2.py")
_ORIGINAL = """\
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}
"""
_PATCHED = """\
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image = image.to(self.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}
"""


def patch_runtime(runtime_root: str | Path, *, restore: bool = False) -> Path:
    """Patch or restore the pinned runtime, failing on source drift."""

    root = Path(runtime_root).expanduser().resolve()
    wrapper = root / _DETECTRON2_WRAPPER
    if not wrapper.is_file():
        raise FileNotFoundError(
            f"PromptHMR Detectron2 wrapper was not found: {wrapper}"
        )

    source, destination = (
        (_PATCHED, _ORIGINAL) if restore else (_ORIGINAL, _PATCHED)
    )
    text = wrapper.read_text()
    if destination in text:
        return wrapper
    if source not in text:
        raise RuntimeError(
            "Pinned PromptHMR Detectron2 wrapper changed; refusing a fuzzy "
            "runtime patch."
        )
    wrapper.write_text(text.replace(source, destination, 1))
    return wrapper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    print(patch_runtime(args.runtime_root, restore=args.restore))


if __name__ == "__main__":
    main()
