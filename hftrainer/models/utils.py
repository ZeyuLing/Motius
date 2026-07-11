"""Small motion-task utilities shared by PRISM and VerMo integration."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mmengine import print_log


def print_colored_log(message: str, level: str = 'INFO'):
    """Compatibility wrapper used by migrated motion code."""
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    print_log(message, logger='current', level=level)


def write_json(obj: Any, path: str):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_txt(text: str, path: str):
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open('w', encoding='utf-8') as f:
        f.write(text)
