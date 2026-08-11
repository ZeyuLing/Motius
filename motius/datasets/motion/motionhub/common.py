import json
import re
from pathlib import Path
from typing import Any

import torch
from mmengine.dataset import Compose


hm3d_pattern = re.compile(
    r"^[^#]+#"
    r"(?:[\w\-']+\/[A-Za-z]+(?:\s[\w\-']+\/[A-Za-z]+)*)"
    r"#(?:-?\d+\.\d+|nan)"
    r"#(?:-?\d+\.\d+|nan)$",
    re.IGNORECASE,
)


def read_json(path: str) -> Any:
    with Path(path).open('r', encoding='utf-8') as f:
        return json.load(f)


def read_txt(path: str) -> str:
    return Path(path).read_text(encoding='utf-8')


def convert_to_tensor(value: Any):
    if isinstance(value, torch.Tensor):
        return value
    if value is None:
        return None
    return torch.as_tensor(value)
