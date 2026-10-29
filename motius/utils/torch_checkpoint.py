"""Torch checkpoint loading for both original pickle and native ZIP formats."""
from pathlib import Path
import torch


def load_torch_checkpoint(path,*,map_location='cpu',weights_only=True,mmap=True):
    path=Path(path)
    with path.open('rb') as stream:is_zip=stream.read(4)==b'PK\x03\x04'
    return torch.load(path,map_location=map_location,weights_only=weights_only,mmap=mmap and is_zip)
