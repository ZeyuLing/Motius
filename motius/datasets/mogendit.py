"""Original MoreDiff .pt motion dataset with explicit paths and validation."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from motius.registry import DATASETS
from motius.motion.representation.mogendit.motion_representation import OccamMotionRep


@DATASETS.register_module()
class MoGenDiTDataset(Dataset):
    """Each .pt file contains local ``pose`` matrices, relative ``joint``, ``trans``.

    Same random start range (leaving >=60 frames when available), normalization
    and zero padding as AdHocMotionData. No stale pickle cache or silent skips.
    """
    def __init__(self, data_roots, *, global_pose=True, fps=30, fix_len=224, min_len=30):
        if not 2 <= fix_len <= 5000 or min_len < 2:
            raise ValueError('Require 2 <= fix_len <= 5000 and min_len >= 2')
        if isinstance(data_roots, (str, Path)):
            data_roots = [data_roots]
        self.motion_rep = OccamMotionRep(global_pose=global_pose, fps=fps)
        self.fix_len = fix_len
        self.file_paths = []
        signatures = []
        candidates = set()
        for root in data_roots:
            root = Path(root).resolve()
            if not root.is_dir():
                raise FileNotFoundError(f'Dataset directory does not exist: {root}')
            candidates.update(root.rglob('*.pt'))
        for path in sorted(candidates):
            record = self._load(path)
            if len(record['pose']) < min_len:
                continue
            self.file_paths.append(path)
            stat = path.stat()
            signatures.append((str(path), stat.st_size, stat.st_mtime_ns))
        if not self.file_paths:
            raise ValueError('No valid training .pt clips meet min_len')
        self.fingerprint = hashlib.sha256(json.dumps(signatures).encode()).hexdigest()

    @staticmethod
    def _load(path):
        data = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(data, dict) or not {'pose', 'joint', 'trans'} <= data.keys():
            raise ValueError(f'{path}: expected pose, joint and trans tensors')
        result = {k: torch.as_tensor(data[k], dtype=torch.float32) for k in ('pose','joint','trans')}
        length = len(result['pose'])
        for key, shape in [('pose',(length,22,3,3)),('joint',(length,22,3)),('trans',(length,3))]:
            if result[key].shape != shape or not torch.isfinite(result[key]).all():
                raise ValueError(f'{path}: {key} must be finite with shape {shape}')
        return result

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        data = self._load(self.file_paths[index])
        max_start = max(0, len(data['pose']) - 60)
        start = np.random.randint(0, max_start + 1) if max_start else 0
        length = min(self.fix_len, len(data['pose']) - start)
        data = {k: v[start:start+length] for k,v in data.items()}
        motion = self.motion_rep.encode(**data)
        motion = self.motion_rep.normalization(motion, height_reset=True)
        if length < self.fix_len:
            motion = torch.cat([motion, motion.new_zeros(self.fix_len-length,201)])
        return {'motion': motion, 'length': length}
