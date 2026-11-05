"""FlowHMR checkpoint normalization buffers, shared with artifact conversion."""
import json
from pathlib import Path
import torch


def load_motion_statistics(path):
    data = json.loads(Path(path).read_text())
    result = {}
    for name, key, shape in (
        ('smooth_root_vel', 'smooth_root_vel', None),
        ('smooth_root_pos', 'smooth_root_pos', None),
        ('local_joints_positions', 'local_joints_positions', (52, 3)),
        ('global_rot_data', 'global_rot_data', (52, 6)),
        ('local_rot_data', 'local_rot_data', (52, 6)),
        ('rep_shapes', 'shapes', None),
        ('rep_foot_contacts', 'foot_contacts', None),
    ):
        for kind in ('mean', 'std'):
            value = torch.FloatTensor(data[key][kind])
            if shape:
                value = value.reshape(*shape)
            if kind == 'std':
                value = value.clamp(min=1e-5)
            result[f'{name}_{kind}'] = value[None, None]
    return result
