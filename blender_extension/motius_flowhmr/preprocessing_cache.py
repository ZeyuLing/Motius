# SPDX-License-Identifier: GPL-3.0-or-later
"""Verified reuse of per-frame preprocessing; motion sampling always runs."""
import hashlib
import json
from pathlib import Path


def load(directory, video, mode, camera_interval, frontend_files=None):
    import numpy as np
    import torch
    directory = Path(directory)
    manifest = json.loads((directory/'frontend-cache.json').read_text())
    if manifest.get('schema') != 'motius.flowhmr.frontend.v1':
        raise ValueError('Unsupported preprocessing cache manifest')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    if manifest['video_sha256'] != sha(video):
        raise ValueError('Preprocessing cache belongs to a different video')
    if manifest['mode'] != mode or manifest['camera_interval'] != camera_interval:
        raise ValueError('Cache mode / camera interval differ from the selected settings')
    for name, checksum in manifest['files'].items():
        if Path(name).name != name or sha(directory/name) != checksum:
            raise ValueError('Preprocessing cache failed integrity check: '+name)
    if frontend_files is not None:
        if set(manifest.get('frontend_weights', {})) != set(frontend_files):
            raise ValueError('Cache must record the identities of all frontend weights')
        for name, path in frontend_files.items():
            if not path or sha(path) != manifest['frontend_weights'][name]:
                raise ValueError('Cache frontend weights do not match: '+name)
    feature = torch.load(directory/'features.pt', weights_only=True, map_location='cpu')
    boxes = torch.from_numpy(np.load(directory/'boxes.npy', allow_pickle=False))
    with np.load(directory/'camera.npz', allow_pickle=False) as camera:
        rt, intrinsics = camera['camera_RT'].copy(), camera['K'].copy()
    if feature.shape != (manifest['frames'], 3072) or len(boxes) != len(feature) or len(rt) != len(feature):
        raise ValueError('Preprocessing cache has inconsistent frame counts')
    if not all(torch.isfinite(torch.as_tensor(value)).all() for value in (feature, boxes, rt, intrinsics)):
        raise ValueError('Preprocessing cache contains nonfinite values')

    class Frontend:
        def prepare_bboxes(self, video, supplied, **kwargs):
            return supplied, None

        def extract_features(self, video, bbox_xyxy, camera_K, **kwargs):
            return feature

    return dict(preprocessor=Frontend(), camera_RT=rt, camera_K=intrinsics, bbox_xyxy=boxes, transcode=False)
