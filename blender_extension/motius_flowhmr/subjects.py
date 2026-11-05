# SPDX-License-Identifier: GPL-3.0-or-later
"""Auditable subject selection, independent of Blender and ML dependencies."""
import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_subjects(path):
    """Resolve portable preview/video paths relative to the detection manifest."""
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    for key in ('video', 'preview_sequence'):
        if data.get(key):
            value = Path(data[key])
            data[key] = str(value if value.is_absolute() else path.parent/value)
    return data


def candidates(data, frame):
    """Only observed boxes are selectable; never invent a person in a gap."""
    result = []
    for track in data['tracks']:
        try:
            i = track['frames'].index(frame)
        except ValueError:
            continue
        result.append((track['id'], track['boxes'][i]))
    return result


def hit_test(boxes, point):
    x, y = point
    hits = [(tid, box) for tid, box in boxes if box[0] <= x <= box[2] and box[1] <= y <= box[3]]
    # Small foreground subjects remain reachable inside a larger bounding box.
    return [tid for tid, box in sorted(hits, key=lambda item: (item[1][2]-item[1][0])*(item[1][3]-item[1][1]))]


def selected_boxes(path, video, track_id, max_gap=10):
    import numpy as np
    data = load_subjects(path)
    if data['schema'] != 'motius.flowhmr.subjects.v1' or data['video_sha256'] != digest(video):
        raise ValueError('Subject selection belongs to another video; detect people again')
    track = next((t for t in data['tracks'] if t['id'] == track_id), None)
    if track is None:
        raise ValueError('Choose a detected subject before generating motion')
    frames = np.asarray(track['frames'], dtype=int)
    boxes = np.asarray(track['boxes'], dtype=np.float32)
    n = int(data['frames'])
    if not len(frames) or boxes.shape != (len(frames), 4) or not np.isfinite(boxes).all():
        raise ValueError('Malformed subject trajectory')
    if np.any(np.diff(frames) <= 0) or frames[0] < 0 or frames[-1] >= n or np.any(boxes[:, 2:] <= boxes[:, :2]):
        raise ValueError('Invalid subject frame order or bounding boxes')
    if frames[0] > 0 or frames[-1] < n-1 or np.any(np.diff(frames) > max_gap+1):
        raise ValueError('Selected subject leaves view or tracking is interrupted. Trim to a continuous visible segment and detect again; no identity substitution was made.')
    return np.stack([np.interp(np.arange(n), frames, boxes[:, j]) for j in range(4)], -1).astype(np.float32), data


def verify_cached_subject(selected, cached, data, track_id):
    """Allow detector round-off only when cached ROIs unambiguously cover the chosen track.

    Features keep their EXACT original boxes; they are never relabelled onto new boxes.
    Competing detections overlapping the cached ROI veto reuse.
    """
    import numpy as np
    cached = np.asarray(cached)
    def iou(a,b):
        lo=np.maximum(a[..., :2],b[..., :2]);hi=np.minimum(a[..., 2:],b[..., 2:])
        inter=np.maximum(0,hi-lo).prod(-1)
        return inter/np.maximum(1e-9,(a[...,2:]-a[...,:2]).prod(-1)+(b[...,2:]-b[...,:2]).prod(-1)-inter)
    if cached.shape != selected.shape or not np.isfinite(cached).all():
        raise ValueError('Cached subject has a different frame range')
    overlap=iou(selected,cached)
    if overlap.min() < .85:
        raise ValueError('Cached features belong to different subject boxes. Disable preprocessing reuse.')
    for other in data['tracks']:
        if other['id']==track_id: continue
        if np.any(iou(np.asarray(other['boxes']),cached[np.asarray(other['frames'])])>.25):
            raise ValueError('Overlapping people make cached identity ambiguous. Disable preprocessing reuse.')
    return dict(minimum_iou=float(overlap.min()), mean_iou=float(overlap.mean()), exact_feature_boxes_preserved=True)
