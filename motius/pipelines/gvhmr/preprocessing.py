"""GVHMR image, track and camera input preparation."""
import numpy as np

def _bbox_xys_from_xyxy(bbox_xyxy: np.ndarray) -> np.ndarray:
    """Match GVHMR's official 192:256, 1.2x bbox conversion."""

    boxes = np.asarray(bbox_xyxy, dtype=np.float32)
    if (
        boxes.ndim != 2
        or boxes.shape[1] != 4
        or not np.isfinite(boxes).all()
        or np.any(boxes[:, 2:] <= boxes[:, :2])
    ):
        raise ValueError("bbox_xyxy must have finite shape (frames,4) and positive area.")
    center = (boxes[:, :2] + boxes[:, 2:]) * 0.5
    width = boxes[:, 2] - boxes[:, 0]
    height = boxes[:, 3] - boxes[:, 1]
    aspect_ratio = 192.0 / 256.0
    wide = width > aspect_ratio * height
    height = height.copy()
    width = width.copy()
    height[wide] = width[wide] / aspect_ratio
    tall = width < aspect_ratio * height
    width[tall] = height[tall] * aspect_ratio
    size = np.maximum(height, width) * 1.2
    return np.concatenate((center, size[:, None]), axis=1).astype(np.float32)
