from __future__ import annotations
import numpy as np
import cv2
class YOLOXDetector:
    """YOLOX person detector using ONNX Runtime.

    Parameters
    ----------
    onnx_path : str or None
        Path to YOLOX ONNX model.  If None, auto-downloads YOLOX-X.
    input_size : tuple
        Model input resolution (H, W).  Default (640, 640).
    score_thr : float
        Minimum person confidence threshold.
    nms_thr : float
        NMS IoU threshold.
    device : str
        "cuda" or "cpu".
    """

    def __init__(self,onnx_path,input_size=(640,640),score_thr=0.5,nms_thr=0.45,device='cuda'):
        from motius.models.vision.gem_x import GemXDetectorBundle
        self.bundle=GemXDetectorBundle(onnx_path,device)
        self.input_size=input_size;self.score_thr=score_thr;self.nms_thr=nms_thr

    def detect(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Detect persons in a single BGR image.

        Returns
        -------
        boxes : ndarray (N, 4)  — xyxy in original image coords
        scores : ndarray (N,)
        """
        padded, ratio = self._preprocess(img_bgr)
        inp = padded.transpose(2, 0, 1)[None].astype(np.float32)
        outputs = self.bundle(inp)
        return self._postprocess(outputs[0], ratio)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float]:
        h, w = img.shape[:2]
        ratio = min(self.input_size[0] / h, self.input_size[1] / w)
        resized = cv2.resize(img, (int(w * ratio), int(h * ratio)), interpolation=cv2.INTER_LINEAR)
        padded = np.full((self.input_size[0], self.input_size[1], 3), 114, dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        return (padded, ratio)

    def _postprocess(self, outputs: np.ndarray, ratio: float) -> tuple[np.ndarray, np.ndarray]:
        if outputs.shape[-1] == 5:
            boxes = outputs[0, :, :4] / ratio
            scores = outputs[0, :, 4]
            mask = scores > self.score_thr
            return (boxes[mask], scores[mask])
        strides = [8, 16, 32]
        grids, exp_strides = ([], [])
        for s in strides:
            hs, ws = (self.input_size[0] // s, self.input_size[1] // s)
            xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
            grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
            grids.append(grid)
            exp_strides.append(np.full((*grid.shape[:2], 1), s))
        grids = np.concatenate(grids, 1)
        exp_strides = np.concatenate(exp_strides, 1)
        outputs[..., :2] = (outputs[..., :2] + grids) * exp_strides
        outputs[..., 2:4] = np.exp(outputs[..., 2:4]) * exp_strides
        preds = outputs[0]
        cx, cy, bw, bh = (preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3])
        x1 = (cx - bw / 2) / ratio
        y1 = (cy - bh / 2) / ratio
        x2 = (cx + bw / 2) / ratio
        y2 = (cy + bh / 2) / ratio
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        obj_conf = preds[:, 4]
        cls_scores = preds[:, 5:]
        person_score = obj_conf * cls_scores[:, 0]
        mask = person_score > self.score_thr
        boxes_f = boxes_xyxy[mask]
        scores_f = person_score[mask]
        keep = _nms(boxes_f, scores_f, self.nms_thr)
        return (boxes_f[keep], scores_f[keep])

def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    """Standard greedy NMS on (N, 4) xyxy boxes."""
    x1, y1, x2, y2 = (boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3])
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(iou <= thr)[0] + 1]
    return keep

def _iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between (M, 4) and (N, 4) xyxy boxes → (M, N)."""
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-07)

class _KalmanBoxTracker:
    """Tracks a single object with a Kalman filter over [x, y, a, h] state.

    State vector: [cx, cy, aspect_ratio, height, vx, vy, va, vh].
    Measurement:  [cx, cy, aspect_ratio, height].
    """
    _count = 0

    def __init__(self, bbox: np.ndarray, score: float):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        a = w / (h + 1e-07)
        self.x = np.array([cx, cy, a, h, 0, 0, 0, 0], dtype=np.float64)
        self.P = np.eye(8) * 10.0
        self.P[4:, 4:] *= 100.0
        self.Q = np.eye(8) * 1.0
        self.Q[4:, 4:] *= 0.01
        self.R = np.diag([1.0, 1.0, 10.0, 1.0])
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.zeros((4, 8))
        self.H[:4, :4] = np.eye(4)
        self.id = _KalmanBoxTracker._count
        _KalmanBoxTracker._count += 1
        self.hits = 1
        self.time_since_update = 0
        self.score = score

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.time_since_update += 1
        return self._state_to_xyxy()

    def update(self, bbox: np.ndarray, score: float):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        z = np.array([cx, cy, w / (h + 1e-07), h])
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P
        self.hits += 1
        self.time_since_update = 0
        self.score = score

    def _state_to_xyxy(self) -> np.ndarray:
        cx, cy, a, h = self.x[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

class ByteTracker:
    """ByteTrack multi-object tracker (MIT license).

    Two-stage association:
      1. Match high-score detections to existing tracks (IoU).
      2. Match remaining low-score detections to unmatched tracks (IoU).
    No appearance model — pure motion-based, very fast.

    Parameters
    ----------
    high_thr : float
        Score threshold to split detections into high / low groups.
    match_thr : float
        IoU threshold for matching.
    max_lost : int
        Frames to keep a lost track before removal.
    """

    def __init__(self, high_thr: float=0.5, match_thr: float=0.3, max_lost: int=30):
        self.high_thr = high_thr
        self.match_thr = match_thr
        self.max_lost = max_lost
        self.tracks: list[_KalmanBoxTracker] = []
        _KalmanBoxTracker._count = 0

    def update(self, boxes: np.ndarray, scores: np.ndarray) -> list[tuple[np.ndarray, int, float]]:
        """Process one frame of detections.

        Parameters
        ----------
        boxes : (N, 4) xyxy
        scores : (N,)

        Returns
        -------
        results : list of (box_xyxy, track_id, score)
        """
        for t in self.tracks:
            t.predict()
        high_mask = scores >= self.high_thr
        low_mask = ~high_mask & (scores > 0.1)
        high_boxes, high_scores = (boxes[high_mask], scores[high_mask])
        low_boxes, low_scores = (boxes[low_mask], scores[low_mask])
        matched_t, matched_d, unmatched_tracks, unmatched_dets = self._match(self.tracks, high_boxes, self.match_thr)
        for t_idx, d_idx in zip(matched_t, matched_d):
            self.tracks[t_idx].update(high_boxes[d_idx], high_scores[d_idx])
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]
        if len(remaining_tracks) > 0 and len(low_boxes) > 0:
            mt2, md2, still_unmatched, _ = self._match(remaining_tracks, low_boxes, self.match_thr)
            for t_idx, d_idx in zip(mt2, md2):
                remaining_tracks[t_idx].update(low_boxes[d_idx], low_scores[d_idx])
            unmatched_tracks = [unmatched_tracks[i] for i in still_unmatched]
        for d_idx in unmatched_dets:
            self.tracks.append(_KalmanBoxTracker(high_boxes[d_idx], high_scores[d_idx]))
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost]
        results = []
        for t in self.tracks:
            if t.time_since_update == 0:
                results.append((t._state_to_xyxy(), t.id, t.score))
        return results

    @staticmethod
    def _match(tracks: list[_KalmanBoxTracker], boxes: np.ndarray, thr: float) -> tuple[list[int], list[int], list[int], list[int]]:
        """Hungarian matching between tracks and detections."""
        if len(tracks) == 0 or len(boxes) == 0:
            return ([], [], list(range(len(tracks))), list(range(len(boxes))))
        from scipy.optimize import linear_sum_assignment
        track_boxes = np.array([t._state_to_xyxy() for t in tracks])
        iou_matrix = _iou_batch(track_boxes, boxes)
        cost = 1.0 - iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_t, matched_d = ([], [])
        unmatched_t = set(range(len(tracks)))
        unmatched_d = set(range(len(boxes)))
        for r, c in zip(row_ind, col_ind):
            if iou_matrix[r, c] >= thr:
                matched_t.append(r)
                matched_d.append(c)
                unmatched_t.discard(r)
                unmatched_d.discard(c)
        return (matched_t, matched_d, sorted(unmatched_t), sorted(unmatched_d))

def detect_and_track(frames: np.ndarray, detector: YOLOXDetector, *, detect_every: int=1, tracker_kwargs: dict | None=None) -> tuple[np.ndarray, np.ndarray]:
    """Run YOLOX detection + ByteTrack on a video and return the primary
    person's bounding boxes.

    Parameters
    ----------
    frames : (T, H, W, 3) uint8 RGB array
    detector : YOLOXDetector instance
    detect_every : int
        Run detection every N frames.  Intermediate frames use Kalman
        prediction only (faster but less accurate for fast motion).
        Set to 1 to detect every frame.
    tracker_kwargs : dict
        Extra kwargs for ByteTracker.

    Returns
    -------
    bbx_xyxy : (T, 4) float32 — primary person boxes (xyxy)
    track_ids : (T,) int — track ID per frame (-1 if no detection)
    """
    from tqdm import tqdm
    T, H, W, _ = frames.shape
    tracker = ByteTracker(**tracker_kwargs or {})
    per_frame: list[list[tuple[np.ndarray, int, float]]] = []
    for i in tqdm(range(T), desc='YOLOX+ByteTrack'):
        if i % detect_every == 0:
            img_bgr = frames[i][..., ::-1].copy()
            boxes, scores = detector.detect(img_bgr)
        else:
            boxes, scores = (np.empty((0, 4)), np.empty((0,)))
        results = tracker.update(boxes, scores)
        per_frame.append(results)
    return _select_primary_track(per_frame, T, H, W)

def _select_primary_track(per_frame: list[list[tuple[np.ndarray, int, float]]], T: int, H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Pick the dominant track and extract per-frame boxes."""
    from collections import defaultdict
    track_area: dict = defaultdict(float)
    track_frames: dict = defaultdict(list)
    for t, results in enumerate(per_frame):
        for box, tid, score in results:
            area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
            track_area[tid] += area
            track_frames[tid].append((t, box, score))
    if not track_area:
        default = np.array([0.0, 0.0, W - 1, H - 1], dtype=np.float32)
        return (np.tile(default, (T, 1)), np.full(T, -1, dtype=np.int32))
    primary_id = max(track_area, key=track_area.get)
    primary_dict = {t: box for t, box, _ in track_frames[primary_id]}
    bbx_xyxy = np.zeros((T, 4), dtype=np.float32)
    track_ids = np.full(T, -1, dtype=np.int32)
    for t in range(T):
        if t in primary_dict:
            bbx_xyxy[t] = primary_dict[t]
            track_ids[t] = primary_id
    _interpolate_gaps(bbx_xyxy, track_ids, primary_id, H, W)
    bbx_xyxy[:, [0, 2]] = np.clip(bbx_xyxy[:, [0, 2]], 0, W - 1)
    bbx_xyxy[:, [1, 3]] = np.clip(bbx_xyxy[:, [1, 3]], 0, H - 1)
    return (bbx_xyxy, track_ids)

def _interpolate_gaps(bbx: np.ndarray, ids: np.ndarray, primary_id: int, H: int, W: int):
    """Fill gaps in the primary track via linear interpolation."""
    T = len(bbx)
    valid = ids == primary_id
    if valid.sum() == 0:
        bbx[:] = [0, 0, W - 1, H - 1]
        return
    valid_idx = np.where(valid)[0]
    if valid_idx[0] > 0:
        bbx[:valid_idx[0]] = bbx[valid_idx[0]]
        ids[:valid_idx[0]] = primary_id
    if valid_idx[-1] < T - 1:
        bbx[valid_idx[-1] + 1:] = bbx[valid_idx[-1]]
        ids[valid_idx[-1] + 1:] = primary_id
    for k in range(len(valid_idx) - 1):
        a, b = (valid_idx[k], valid_idx[k + 1])
        if b - a > 1:
            for t in range(a + 1, b):
                alpha = (t - a) / (b - a)
                bbx[t] = (1 - alpha) * bbx[a] + alpha * bbx[b]
                ids[t] = primary_id
