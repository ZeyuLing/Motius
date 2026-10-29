"""Detection selection, prompt construction and track assembly in native modules."""
import cv2
import numpy as np
import torch
from scipy.interpolate import make_interp_spline,interp1d
from .segmentation import segment_people

def batched_mask_to_box(masks: torch.Tensor) -> torch.Tensor:
    """
    Calculates boxes in XYXY format around masks. Return [0,0,0,0] for
    an empty mask. For input shape C1xC2x...xHxW, the output shape is C1xC2x...x4.
    """
    if torch.numel(masks) == 0:
        return torch.zeros(*masks.shape[:-2], 4, device=masks.device)
    shape = masks.shape
    h, w = shape[-2:]
    if len(shape) > 2:
        masks = masks.flatten(0, -3)
    else:
        masks = masks.unsqueeze(0)
    in_height, _ = torch.max(masks, dim=-1)
    in_height_coords = in_height * torch.arange(h, device=in_height.device)[None, :]
    bottom_edges, _ = torch.max(in_height_coords, dim=-1)
    in_height_coords = in_height_coords + h * ~in_height
    top_edges, _ = torch.min(in_height_coords, dim=-1)
    in_width, _ = torch.max(masks, dim=-2)
    in_width_coords = in_width * torch.arange(w, device=in_width.device)[None, :]
    right_edges, _ = torch.max(in_width_coords, dim=-1)
    in_width_coords = in_width_coords + w * ~in_width
    left_edges, _ = torch.min(in_width_coords, dim=-1)
    empty_filter = (right_edges < left_edges) | (bottom_edges < top_edges)
    out = torch.stack([left_edges, top_edges, right_edges, bottom_edges], dim=-1)
    out = out * (~empty_filter).unsqueeze(-1)
    if len(shape) > 2:
        out = out.reshape(*shape[:-2], 4)
    else:
        out = out[0]
    return out

def nms_oks(kp_predictions, rois, thresh):
    """Nms based on kp predictions."""
    scores = np.mean(kp_predictions[:, 2, :], axis=1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(kp_predictions[i], rois[i], kp_predictions[order[1:]], rois[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """
    sigmas = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0
    vars = (sigmas * 2) ** 2
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)
    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]
    e = (dx ** 2 + dy ** 2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / e.shape[1]
    return e

def interpolate_bboxes(bboxes, frames, masks, fn='linear'):
    """
    bboxes: numpy array of shape (len(frames), 4) representing the bounding boxes in x, y, x, y format
    frames: example [0, 1, 2, 8, 9, 10] -> here the frames 3-7 are missing and should be interpolated
    """
    all_frames = np.arange(frames[0], frames[-1] + 1)
    if fn == 'spline':
        interp_bboxes = make_interp_spline(frames, bboxes, k=3)(all_frames)
    elif fn == 'linear':
        interp_func = interp1d(frames, bboxes, axis=0, kind='linear')
        interp_bboxes = interp_func(all_frames)
    else:
        raise ValueError("Invalid interpolation function. Choose 'spline' or 'linear'.")
    all_masks = np.full([len(all_frames), masks.shape[1], masks.shape[2]], False, dtype=bool)
    indices = frames - all_frames[0]
    all_masks[indices] = masks
    return (interp_bboxes, all_frames, all_masks)

def detect_and_track(images, detector, predictor, segmenter, *, num_max_people=10, height_thresh=0.3, score_thresh=0.4, det_thresh=0.5, bbox_interp=False, filter_ng_points=False, kp_thres=0.1):
    detector_type = 'detectron2'
    masks = segment_people(segmenter, images)
    start_frame = -1
    for i in range(len(images)):
        if isinstance(images[i], str):
            img_cv2 = cv2.imread(images[i])
        elif isinstance(images[i], np.ndarray):
            img_cv2 = images[i][..., ::-1]
        if detector_type == 'detectron2':
            det_out = detector(img_cv2)
            keypoints = det_out['instances'].pred_keypoints.cpu().numpy()
            scores = det_out['instances'].scores.cpu().numpy()
            boxes = det_out['instances'].pred_boxes.tensor.cpu().numpy()
        if len(scores) > 0:
            keep = nms_oks(keypoints.transpose(0, 2, 1), boxes, 0.5)
            keypoints = keypoints[keep]
            scores = scores[keep]
            boxes = boxes[keep]
            conf = scores > det_thresh
            keypoints = keypoints[conf]
            scores = scores[conf]
            boxes = boxes[conf]
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            bbox_height = boxes[:, 3] - boxes[:, 1]
            normalized_areas = areas / np.max(areas)
            normalized_scores = scores / np.max(scores)
            combined_score = 0.5 * normalized_areas + 0.5 * normalized_scores
            print(f'Combined score: {combined_score}')
            best_idxs = [x for x in np.argsort(combined_score)[::-1] if combined_score[x] > score_thresh]
            if len(best_idxs) > 1:
                largest_person_idx = np.argmax(bbox_height)
                largest_person_height = bbox_height[largest_person_idx]
                print(f'number of people before filtering (height < {height_thresh} max): {len(best_idxs)}')
                best_idxs = [x for x in best_idxs if bbox_height[x] > largest_person_height * height_thresh]
                print(f'number of people after filtering (height < {height_thresh} max): {len(best_idxs)}')
            best_boxes = boxes[best_idxs]
            best_scores = scores[best_idxs]
            best_keypoints = keypoints[best_idxs]
            best_combined_scores = combined_score[best_idxs]
            start_frame = i
            print(f'Best person scores: {best_scores}')
            print(f'Best person combined scores: {best_combined_scores}')
            break
        else:
            print('No persons detected in the image.')
            continue
    if start_frame == -1:
        raise ValueError('No persons detected in any of the images.')
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        if isinstance(images[0], str):
            state = predictor.init_state(video_path=os.path.dirname(images[0]), async_loading_frames=False, offload_video_to_cpu=True)
        elif isinstance(images[0], np.ndarray):
            state = predictor.init_state(video_frames=images, async_loading_frames=False, offload_video_to_cpu=True)
        for idx in range(len(best_boxes)):
            pos_kp = best_keypoints[idx][best_keypoints[idx, :, 2] > kp_thres]
            neg_kp = []
            if filter_ng_points:
                for pid in range(len(best_boxes)):
                    if pid != idx:
                        neg_p_kp = best_keypoints[pid][best_keypoints[pid, :, 2] > kp_thres]
                        x1, y1, x2, y2 = best_boxes[idx]
                        mask = (neg_p_kp[:, 0] >= x1) & (neg_p_kp[:, 0] <= x2) & (neg_p_kp[:, 1] >= y1) & (neg_p_kp[:, 1] <= y2)
                        if mask.any():
                            neg_kp.append(neg_p_kp[mask])
            if len(neg_kp) > 0:
                neg_kp = np.concatenate(neg_kp, axis=0)
                box_p = best_boxes[idx].reshape(-1, 2)
                box_labels = np.array([2, 3])
                points = np.concatenate([pos_kp[:, :2], neg_kp[:, :2], box_p], axis=0)
                point_labels = np.concatenate([np.ones(len(pos_kp)), np.zeros(len(neg_kp)), box_labels]).astype(np.int64)
            else:
                points = np.concatenate([pos_kp[:, :2], best_boxes[idx].reshape(-1, 2)], axis=0)
                point_labels = np.concatenate([np.ones(len(pos_kp)), np.array([2, 3])]).astype(np.int64)
            _, out_id, msk = predictor.add_new_points(state, frame_idx=start_frame, obj_id=idx + 1, labels=point_labels, points=points)
        track_results = {}
        for t, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
            boxes = batched_mask_to_box(out_mask_logits > 0.0)
            for obj_id, box, obj_msk in zip(out_obj_ids, boxes, out_mask_logits):
                if obj_id not in track_results:
                    track_results[obj_id] = {'track_id': obj_id, 'bbox_format': 'x1y1x2y2', 'frames': [], 'bboxes': [], 'masks': []}
                track_results[obj_id]['frames'].append(t)
                obj_msk = (obj_msk[0] > 0.0).cpu().numpy().astype(np.bool_)
                track_results[obj_id]['bboxes'].append(box[0].cpu().numpy())
                track_results[obj_id]['masks'].append(obj_msk)
        for k in track_results.keys():
            track_results[k]['frames'] = np.array(track_results[k]['frames'])
            track_results[k]['bboxes'] = np.array(track_results[k]['bboxes'])
            track_results[k]['masks'] = np.array(track_results[k]['masks'])
            bboxes = track_results[k]['bboxes']
            frames = track_results[k]['frames']
            obj_masks = track_results[k]['masks']
            bb_heights = bboxes[:, 3] - bboxes[:, 1]
            bb_widths = bboxes[:, 2] - bboxes[:, 0]
            non_zero_indices = np.where(np.logical_and(bb_heights > 0, bb_widths > 0))[0]
            if len(non_zero_indices) > 0:
                start_index = non_zero_indices[0]
                end_index = non_zero_indices[-1] + 1
                frames = frames[start_index:end_index]
                bboxes = bboxes[start_index:end_index]
                obj_masks = obj_masks[start_index:end_index]
                bb_heights = bb_heights[start_index:end_index]
                bb_widths = bb_widths[start_index:end_index]
            bboxes = bboxes[np.logical_and(bb_heights > 10, bb_widths > 10)]
            frames = frames[np.logical_and(bb_heights > 10, bb_widths > 10)]
            obj_masks = obj_masks[np.logical_and(bb_heights > 10, bb_widths > 10)]
            if len(bboxes) == 0 or len(frames) == 0:
                print(f'No valid bboxes or frames for track {k}')
                continue
            if bbox_interp:
                interp_bboxes, interp_frames, interp_masks = interpolate_bboxes(bboxes, frames, obj_masks, fn='linear')
            else:
                interp_bboxes, interp_frames, interp_masks = (bboxes, frames, obj_masks)
            track_results[k]['frames'] = interp_frames
            track_results[k]['bboxes'] = interp_bboxes
            track_results[k]['masks'] = interp_masks
            track_results[k]['detected'] = np.sum(interp_masks, axis=(1, 2)) > 1
    track_results = dict(sorted(track_results.items(), key=lambda x: len(x[1]['frames']), reverse=True))
    keep_track_ids = list(track_results.keys())[:num_max_people]
    track_results = {k: v for k, v in track_results.items() if k in keep_track_ids}
    return (track_results, masks)
