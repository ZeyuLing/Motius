# Copyright (c) Meta Platforms, Inc. and affiliates.
# Fast-SAM-3D-Body GPU crop/merged-hand design: yangtiming/Fast-SAM-3D-Body.
# See motius/licenses/flowhmr/FAST_SAM_LICENSE and ATTRIBUTIONS.md.
"""GPU image preprocessing with the released OpenCV crop's integer arithmetic.

Only small affine metadata is constructed on CPU using the reference transforms.
RGB pixels are uploaded once; warp, zero padding, uint8 rounding, scaling and
normalization stay on the selected device. No learned model is defined here.
"""
import cv2
import numpy as np
import torch


@torch.no_grad()
def warp_rgb(image, matrices, output_size):
    """Match cv2.warpAffine(INTER_LINEAR, BORDER_CONSTANT) on uint8 RGB.

    OpenCV affine warping rounds inverse coordinates at 10 fractional bits,
    then indexes a 5-bit bilinear table. Smooth grid_sample is not equivalent.
    """
    if image.dtype != torch.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError('Expected HWC uint8 RGB image')
    width, height = map(int, output_size)
    inverse = np.stack([cv2.invertAffineTransform(m) for m in matrices])
    m = torch.as_tensor(inverse, dtype=torch.float64, device=image.device)
    x = torch.arange(width, device=image.device, dtype=torch.float64)[None, None, :]
    y = torch.arange(height, device=image.device, dtype=torch.float64)[None, :, None]
    # Separate rounding mirrors OpenCV's affine blockline implementation.
    sx = (torch.round(m[:, 0, 0, None, None]*x*1024).long()
          + torch.round((m[:, 0, 1, None, None]*y+m[:, 0, 2, None, None])*1024).long()+16) >> 5
    sy = (torch.round(m[:, 1, 0, None, None]*x*1024).long()
          + torch.round((m[:, 1, 1, None, None]*y+m[:, 1, 2, None, None])*1024).long()+16) >> 5
    ix, iy = sx >> 5, sy >> 5
    fx, fy = sx & 31, sy & 31
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3)
    result = torch.zeros((*sx.shape, 3), device=image.device, dtype=torch.int32)
    for dx, dy, weight in ((0, 0, (32-fx)*(32-fy)), (1, 0, fx*(32-fy)),
                           (0, 1, (32-fx)*fy), (1, 1, fx*fy)):
        px, py = ix+dx, iy+dy
        valid = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        index = py.clamp(0, h-1)*w+px.clamp(0, w-1)
        result += pixels[index].int()*weight[..., None].int()*valid[..., None]
    result = ((result+512) >> 10).to(torch.uint8)
    return result.permute(0, 3, 1, 2).contiguous().float()/255.0


def prepare_gpu_batch(image, boxes, camera, transform):
    """Original crop geometry, GPU RGB warp and ToTensor scaling; (1,N,...) batch."""
    if len(transform.transforms) != 3:
        raise ValueError('Fast SAM requires the released three-stage crop transform')
    geometry = []
    for box in np.asarray(boxes, dtype=np.float32).reshape(-1, 4):
        item = dict(bbox=box.copy(), bbox_format='xyxy')
        for operation in transform.transforms[:2]:
            item = operation(item)
        geometry.append(item)
    if not geometry:
        raise ValueError('At least one selected person is required')
    width, height = map(int, geometry[0]['input_size'])
    batch = {}
    for key in ('bbox', 'bbox_center', 'bbox_scale', 'affine_trans'):
        batch[key] = torch.as_tensor(np.stack([g[key] for g in geometry]),
                                    dtype=torch.float32, device=image.device)[None]
    batch['img'] = warp_rgb(image, [g['affine_trans'] for g in geometry], (width, height))[None]
    count = len(geometry)
    for key, value in (('img_size', (width, height)), ('ori_img_size', (image.shape[1], image.shape[0]))):
        batch[key] = torch.tensor(value, dtype=torch.float32, device=image.device).expand(1, count, 2)
    batch['mask'] = torch.zeros(1, count, 1, height, width, device=image.device)
    batch['mask_score'] = torch.zeros(1, count, device=image.device)
    batch['person_valid'] = torch.ones(1, count, device=image.device)
    batch['_unit_rgb'] = True
    batch['cam_int'] = torch.as_tensor(camera, device=image.device, dtype=torch.float32).reshape(1, 3, 3)
    return batch


def merge_hands(left, right):
    """Concatenate person axis, in [all left, all right] order; intrinsics are shared."""
    keys = ('img', 'img_size', 'ori_img_size', 'bbox_center', 'bbox_scale',
            'bbox', 'affine_trans', 'mask', 'mask_score', 'person_valid')
    return {**{key: torch.cat((left[key], right[key]), dim=1) for key in keys},
            'cam_int': left['cam_int'], '_unit_rgb': True}
