"""COCO17 heatmap test-time flip."""
import torch


def flip_heatmap_coco17(output_flipped):
    assert output_flipped.ndim == 4, 'output_flipped should be [B, J, H, W]'
    shape_ori = output_flipped.shape
    channels = 1
    output_flipped = output_flipped.reshape(shape_ori[0], -1, channels, shape_ori[2], shape_ori[3])
    output_flipped_back = output_flipped.clone()
    for left, right in [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]:
        output_flipped_back[:, left, ...] = output_flipped[:, right, ...]
        output_flipped_back[:, right, ...] = output_flipped[:, left, ...]
    output_flipped_back = output_flipped_back.reshape(shape_ori)
    output_flipped_back = output_flipped_back.flip(3)
    return output_flipped_back
