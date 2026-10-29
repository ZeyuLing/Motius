"""Published metric-depth processing and scale fitting; no neural modules."""
import cv2
import torch
import numpy as np

def prep_metric3d(rgb_origin, intrinsic, model_version):
    # rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]

    #### ajust input size to fit pretrained model
    # keep ratio resize
    if 'convnext' in model_version:
        input_size = (544, 1216)
    else:
        input_size = (616, 1064) # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic
    intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                             cv2.BORDER_CONSTANT, value=padding)
    pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    rgb = rgb[None, :, :, :].cuda()
    return rgb, intrinsic, pad_info, rgb_origin


def prep_metric3d_img(rgb_file):
    if isinstance(rgb_file, str):
        rgb_origin = cv2.imread(rgb_file)[:, :, ::-1]
    elif isinstance(rgb_file, np.ndarray):
        rgb_origin = rgb_file

    #### ajust input size to fit pretrained model
    # keep ratio resize
    # if 'convnext' in model_version:
    #     input_size = (544, 1216)
    # else:
    input_size = (616, 1064) # for vit model
    # input_size = (544, 1216) # for convnext model
    h, w = rgb_origin.shape[:2]
    scale = min(input_size[0] / h, input_size[1] / w)
    rgb = cv2.resize(rgb_origin, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
    # remember to scale intrinsic
    # intrinsic = [intrinsic[0] * scale, intrinsic[1] * scale, intrinsic[2] * scale, intrinsic[3] * scale]
    # padding to input_size
    padding = [123.675, 116.28, 103.53]
    h, w = rgb.shape[:2]
    pad_h = input_size[0] - h
    pad_w = input_size[1] - w
    pad_h_half = pad_h // 2
    pad_w_half = pad_w // 2
    rgb = cv2.copyMakeBorder(rgb, pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half,
                             cv2.BORDER_CONSTANT, value=padding)
    # pad_info = [pad_h_half, pad_h - pad_h_half, pad_w_half, pad_w - pad_w_half]

    #### normalize
    mean = torch.tensor([123.675, 116.28, 103.53]).float()[:, None, None]
    std = torch.tensor([58.395, 57.12, 57.375]).float()[:, None, None]
    rgb = torch.from_numpy(rgb.transpose((2, 0, 1))).float()
    rgb = torch.div((rgb - mean), std)
    # rgb = rgb[None, :, :, :]#.cuda()
    return rgb


def post_metric3d_batch(pred_depth, confidence, pad_info, rgb_origin, intrinsic, resize_shape=None):
    # un pad
    pred_depth = pred_depth[:, 0]
    pred_depth = pred_depth[:, pad_info[0]:pred_depth.shape[-2]-pad_info[1], pad_info[2]:pred_depth.shape[-1]-pad_info[3]]

    # upsample to original size
    if not(resize_shape is None):
        pred_depth = torch.nn.functional.interpolate(pred_depth[:, None, :, :], resize_shape, mode='bilinear').squeeze()
    else:
        pred_depth = torch.nn.functional.interpolate(pred_depth[:, None, :, :], rgb_origin.shape[:2], mode='bilinear').squeeze()

    #### de-canonical transform
    canonical_to_real_scale = intrinsic[0] / 1000.0 # 1000.0 is the focal length of canonical camera
    pred_depth = pred_depth * canonical_to_real_scale # now the depth is metric
    pred_depth = torch.clamp(pred_depth, 0, 300)

    return pred_depth
