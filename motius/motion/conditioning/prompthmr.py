"""Published PromptHMR image padding, prompt conversion and normalization."""
import torch
import numpy as np
from PIL import Image,ImageOps
from torchvision.transforms import Normalize,ToTensor,Compose,CenterCrop

def prepare_batch(inputs, kpt_conf=0.3, img_size=896, zoom=None, interaction=None):
    normalization = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    batch = []
    for inp in inputs:
        img = inp.get('image_cv')
        boxes = inp.get('boxes', None)
        keypoints = inp.get('kpts', None)
        masks = inp.get('masks', None)
        text = inp.get('text', None)
        cam_int = inp.get('cam_int', None)
        focal_scale = inp.get('focal_scale', None)
        item = {}
        item['image_cv'] = img
        item['cam_int'] = cam_int
        if boxes is not None:
            item['boxes'] = boxes[:, :4].clone()
            item['boxes'] = torch.cat([item['boxes'], torch.ones_like(item['boxes'])[..., :1]], dim=-1)
        else:
            item['boxes'] = None
        if keypoints is not None:
            num_k = keypoints.shape[1]
            kpts = keypoints.reshape(-1, 3).clone()
            kpts[kpts[:, -1] < kpt_conf] = 0
            kpts[kpts[:, -1] > kpt_conf, -1] = 1
            kpts = kpts.reshape(-1, num_k, 3)
            item['kpts'] = kpts
        else:
            item['kpts'] = None
        if masks is not None:
            if isinstance(masks, torch.Tensor):
                masks = masks.numpy()
            mm = []
            for mask in masks:
                msk_size = int(img_size / 14 * 4)
                mask = Image.fromarray(mask)
                mask = ImageOps.contain(mask, (msk_size, msk_size))
                mask = ImageOps.pad(mask, size=(msk_size, msk_size))
                mm.append(np.array(mask))
            masks = np.array(mm)
            masks = torch.tensor(masks).unsqueeze(1)
            item['masks'] = masks.float()
        else:
            item['masks'] = None
        if text is not None:
            item['text'] = text
        else:
            item['text'] = None
        if interaction is not None:
            item['interaction'] = interaction
        if cam_int is None:
            h, w = img.shape[:2]
            cam_int = torch.tensor([[[(h ** 2 + w ** 2) ** 0.5, 0, w / 2.0], [0, (h ** 2 + w ** 2) ** 0.5, h / 2.0], [0, 0, 1]]])
            item['cam_int'] = cam_int
            if focal_scale is not None:
                print('use focal scale ..')
                cam_int[:, 0, 0] *= focal_scale
                cam_int[:, 1, 1] *= focal_scale
        item = pad_image(item, img_size, zoom)
        item['image'] = normalization(item['image_cv'])
        item['image_cv'] = torch.tensor(item['image_cv'])
        batch.append(item)
    return batch

def pad_image(item, IMG_SIZE=896, zoom=None):
    img = item['image_cv']
    size = np.array([img.shape[1], img.shape[0]])
    scale = IMG_SIZE / max(size)
    offset = (IMG_SIZE - scale * size) / 2
    img_pil = Image.fromarray(img)
    img_pil = ImageOps.contain(img_pil, (IMG_SIZE, IMG_SIZE))
    img_pil = ImageOps.pad(img_pil, size=(IMG_SIZE, IMG_SIZE))
    if zoom is not None:
        assert type(zoom) == float
        zooming = int(zoom * IMG_SIZE)
        zooming = CenterCrop(zooming)
        img_pil = ImageOps.contain(zooming(img_pil), (IMG_SIZE, IMG_SIZE))
    else:
        zoom = 1.0
    img = np.array(img_pil)
    item['image_cv'] = img
    item['cam_int_original'] = item['cam_int'].clone()
    item['cam_int'] = item['cam_int'].mean(dim=0, keepdim=True)
    item['cam_int'][:, :2] *= scale
    item['cam_int'][:, :2, -1] += offset
    item['cam_int'][:, :2] /= zoom
    if item['boxes'] is not None:
        item['boxes'][:, :4] *= scale
        item['boxes'][:, :2] += offset
        item['boxes'][:, 2:4] += offset
        zoom_offset = (IMG_SIZE - int(zoom * IMG_SIZE)) / 2
        item['boxes'][:, :4] -= zoom_offset
        item['boxes'][:, :4] /= zoom
        item['boxes'][:, :4] = item['boxes'][:, :4].clip(0, IMG_SIZE)
    if item['kpts'] is not None:
        kpts = item['kpts'].clone()
        kpts[:, :, :2] *= scale
        kpts[:, :, :2] += offset
        item['kpts'] = kpts
    item['image_scale'] = scale
    item['image_offset'] = offset
    return item
def pad_video_frame(item, IMG_SIZE=896):
    img = item['image_cv']
    size = np.array([img.shape[1], img.shape[0]])
    scale = IMG_SIZE / max(size)
    offset = (IMG_SIZE - scale * size) / 2
    img_pil = Image.fromarray(img)
    img_pil = ImageOps.contain(img_pil, (IMG_SIZE, IMG_SIZE))
    img_pil = ImageOps.pad(img_pil, size=(IMG_SIZE, IMG_SIZE))
    img = np.array(img_pil)
    item['image_cv'] = img
    item['cam_int'] = item['cam_int'].mean(dim=0, keepdim=True)
    item['cam_int'][:, :2] *= scale
    item['cam_int'][:, :2, -1] += offset
    item['boxes'] *= scale
    item['boxes'][:, :2] += offset
    item['boxes'][:, 2:4] += offset
    kpts = item.get('kpts', None)
    if kpts is not None:
        kpts[:, :, :2] *= scale
        kpts[:, :, :2] += offset
        item['kpts'] = kpts
    return item
