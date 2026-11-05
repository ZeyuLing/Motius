"""Index tracked people and prompts in a preprocessed PromptHMR video."""
import torch
import numpy as np
from PIL import Image,ImageOps
from torch.utils.data import Dataset
from torchvision.transforms import Normalize,ToTensor,Compose
from motius.motion.conditioning.prompthmr import pad_video_frame

class PromptHMRVideoDataset(Dataset):

    def __init__(self, images, tracks, cam_int):
        self.images = images
        self.tracks = tracks
        frames = set([x for t in tracks.values() for x in t['frames'].tolist()])
        self.frames = sorted(list(frames))
        self.cam_int = cam_int
        self.normalization = Compose([ToTensor(), Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        idx = self.frames[idx]
        image_cv = self.images[idx]
        boxes = []
        kpts = []
        masks = []
        track_ids = []
        for person_id, track in self.tracks.items():
            if idx in track['frames']:
                bbox = track['bboxes'][track['frames'] == idx]
                bbox = np.concatenate([bbox, np.ones_like(bbox)[..., :1]], axis=-1)
                kpt = track['keypoints_2d'][track['frames'] == idx][:, :25]
                obj_masks = track['masks'][track['frames'] == idx]
                mm = []
                for mask in obj_masks:
                    msk_size = int(896 / 14 * 4)
                    mask = Image.fromarray(mask)
                    mask = ImageOps.contain(mask, (msk_size, msk_size))
                    mask = ImageOps.pad(mask, size=(msk_size, msk_size))
                    mm.append(np.array(mask))
                mm = np.array(mm)
                mm = torch.from_numpy(mm).float()
                track_ids.append(person_id)
                boxes.append(bbox)
                kpts.append(kpt)
                masks.append(mm)
        boxes = torch.from_numpy(np.concatenate(boxes)).float()
        kpts = torch.from_numpy(np.concatenate(kpts)).float()
        masks = torch.from_numpy(np.concatenate(masks)).float()[:, None]
        cam_int_batch = self.cam_int.float()[None].repeat(len(boxes), 1, 1)
        item = {'boxes': boxes, 'cam_int': cam_int_batch, 'image_cv': image_cv, 'track_ids': track_ids, 'kpts': kpts, 'masks': masks}
        item = pad_video_frame(item, IMG_SIZE=896)
        item['image'] = self.normalization(item['image_cv'])
        item['image_cv'] = torch.tensor(item['image_cv'])
        return item

from torchvision import transforms

class CalibrationImageDataset(Dataset):

    def __init__(self, image_list, min_size=600, max_size=1000):
        self.images = image_list
        self.data_transform = transforms.Compose([transforms.Resize(min_size), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = {}
        if isinstance(self.images[index], str):
            imgname = self.images[index]
            img = Image.open(imgname).convert('RGB')
            orig_img_shape = np.array(img).shape[:2]
        elif isinstance(self.images, np.ndarray):
            img = self.images[index]
            orig_img_shape = img.shape[:2]
            img = Image.fromarray(img)
            imgname = index
        norm_img = self.data_transform(img)
        item['img'] = norm_img
        item['imgname'] = imgname
        item['orig_shape'] = orig_img_shape
        return item
