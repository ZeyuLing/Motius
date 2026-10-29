"""Published person segmentation preprocessing and pixel-label conversion."""
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import Compose,ToTensor,Normalize


@torch.no_grad()
def segment_people(bundle,images,batch_size=8):
    transform=Compose([ToTensor(),Normalize(mean=[.485,.456,.406],std=[.229,.224,.225])])
    device=next(bundle.parameters()).device
    masks=[]
    for start in range(0,len(images),batch_size):
        batch=torch.stack([transform(Image.fromarray(image)) for image in images[start:start+batch_size]]).to(device)
        output=bundle(batch)['out']
        masks.append(((output.argmax(1)==15).to(torch.float).cpu()>.5))
    return torch.cat(masks,dim=0).numpy()
