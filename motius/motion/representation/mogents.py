"""MoGenTS HumanML/KIT joint-field packing and released decoder layout."""
import torch
import torch.nn.functional as F


def pack_motion(motion,joints=22):
    batch,frames,_=motion.shape
    fields=torch.zeros((batch,frames,joints,12),device=motion.device,dtype=torch.float32)
    fields[:,:,1:,:3]=motion[:,:,4:4+(joints-1)*3].reshape(batch,frames,joints-1,3)
    fields[:,:,1:,3:9]=motion[:,:,4+(joints-1)*3:4+(joints-1)*9].reshape(batch,frames,joints-1,6)
    fields[:,:,:,9:12]=motion[:,:,4+(joints-1)*9:4+(joints-1)*9+joints*3].reshape(batch,frames,joints,3)
    return F.pad(fields,(0,0,1,1 if joints==22 else 2))


def unpack_motion(decoded,joints=22):
    batch,frames=decoded.shape[:2]
    extra=2 if joints==22 else 3
    fields=decoded.reshape(batch,frames,joints+extra,12)[:,:,1:-1 if joints==22 else -2,:].reshape(batch*frames,joints,12)
    motion=torch.zeros((batch,frames,263 if joints==22 else 251),device=decoded.device)
    motion[:,:,:4]=fields[:,0,:4].reshape(batch,frames,-1)
    motion[:,:,-4:]=fields[:,0,4:8].reshape(batch,frames,-1)
    motion[:,:,4:4+(joints-1)*3]=fields[:,1:,:3].reshape(batch,frames,-1)
    motion[:,:,4+(joints-1)*3:4+(joints-1)*9]=fields[:,1:,3:9].reshape(batch,frames,-1)
    motion[:,:,4+(joints-1)*9:4+(joints-1)*9+joints*3]=fields[:,:,9:12].reshape(batch,frames,-1)
    return motion
