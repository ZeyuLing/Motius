"""InterMask joint-channel packing; keeps the released InterHuman ordering."""
import torch

def pack_motion(x, dataset_name, joints_num):
    if dataset_name == "interhuman":
        pos = x[..., :joints_num*3].reshape([x.shape[0], x.shape[1], joints_num, 3])
        vel = x[..., joints_num*3 : joints_num*3*2].reshape([x.shape[0], x.shape[1], joints_num, 3])

        rot = x[..., joints_num*3*2 : joints_num*3*2 + (joints_num-1)*6].reshape([x.shape[0], x.shape[1], joints_num-1, 6])
        rot = torch.cat([torch.zeros(rot.shape[0], rot.shape[1], 1, 6).to(x.device), rot], dim=2)

        joints = torch.cat([pos, vel, rot], dim=-1)
    else:
        joints = x
    joints = joints.permute(0, 3, 2, 1).float() # B, D=12, J=22, T

    return joints


def unpack_motion(x, dataset_name):
    x = x.permute(0, 3, 2, 1).float()

    if dataset_name == "interhuman":
        pos = x[:,:,:,:3].reshape([x.shape[0], x.shape[1], -1])
        vel = x[:,:,:,3:6].reshape([x.shape[0], x.shape[1], -1])
        rot = x[:,:,1:,6:6+6].reshape([x.shape[0], x.shape[1], -1])
        fc = torch.zeros((x.shape[0], x.shape[1], 4)).to(x.device)

        x = torch.cat([pos, vel, rot, fc], dim=-1)
    return x
