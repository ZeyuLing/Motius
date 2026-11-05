import torch
import torch.nn.functional as F

def cvx_upsample(data, mask):
    """ upsample pixel-wise transformation field """
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)
    up_data = F.unfold(data, [3, 3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)
    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8 * ht, 8 * wd, dim)
    return up_data

def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch * num, ht, wd, 1)
    mask = mask.view(batch * num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8 * ht, 8 * wd)
