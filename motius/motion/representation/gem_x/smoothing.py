import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from scipy.ndimage._filters import _gaussian_kernel1d

def gaussian_smooth(x, sigma=3, dim=-1):
    kernel_smooth = _gaussian_kernel1d(sigma=sigma, order=0, radius=int(4 * sigma + 0.5))
    kernel_smooth = torch.from_numpy(kernel_smooth).float()[None, None].to(x)
    rad = kernel_smooth.size(-1) // 2
    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, '... f -> (...) 1 f')
    x = F.pad(x[None], (rad, rad, 0, 0), mode='replicate')[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)
    x = x.transpose(-1, dim)
    return x

def moving_average_smooth(x, window_size=5, dim=-1):
    kernel_smooth = torch.ones(window_size).float() / window_size
    kernel_smooth = kernel_smooth[None, None].to(x)
    rad = kernel_smooth.size(-1) // 2
    x = x.transpose(dim, -1)
    x_shape = x.shape[:-1]
    x = rearrange(x, '... f -> (...) 1 f')
    x = F.pad(x[None], (rad, rad, 0, 0), mode='replicate')[0]
    x = F.conv1d(x, kernel_smooth)
    x = x.squeeze(1).reshape(*x_shape, -1)
    x = x.transpose(-1, dim)
    return x
