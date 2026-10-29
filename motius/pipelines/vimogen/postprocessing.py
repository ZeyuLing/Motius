import torch

def gaussian_kernel(kernel_size: int, sigma: float):
    x = torch.linspace(-(kernel_size - 1) / 2, (kernel_size - 1) / 2, kernel_size)
    gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return gauss / gauss.sum()

def smooth_motion_rep(motion_rep, kernel_size: int, sigma: float):
    """Temporal Gaussian smoothing for motion representation."""
    assert kernel_size % 2 == 1, 'kernel_size must be odd'
    data_dim = motion_rep.shape[-1]
    padding = (kernel_size - 1) // 2
    kernel = gaussian_kernel(kernel_size, sigma).to(motion_rep.device)[None, None, :].repeat(data_dim, 1, 1)
    motion_rep_smoothed = torch.nn.functional.conv1d(motion_rep.transpose(0, 1).unsqueeze(0), kernel, padding=padding, groups=data_dim)
    motion_rep_smoothed = motion_rep_smoothed.squeeze(0).transpose(0, 1)
    motion_rep_smoothed[:padding] = motion_rep[:padding]
    motion_rep_smoothed[-padding:] = motion_rep[-padding:]
    return motion_rep_smoothed
