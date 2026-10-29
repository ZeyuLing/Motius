"""InterHuman-262 artifact normalization used by InterGen."""
import numpy as np
import torch

face_joint_indx = [2, 1, 17, 16]

class MotionNormalizerTorch():
    def __init__(self, mean_path, std_path):
        mean = np.load(mean_path)
        std = np.load(std_path)

        self.motion_mean = torch.from_numpy(mean).float()
        self.motion_std = torch.from_numpy(std).float()


    def forward(self, x):
        device = x.device
        x = x.clone()
        x = (x - self.motion_mean.to(device)) / self.motion_std.to(device)
        return x

    def backward(self, x, global_rt=False):
        device = x.device
        x = x.clone()
        x = x * self.motion_std.to(device) + self.motion_mean.to(device)
        return x
