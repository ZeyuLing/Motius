"""Validate OpenCV camera-from-world transforms before conditioning FlowHMR."""
import torch


def validate_extrinsics(camera_rt, frames):
    rt = torch.as_tensor(camera_rt, dtype=torch.float32).detach().cpu()
    if rt.shape == (4, 4):
        rt = rt[None].expand(frames, -1, -1).clone()
    if rt.shape != (frames, 4, 4) or frames < 1 or not torch.isfinite(rt).all():
        raise ValueError('camera_RT must have exactly one finite world-to-camera transform per feature frame')
    if not torch.allclose(rt[:, 3], torch.tensor([0., 0., 0., 1.]).expand(frames, -1), atol=1e-5):
        raise ValueError('camera_RT has invalid homogeneous rows')
    r = rt[:, :3, :3]
    if not torch.allclose(r @ r.transpose(-1, -2), torch.eye(3).expand(frames, -1, -1), atol=2e-4) or not torch.allclose(torch.linalg.det(r), torch.ones(frames), atol=2e-4):
        raise ValueError('camera_RT rotations must be proper orthonormal rotations, not reflections')
    return rt
