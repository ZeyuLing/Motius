import torch
from torch.nn import functional as F
from motius.models.hymotion_t2m.network.geometry import (
    axis_angle_to_matrix as _axis_angle_to_matrix,
    axis_angle_to_quaternion as _axis_angle_to_quaternion,
    matrix_to_axis_angle as _matrix_to_axis_angle,
    quaternion_to_axis_angle as _quaternion_to_axis_angle,
)

def mat3x3_to_rot6d(R):
    # rot6d takes the first two columns of R
    return R[..., :, :2].reshape(R.shape[0], 6)

def rot6d_to_mat3x3(rot6d):
    """
    Convert 6d rotation representation to 3x3 rotation matrix.
    Shape:
        - Input: :Torch:`(N, 6)`
        - Output: :Torch:`(N, 3, 3)`
    """
    rot6d = rot6d.view(-1, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix
    return rot_mat
    
def axis_angle_to_rot6d(angle_axis):
    """Convert 3d vector of axis-angle rotation to 6d rotation representation.
    Shape:
        - Input: :Torch:`(N, 3)`
        - Output: :Torch:`(N, 6)`
    """
    rot_mat = axis_angle_to_mat3x3(angle_axis)
    rot6d = rot_mat[:, :3, :2]
    rot6d = rot6d.reshape(-1, 6)

    return rot6d

def rot6d_to_axis_angle(rot6d):
    """Convert 6d rotation representation to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(N, 6)`
        - Output: :Torch:`(N, 3)`
    """
    batch_size = rot6d.shape[0]

    rot6d = rot6d.view(batch_size, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=-1)
    rot_mat = torch.stack((b1, b2, b3), dim=-1)  # 3x3 rotation matrix

    axis_angle = _matrix_to_axis_angle(rot_mat).reshape(-1, 3)
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def axis_angle_to_mat3x3(angle_axis):
    """
    Convert 3d vector of axis-angle rotation to 3x3 rotation matrix.
    Shape:
        - Input: :Torch:`(N, 3)`
        - Output: :Torch:`(N, 3, 3)`
    """
    return _axis_angle_to_matrix(angle_axis)

def mat3x3_to_axis_angle(rot_mat):
    """
    Convert 3x3 rotation matrix to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(N, 3, 3)`
        - Output: :Torch:`(N, 3)`
    """
    axis_angle = _matrix_to_axis_angle(rot_mat).reshape(-1, 3)
    axis_angle[torch.isnan(axis_angle)] = 0.0
    return axis_angle

def quaternion_to_axis_angle(quaternion):
    """
    Convert 4d quaternion to 3d vector of axis-angle rotation.
    Shape:
        - Input: :Torch:`(..., 4)`
        - Output: :Torch:`(..., 3)`
    """
    return _quaternion_to_axis_angle(quaternion)

def axis_angle_to_quaternion(angle_axis):
    """
    Convert 3d vector of axis-angle rotation to 4d quaternion.
    Shape:
        - Input: :Torch:`(..., 3)`
        - Output: :Torch:`(..., 4)`
    """
    return _axis_angle_to_quaternion(angle_axis)

def quaternion_to_rot6d(quaternion):
    """
    Convert 4d quaternion to 6d rotation representation.
    Shape:
        - Input: :Torch:`(N, 4)`
        - Output: :Torch:`(N, 6)`
    """

    return axis_angle_to_rot6d(quaternion_to_axis_angle(quaternion))

def rot6d_to_quaternion(rot6d):
    """
    Convert 6d rotation representation to 4d quaternion.
    Shape:
        - Input: :Torch:`(N, 4)`
        - Output: :Torch:`(N, 6)`
    """

    return axis_angle_to_quaternion(rot6d_to_axis_angle(rot6d))
