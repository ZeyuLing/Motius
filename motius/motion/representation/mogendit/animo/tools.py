import enum
import torch
import numpy as np
from motius.motion.representation.mogendit.animo.utils.angular import rotation_matrix_to_euler_angle_np
from motius.motion.representation.mogendit.animo.utils.angular import euler_angle_to_rotation_matrix_np
from motius.motion.representation.mogendit.animo.utils.angular import rotation_matrix_to_r6d
from motius.motion.representation.mogendit.animo.utils.angular import normalize_angle
from motius.motion.representation.mogendit.animo.utils.angular import r6d_to_rotation_matrix
from motius.motion.representation.mogendit.animo.utils.angular import rotation_matrix_to_axis_angle
from motius.motion.representation.mogendit.animo.utils.angular import axis_angle_to_rotation_matrix

# 欧拉角相关计算从rotation_conversions中导入
from motius.motion.representation.mogendit.animo.utils.rotation_conversions import euler_angles_to_matrix
from motius.motion.representation.mogendit.animo.utils.rotation_conversions import matrix_to_euler_angles

# 在Jacobian中屏蔽无生理自由度的旋转
# 注意: 如果没对姿态使用constrain_joint_dot操作, 关节18, 19的数据要删除
constrain_joints = np.array([18, 19, 4, 5], dtype=np.int_)
constrain_axis = np.array([2, 2, 2, 2], dtype=np.int_)
# constrain_joints = np.array([18, 18, 18, 19, 19, 19], dtype=np.int_)
# constrain_axis = np.array([0, 1, 2, 0, 1, 2], dtype=np.int_)
constrain_channels_euler = (constrain_joints * 3 + constrain_axis).tolist()
constrain_channels_q = (3 + constrain_joints * 3 + constrain_axis).tolist()


def constrain_joint_dofs(poses):
    # 肘关节与膝关节
    joints_origin = poses[:, [4, 5, 18, 19]]
    joints = rotation_matrix_to_r6d(joints_origin).reshape(-1, 4, 6)
    # z轴不旋转 -> x轴方向没有y分量
    joints[:, :, :3] *= torch.FloatTensor([[[1, 0, 1]]])
    joints = r6d_to_rotation_matrix(joints).reshape(-1, 4, 3, 3)
    poses[:, [4, 5, 18, 19]] = joints
    # 减少的旋转分量转移到髋/肩关节
    # delta_R = joints_origin.matmul(joints.transpose(-2, -1))
    delta_R = joints_origin.matmul(joints.transpose(-2, -1))
    poses[:, [1, 2, 16, 17]] = poses[:, [1, 2, 16, 17]].matmul(delta_R)

    return poses


def mask_jacobian(J):
    # J: 72 x 75
    J[:, constrain_channels_q] *= 0
    return J
