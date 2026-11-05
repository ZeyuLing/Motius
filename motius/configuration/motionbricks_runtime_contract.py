"""Pose layouts and timing for native MotionBricks runtime services."""

from enum import Enum
from types import SimpleNamespace

MODEL_SAMPLE_RATE = 30.0
MODEL_POSES_PER_TOKEN = 4
MODEL_CONTEXT_AND_TARGET_POSES = 8
MODEL_MIN_TOKEN_COUNT = 6
MODEL_MAX_TOKEN_COUNT = 16
MODEL_BOUNDARY_FRAMES = 3
MODEL_MAX_VALID_FRAMES = 58


class ModelRig(str, Enum):
    SOMA = "soma"
    UNITREE_G1 = "unitree_g1"

    def __str__(self):
        return self.value


class TensorContractError(ValueError):
    pass


def valid_frame_count(token_count):
    if not 6 <= token_count <= 16:
        raise ValueError("Token count must be between 6 and 16")
    return 4 * token_count - 6


SOMA_N1_HEADING_JOINTS = ("RightLeg", "LeftLeg")
SOMA_N1_FOOT_CONTACT_JOINTS = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")
UNITREE_G1_N1_FOOT_CONTACT_BODIES = ("left_ankle_roll_link",) * 2 + (
    "right_ankle_roll_link",
) * 2
UNITREE_G1_N1_FOOT_CONTACT_OFFSETS = ((0.0, 0.0, 0.0), (0.14, 0.0, -0.035)) * 2
SOMA_N1_CONTRACT = SimpleNamespace(
    joint_names=(
        "Hips",
        "Spine1",
        "Spine2",
        "Chest",
        "Neck1",
        "Neck2",
        "Head",
        "Jaw",
        "LeftEye",
        "RightEye",
        "LeftShoulder",
        "LeftArm",
        "LeftForeArm",
        "LeftHand",
        "LeftHandThumbEnd",
        "LeftHandMiddleEnd",
        "RightShoulder",
        "RightArm",
        "RightForeArm",
        "RightHand",
        "RightHandThumbEnd",
        "RightHandMiddleEnd",
        "LeftLeg",
        "LeftShin",
        "LeftFoot",
        "LeftToeBase",
        "RightLeg",
        "RightShin",
        "RightFoot",
        "RightToeBase",
    )
)
UNITREE_G1_N1_CONTRACT = SimpleNamespace(
    joint_names=(
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    )
)
