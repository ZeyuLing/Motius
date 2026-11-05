"""Selected SONIC G1 order components; see licenses/sonic."""

import torch

G1_ISAACLab_ORDER = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]
G1_HAND_JOINTS = [
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
]
_body_joint_indices_cache = {}
_hand_joint_indices_cache = {}


def _get_joint_indices_by_names(asset, joint_names: list, cache: dict) -> torch.Tensor:
    """Get indices of specified joints in the robot's joint list."""
    cache_key = (id(asset), tuple(joint_names))
    if cache_key in cache:
        return cache[cache_key]
    robot_joint_names = asset.joint_names
    indices = [
        robot_joint_names.index(n) for n in joint_names if n in robot_joint_names
    ]
    indices_tensor = torch.tensor(indices, dtype=torch.long, device=asset.device)
    cache[cache_key] = indices_tensor
    return indices_tensor


def get_body_joint_indices(asset) -> torch.Tensor:
    """Get indices of body joints (29 DOF) using G1_ISAACLab_ORDER."""
    return _get_joint_indices_by_names(
        asset, G1_ISAACLab_ORDER, _body_joint_indices_cache
    )


def get_hand_joint_indices(asset) -> torch.Tensor:
    """Get indices of hand joints (14 DOF) using G1_HAND_JOINTS."""
    return _get_joint_indices_by_names(asset, G1_HAND_JOINTS, _hand_joint_indices_cache)


G1_ISAACLAB_JOINTS = [
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
]
G1_ISAACLAB_TO_MUJOCO_DOF = [
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
]
G1_MUJOCO_TO_ISAACLAB_DOF = [
    0,
    6,
    12,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    22,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
]
G1_ISAACLAB_TO_MUJOCO_BODY = [
    0,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    3,
    6,
    9,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
    13,
    17,
    21,
    23,
    25,
    27,
    29,
]
G1_MUJOCO_TO_ISAACLAB_BODY = [
    0,
    1,
    7,
    13,
    2,
    8,
    14,
    3,
    9,
    15,
    4,
    10,
    16,
    23,
    5,
    11,
    17,
    24,
    6,
    12,
    18,
    25,
    19,
    26,
    20,
    27,
    21,
    28,
    22,
    29,
]
G1_ISAACLAB_TO_MUJOCO_MAPPING = {
    "isaaclab_joints": G1_ISAACLAB_JOINTS,
    "isaaclab_to_mujoco_dof": G1_ISAACLAB_TO_MUJOCO_DOF,
    "mujoco_to_isaaclab_dof": G1_MUJOCO_TO_ISAACLAB_DOF,
    "isaaclab_to_mujoco_body": G1_ISAACLAB_TO_MUJOCO_BODY,
    "mujoco_to_isaaclab_body": G1_MUJOCO_TO_ISAACLAB_BODY,
}
from abc import ABC
import torch


class IsaacLabMuJoCoConverter(ABC):
    """Abstract base class for DOF/body order conversion between IsaacLab and MuJoCo.

    Subclasses must define DOF_MAPPINGS for their specific robot.
    """

    ROOT_QPOS_OFFSET = 7
    DOF_MAPPINGS: dict = {}
    VALID_DOF_ORDERS = ("mujoco", "isaaclab")

    def convert(
        self, data: torch.Tensor, from_order: str, to_order: str
    ) -> torch.Tensor:
        """Convert DOF order between conventions.

        Auto-detects input format:
            - qpos:           [..., 7 + num_dof]           -> reorder DOF portion, keep root
            - dof angles:     [..., num_dof]                -> reorder directly
            - body transforms [..., num_dof + 1, D]         -> keep root body, reorder DOF bodies
            - body transforms [..., num_dof, D]             -> reorder directly
            - rotation mats   [..., num_dof (+ 1), 3, 3]   -> same as above on dim -3

        Note: Body reordering for per-body transforms uses DOF_MAPPINGS (not
        BODY_MAPPINGS) because each DOF link has exactly one parent body in G1,
        so DOF order and DOF-body order are consistent by construction.  The
        separate BODY_MAPPINGS include the root (pelvis) at index 0 and are used
        only when the full 30-body ordering is needed.
        """
        if from_order == to_order:
            return data
        mapping = self.DOF_MAPPINGS[from_order, to_order]
        last_dim = data.shape[-1]
        if last_dim == self.ROOT_QPOS_OFFSET + self.num_dof:
            return torch.cat(
                [
                    data[..., : self.ROOT_QPOS_OFFSET],
                    data[..., self.ROOT_QPOS_OFFSET :][..., mapping],
                ],
                dim=-1,
            )
        elif last_dim == self.num_dof:
            return data[..., mapping]
        else:
            body_dim = (
                -3
                if data.ndim >= 3 and data.shape[-1] == 3 and (data.shape[-2] == 3)
                else -2
            )
            num_bodies = data.shape[body_dim]
            if num_bodies == self.num_dof + 1:
                body_order = [0] + [m + 1 for m in mapping]
            elif num_bodies == self.num_dof:
                body_order = mapping
            else:
                raise ValueError(
                    f"Cannot detect format: last_dim={last_dim}, body_dim size={num_bodies}, expected num_dof+1={self.num_dof + 1} or num_dof={self.num_dof}"
                )
            if body_dim == -2:
                return data[..., body_order, :]
            else:
                return data[..., body_order, :, :]

    def to_mujoco(self, data: torch.Tensor) -> torch.Tensor:
        """Convert from IsaacLab to MuJoCo DOF order."""
        return self.convert(data, from_order="isaaclab", to_order="mujoco")

    def to_isaaclab(self, data: torch.Tensor) -> torch.Tensor:
        """Convert from MuJoCo to IsaacLab DOF order."""
        return self.convert(data, from_order="mujoco", to_order="isaaclab")

    @property
    def num_dof(self) -> int:
        """Number of actuated DOFs (excluding root)."""
        return len(
            self.DOF_MAPPINGS[self.VALID_DOF_ORDERS[0], self.VALID_DOF_ORDERS[1]]
        )


class G1Converter(IsaacLabMuJoCoConverter):
    """G1 ordering converter.

    Uses the selected G1 body/DOF/joint metadata in this module.
    """

    def __init__(self):
        self.JOINT_NAMES = G1_ISAACLAB_JOINTS
        self.DOF_MAPPINGS = {
            ("isaaclab", "mujoco"): G1_ISAACLAB_TO_MUJOCO_DOF,
            ("mujoco", "isaaclab"): G1_MUJOCO_TO_ISAACLAB_DOF,
        }
        self.BODY_MAPPINGS = {
            ("isaaclab", "mujoco"): G1_ISAACLAB_TO_MUJOCO_BODY,
            ("mujoco", "isaaclab"): G1_MUJOCO_TO_ISAACLAB_BODY,
        }

    VR_3POINTS_BODY_NAMES = [
        "torso_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
    ]
    FOOT_BODY_NAMES = ["left_ankle_roll_link", "right_ankle_roll_link"]

    @property
    def vr_3points_mujoco_indices(self):
        """VR 3-point body indices in full (30-body) MuJoCo body order.

        These index into the full body array after isaaclab_to_mujoco_body
        reordering, NOT the 14-body motion.yaml body_names subset.
        """
        mj_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mj_names.index(n) for n in self.VR_3POINTS_BODY_NAMES]

    @property
    def foot_mujoco_indices(self):
        """Foot body indices in full (30-body) MuJoCo body order.

        These index into the full body array after isaaclab_to_mujoco_body
        reordering, NOT the 14-body motion.yaml body_names subset.
        """
        mj_names = [self.JOINT_NAMES[i] for i in self.isaaclab_to_mujoco_body]
        return [mj_names.index(n) for n in self.FOOT_BODY_NAMES]

    @property
    def isaaclab_to_mujoco_dof(self):
        """DOF reorder indices: IsaacLab -> MuJoCo."""
        return self.DOF_MAPPINGS["isaaclab", "mujoco"]

    @property
    def mujoco_to_isaaclab_dof(self):
        """DOF reorder indices: MuJoCo -> IsaacLab."""
        return self.DOF_MAPPINGS["mujoco", "isaaclab"]

    @property
    def isaaclab_to_mujoco_body(self):
        """Body reorder indices: IsaacLab -> MuJoCo."""
        return self.BODY_MAPPINGS["isaaclab", "mujoco"]

    @property
    def mujoco_to_isaaclab_body(self):
        """Body reorder indices: MuJoCo -> IsaacLab."""
        return self.BODY_MAPPINGS["mujoco", "isaaclab"]

    def get_isaaclab_to_mujoco_mapping(self):
        """Return the full mapping dict for body/DOF reordering."""
        return {
            "isaaclab_joints": self.JOINT_NAMES,
            "isaaclab_to_mujoco_dof": self.isaaclab_to_mujoco_dof,
            "mujoco_to_isaaclab_dof": self.mujoco_to_isaaclab_dof,
            "isaaclab_to_mujoco_body": self.isaaclab_to_mujoco_body,
            "mujoco_to_isaaclab_body": self.mujoco_to_isaaclab_body,
        }
