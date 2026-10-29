"""MaskControl joint anchors and differentiable control coordinates."""
import torch

CONTROL_JOINT_IDS = (0, 10, 11, 15, 20, 21)


CONTROL_JOINT_NAMES = (
    "pelvis",
    "left_foot",
    "right_foot",
    "head",
    "left_wrist",
    "right_wrist",
)


BODY_PART_JOINTS = {
    "pelvis": (0,),
    "left_foot": (10,),
    "l_foot": (10,),
    "right_foot": (11,),
    "r_foot": (11,),
    "head": (15,),
    "left_wrist": (20,),
    "right_wrist": (21,),
    "lower": (0, 10, 11),
    "upper": (15, 20, 21),
    "left_arm": (20, 18),
    "right_arm": (21, 19),
    "legs": (10, 4, 11, 5),
    "spine": (0,),
    "full_body": CONTROL_JOINT_IDS,
    "all": CONTROL_JOINT_IDS,
}


def relative_hml263_positions(motion: torch.Tensor) -> torch.Tensor:
    """Return MaskControl's differentiable relative-joint proxy.

    The STMC extension in MaskControl optimizes HML263 channels 1:67 as a
    22x3 tensor rather than recovering global joints.  Channel zero is added
    to the proxy root so gradients also reach root angular velocity.
    """

    if motion.ndim != 3 or motion.shape[-1] != 263:
        raise ValueError(f"motion must have shape (B,T,263), got {tuple(motion.shape)}")
    relative = motion[..., 1 : (22 - 1) * 3 + 4].clone()
    relative[..., :3] = relative[..., :3] + motion[..., :1]
    return relative.reshape(*relative.shape[:-1], 22, 3)
