"""SONIC observation layout and modality feature packing outside neural models."""

from collections import OrderedDict
import math
import torch

MODALITY_FIELDS = {
    "g1": ("command_multi_future_nonflat", "motion_anchor_ori_b_mf_nonflat"),
    "teleop": (
        "command_multi_future_lower_body",
        "vr_3point_local_target",
        "vr_3point_local_orn_target",
        "motion_anchor_ori_b",
    ),
    "smpl": (
        "smpl_joints_multi_future_local_nonflat",
        "smpl_root_ori_b_multi_future",
        "joint_pos_multi_future_wrist_for_smpl",
    ),
}


class SonicObservationProcessor:
    def __init__(self, schema):
        self.schema = OrderedDict(
            (name, tuple(shape) if not isinstance(shape, int) else (shape,))
            for name, shape in schema.items()
        )
        self.width = sum(math.prod(shape) for shape in self.schema.values())
        if "encoder_index" not in self.schema or self.schema["encoder_index"] != (3,):
            raise ValueError(
                "The released SONIC policy requires three modality selector bits"
            )
        missing = {
            name for fields in MODALITY_FIELDS.values() for name in fields
        } - self.schema.keys()
        if missing:
            raise ValueError(
                "Missing SONIC observation fields: " + str(sorted(missing))
            )

    def __call__(self, observations):
        values = observations["tokenizer_obs"]
        if values.ndim == 2:
            values = values[:, None]
        if values.shape[-1] != self.width:
            raise ValueError("SONIC observation schema does not match tokenizer_obs")
        batch, sequence = values.shape[:2]
        fields = {}
        offset = 0
        for name, shape in self.schema.items():
            size = math.prod(shape)
            fields[name] = values[..., offset : offset + size].view(
                batch, sequence, *shape
            )
            offset += size
        modes = fields["encoder_index"].view(batch * sequence, 3)
        features = {
            name: torch.cat([fields[key] for key in keys], dim=-1)
            for name, keys in MODALITY_FIELDS.items()
        }
        masks = {
            name: modes[:, index].bool() for index, name in enumerate(MODALITY_FIELDS)
        }
        proprioception = observations["actor_obs"]
        critic = observations["critic_obs"]
        if proprioception.ndim == 2:
            proprioception = proprioception[:, None]
        if critic.ndim == 2:
            critic = critic[:, None]
        return dict(
            features=features,
            encoder_masks=masks,
            proprioception=proprioception,
            critic_observations=critic,
        )
