# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"Select root and pose features and expand pose constraints."

from __future__ import annotations


import numpy as np
import torch as t
from torch import nn


def convert_sparse_cond_to_dense_cond_if_needed(
    cond: t.Tensor, has_cond: t.Tensor, num_max_cond: int
):
    """Return conditions as a dense [batch, slot, feature] tensor with a boolean mask.

    A boolean has_cond is already dense. Integer has_cond lists the slot of each
    condition.
    """
    assert cond.shape[1] == has_cond.shape[1]
    batch_size, feat_dim = (cond.shape[0], cond.shape[2])
    if has_cond.dtype == t.bool:
        assert cond.shape[1] == num_max_cond
        return (cond, has_cond)
    else:
        assert cond.shape[1] <= num_max_cond
        has_cond_map = t.nn.functional.one_hot(has_cond, num_classes=num_max_cond)
        dense_has_cond = (
            has_cond_map.sum(dim=1)
            .reshape([batch_size, num_max_cond, 1])
            .tile([1, 1, feat_dim])
            .bool()
        )
        dense_cond = t.zeros([batch_size, num_max_cond, feat_dim]).to(
            cond.device, cond.dtype
        )
        dense_cond[dense_has_cond] = cond.view(-1)
        return (dense_cond, dense_has_cond)


def extract_feature_from_motion_rep(
    x: t.Tensor | None,
    motion_rep: nn.Module,
    feature: str,
    fetch_feat_idx: bool = False,
):
    """Select a named feature from a motion representation.

    Names that end in _with_mask append a zero mask column. With fetch_feat_idx, it
    returns the feature's indices instead.
    """
    if feature in [
        "root_with_mask",
        "root_without_heading_with_mask",
        "root_without_hip_height_with_mask",
        "root_without_hip_height_without_heading_with_mask",
    ]:
        if fetch_feat_idx:
            raise NotImplementedError(
                "fetch_feat_idx does not support with_mask features."
            )
        else:
            raw_feature = extract_feature_from_motion_rep(
                x, motion_rep, feature.split("_with_mask")[0]
            )
            mask = t.zeros([x.shape[0], x.shape[1], 1]).to(x.device)
            return t.concat([raw_feature, mask], dim=-1)
    elif feature == "root":
        if fetch_feat_idx:
            return motion_rep.indices["root"]
        else:
            return (
                x
                if x.shape[-1] == len(motion_rep.indices["root"])
                else x[:, :, motion_rep.indices["root"]]
            )
    elif feature == "root_without_heading":
        if motion_rep.root_mode == "global":
            feat_idx = np.concatenate(
                [
                    motion_rep.indices["global_root_pos_2d"],
                    np.array(
                        [
                            i
                            for i in motion_rep.indices["global_root_pos"]
                            if i not in motion_rep.indices["global_root_pos_2d"]
                        ]
                    ),
                ]
            )
        else:
            feat_idx = np.concatenate(
                [
                    motion_rep.indices["local_root_vel"],
                    motion_rep.indices["global_root_y"],
                ]
            )
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]
    elif feature == "root_without_hip_height":
        if motion_rep.root_mode == "global":
            feat_idx = np.concatenate(
                [
                    motion_rep.indices["global_root_pos_2d"],
                    motion_rep.indices["global_root_heading"],
                ]
            )
        else:
            feat_idx = np.concatenate(
                [
                    motion_rep.indices["local_root_rot_vel"],
                    motion_rep.indices["local_root_vel"],
                ]
            )
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]
    elif feature == "root_without_hip_height_without_heading":
        if motion_rep.root_mode == "global":
            feat_idx = motion_rep.indices["global_root_pos_2d"]
        else:
            feat_idx = motion_rep.indices["local_root_vel"]
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]
    elif feature == "pose":
        if fetch_feat_idx:
            return motion_rep.indices["all"]
        else:
            assert x.shape[-1] == len(motion_rep.indices["all"])
            return x
    elif feature == "joint_positions_and_rotations":
        feat_idx = np.concatenate(
            [motion_rep.indices["ric_data"], motion_rep.indices["global_rot_data"]]
        )
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]
    elif feature == "joint_positions_and_rotations_and_hip_height":
        if motion_rep.root_mode == "global":
            feat_idx = np.concatenate(
                [
                    np.array(
                        [
                            i
                            for i in motion_rep.indices["global_root_pos"]
                            if i not in motion_rep.indices["global_root_pos_2d"]
                        ]
                    ),
                    motion_rep.indices["ric_data"],
                    motion_rep.indices["global_rot_data"],
                ]
            )
        else:
            feat_idx = np.concatenate(
                [
                    motion_rep.indices["global_root_y"],
                    motion_rep.indices["ric_data"],
                    motion_rep.indices["global_rot_data"],
                ]
            )
        if fetch_feat_idx:
            return feat_idx
        else:
            return x if x.shape[-1] == len(feat_idx) else x[:, :, feat_idx]
    else:
        raise NotImplementedError


def numeric_mask_union(*masks):
    if not masks:
        raise ValueError("At least one mask is required")
    result = masks[0]
    for mask in masks[1:]:
        result = t.maximum(result, mask)
    return result


def release_input_shapes(model):
    common = {
        "has_global_root_values": (8,),
        "has_local_root_values": (8,),
        "has_local_poses": (8,),
        "allowed_pred_num_tokens": (11,),
    }
    common.update(
        {"context_qpos": (4, 36), "target_qpos": (4, 36)}
        if model == "g1"
        else {
            "global_root_values": (8, 5),
            "local_root_values": (8, 4),
            "local_poses": (8, 267),
        }
    )
    return common


def validate_release_inputs(inputs, model, device):
    shapes = release_input_shapes(model)
    if set(inputs) != set(shapes):
        raise ValueError(f"Expected exactly these inputs: {sorted(shapes)}")
    feed, batch = ({}, None)
    for name, shape in shapes.items():
        value = inputs[name]
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value)
        if (
            value.ndim != len(shape) + 1
            or value.shape[1:] != shape
            or value.shape[0] < 1
        ):
            raise ValueError(f"{name} must have shape (N, {shape}), N > 0")
        if batch is None:
            batch = value.shape[0]
        if value.shape[0] != batch:
            raise ValueError("All inputs must have the same batch size")
        mask = name.startswith("has_") or name == "allowed_pred_num_tokens"
        if mask and (not np.isin(value, (0, 1)).all()):
            raise ValueError(f"{name} must contain binary flags")
        value = np.ascontiguousarray(value, dtype=np.int32 if mask else np.float32)
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain finite values")
        if (
            name.endswith("_qpos")
            and (np.linalg.norm(value[..., 3:7], axis=-1) == 0).any()
        ):
            raise ValueError(f"{name} contains zero root quaternions")
        feed[name] = value
    if not feed["allowed_pred_num_tokens"].any(axis=1).all():
        raise ValueError("Each request must allow at least one token count")
    return {name: t.as_tensor(value, device=device) for name, value in feed.items()}
