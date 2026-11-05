# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Adapted into Motius native component boundaries from NVlabs/motionbricks
# commit 54fc563aeaaf819a753d72cc9fdc6713989cdd43. See motius/licenses/motionbricks.
import torch
import torch as t


def select_length(
    logits,
    num_tokens,
    args,
    allowed_pred_num_tokens=None,
    config=None,
    groundtruth_num_tokens=None,
    out_of_reach=None,
    masked=None,
):
    config = config or {}
    num_token_logits = logits
    batch_size, device = logits.shape[0], logits.device
    if groundtruth_num_tokens is not None:
        chosen_token = groundtruth_num_tokens.view([batch_size]) - args["min_tokens"]
    else:
        if not config.get("allow_pred_out_of_reach_num_tokens", True):
            num_token_logits = num_token_logits.clone()
            num_token_logits[:, out_of_reach - args["min_tokens"]] = -t.inf
        if allowed_pred_num_tokens is not None:
            num_time_tokens = args["max_tokens"] - args["min_tokens"] + 1
            modified_num_token_logits = t.where(
                allowed_pred_num_tokens == 1,
                num_token_logits[:, :num_time_tokens],
                t.full([batch_size, num_time_tokens], -t.inf).to(device),
            )
            num_token_logits = t.cat(
                [modified_num_token_logits, num_token_logits[:, num_time_tokens:]],
                dim=-1,
            )
        chosen_token = torch.argmax(num_token_logits, dim=-1).int()
        chosen_token = t.where(
            num_tokens.view([batch_size]) == masked,
            chosen_token,
            num_tokens.view([batch_size]) - args["min_tokens"],
        )
    return chosen_token, num_token_logits
