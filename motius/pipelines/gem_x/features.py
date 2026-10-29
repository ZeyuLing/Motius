# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from motius.pipelines.gem_video import read_video_np


from motius.motion.conditioning.sam3d_images import prepare_sam3d_images


def _encode_primary_body(bundle, batch):
    """GEM-X's original body-only token graph, including its second decoder pass."""
    model = bundle.model
    prepare_sam3d_images(model, batch)
    output = bundle.predict(batch, decoder_type="body")
    if "pose_token" not in output:
        batch_size, people = batch["img"].shape[:2]
        prompts = torch.zeros((batch_size * people, 1, 3)).to(batch["img"])
        prompts[:, :, -1] = -2
        indices = model.body_batch_idx
        tokens, _ = model.forward_decoder(
            output["image_embeddings"][indices],
            init_estimate=None,
            keypoints=prompts[indices],
            prev_estimate=None,
            condition_info=output["condition_info"][indices],
            batch=batch,
        )
        output["pose_token"] = tokens
    return output


class SAM3DBExtractor:
    def __init__(
        self,
        checkpoint_path=None,
        mhr_path=None,
        device="cuda:0",
        tqdm_leave=True,
        feature_dim=1024,
    ):
        self.device = device
        self.tqdm_leave = tqdm_leave
        self.feature_dim = feature_dim

        from motius.models.vision.sam3d_body.bundle import SAM3DBodyBundle
        from motius.pipelines.flowhmr.sam.sam_3d_body_estimator import (
            SAM3DBodyEstimator,
        )

        self.bundle = SAM3DBodyBundle(checkpoint_path, mhr_path).to(device).eval()
        self.bundle.config.defrost()
        self.bundle.config.MODEL.DECODER.DO_HAND_DETECT_TOKENS = False
        self.bundle.config.freeze()
        self.estimator = SAM3DBodyEstimator(self.bundle.model, self.bundle.config)

    @staticmethod
    def _bbox_xys_to_xyxy(b):
        cx, cy, s = b
        hs = float(s) * 0.5
        return np.array([cx - hs, cy - hs, cx + hs, cy + hs], dtype=np.float32)

    def _to_feature(self, out):
        vec = out.get("mhr_model_params", None)
        if vec is None:
            return torch.zeros(self.feature_dim, dtype=torch.float32)
        vec = torch.as_tensor(vec, dtype=torch.float32).flatten()
        if vec.numel() >= self.feature_dim:
            return vec[: self.feature_dim]
        pad = torch.zeros(self.feature_dim - vec.numel(), dtype=torch.float32)
        return torch.cat([vec, pad], dim=0)

    def extract_video_features(
        self,
        video_path,
        bbx_xys,
        img_ds=1.0,
        batch_size=16,
        render_mhr=False,
    ):
        del render_mhr
        from motius.pipelines.flowhmr.sam_data.utils.prepare_batch import prepare_batch  # type: ignore[reportMissingImports]
        from motius.utils.sam3d import recursive_to  # type: ignore[reportMissingImports]

        imgs = read_video_np(video_path, scale=img_ds)
        bbx_xys = torch.as_tensor(bbx_xys).float().cpu().numpy()

        tokens = []
        transls = []
        for i in tqdm(
            range(0, len(imgs), batch_size),
            desc="SAM3D Body",
            disable=not self.tqdm_leave,
        ):
            batch_imgs = imgs[i : i + batch_size]
            batch_bbx = bbx_xys[i : i + batch_size]
            model_batch_list = []

            for j, img in enumerate(batch_imgs):
                cx, cy, s = batch_bbx[j]
                s_scaled = s * img_ds
                cx_scaled = cx * img_ds
                cy_scaled = cy * img_ds
                box = np.array(
                    [
                        cx_scaled - s_scaled / 2,
                        cy_scaled - s_scaled / 2,
                        cx_scaled + s_scaled / 2,
                        cy_scaled + s_scaled / 2,
                    ],
                    dtype=np.float32,
                ).reshape(1, 4)
                data = prepare_batch(
                    img, self.estimator.transform, box, masks=None, masks_score=None
                )
                model_batch_list.append(data)

            if len(model_batch_list) == 0:
                continue

            collated_batch = {}
            keys = model_batch_list[0].keys()
            for key in keys:
                val0 = model_batch_list[0][key]
                if isinstance(val0, torch.Tensor):
                    collated_batch[key] = torch.cat([d[key] for d in model_batch_list])
                elif isinstance(val0, np.ndarray):
                    collated_batch[key] = np.concatenate(
                        [d[key] for d in model_batch_list]
                    )
                else:
                    collated_batch[key] = [d[key] for d in model_batch_list]

            collated_batch = recursive_to(collated_batch, self.device)
            with torch.no_grad():
                output = _encode_primary_body(self.bundle, collated_batch)

            pose_token = output.get("pose_token", None)
            if pose_token is None:
                # Fallback path when pose_token cannot be reconstructed.
                fallback = output["mhr"]["mhr_model_params"].detach().float().cpu()
                if fallback.shape[-1] >= self.feature_dim:
                    pose_token = fallback[:, : self.feature_dim]
                else:
                    pad = torch.zeros(
                        fallback.shape[0],
                        self.feature_dim - fallback.shape[-1],
                        dtype=torch.float32,
                    )
                    pose_token = torch.cat([fallback, pad], dim=-1)
            else:
                pose_token = pose_token.detach().float().cpu()
                if pose_token.ndim == 3:
                    # Keep only the primary body token; extra tokens are detector prompts.
                    pose_token = pose_token[:, 0]

            pred_cam_t = output["mhr"]["pred_cam_t"].detach().float().cpu()
            tokens.append(pose_token)
            transls.append(pred_cam_t)

        return {
            "pose_tokens": torch.cat(tokens, dim=0),
            "transls": torch.cat(transls, dim=0),
            "rendered_imgs": [],
        }
