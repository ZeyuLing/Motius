# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: I001
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from gem.utils.video_io_utils import read_video_np


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

        project_root = Path(__file__).resolve().parents[2]
        sam3d_root = project_root / "third_party" / "sam-3d-body"
        if str(sam3d_root) not in sys.path:
            sys.path.insert(0, str(sam3d_root))

        from sam_3d_body import SAM3DBodyEstimator, load_sam_3d_body  # type: ignore[reportMissingImports]

        if checkpoint_path is None:
            from gem.utils.hf_utils import download_sam3d_checkpoint

            checkpoint_path = download_sam3d_checkpoint()
        if mhr_path is None:
            from gem.utils.hf_utils import download_mhr_model

            mhr_path = download_mhr_model()

        model, model_cfg = load_sam_3d_body(
            checkpoint_path,
            device=torch.device(device),
            mhr_path=mhr_path,
        )

        # Match legacy extractor semantics: use a single primary body token.
        # With hand-detect tokens enabled, token layout changes and features drift.
        model.cfg.defrost()
        model.cfg.MODEL.DECODER.DO_HAND_DETECT_TOKENS = False
        model.cfg.freeze()

        self.estimator = SAM3DBodyEstimator(
            sam_3d_body_model=model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )
        self._patch_model_to_expose_pose_token()

    def _patch_model_to_expose_pose_token(self):
        model = self.estimator.model
        original_forward_pose_branch = model.forward_pose_branch

        def _forward_pose_branch_with_pose_token(batch):
            out = original_forward_pose_branch(batch)
            if "pose_token" in out:
                return out

            # Upstream sam_3d_body removed pose_token from output dict.
            # Reconstruct it from decoder inputs to preserve old feature semantics.
            body_batch_idx = getattr(model, "body_batch_idx", [])
            if len(body_batch_idx) > 0:
                batch_size, num_person = batch["img"].shape[:2]
                keypoints_prompt = torch.zeros((batch_size * num_person, 1, 3)).to(batch["img"])
                keypoints_prompt[:, :, -1] = -2
                tokens_output, _ = model.forward_decoder(
                    out["image_embeddings"][body_batch_idx],
                    init_estimate=None,
                    keypoints=keypoints_prompt[body_batch_idx],
                    prev_estimate=None,
                    condition_info=out["condition_info"][body_batch_idx],
                    batch=batch,
                )
                out["pose_token"] = tokens_output
            else:
                out["pose_token"] = None

            hand_batch_idx = getattr(model, "hand_batch_idx", [])
            if len(hand_batch_idx) > 0:
                batch_size, num_person = batch["img"].shape[:2]
                keypoints_prompt = torch.zeros((batch_size * num_person, 1, 3)).to(batch["img"])
                keypoints_prompt[:, :, -1] = -2
                tokens_output_hand, _ = model.forward_decoder_hand(
                    out["image_embeddings"][hand_batch_idx],
                    init_estimate=None,
                    keypoints=keypoints_prompt[hand_batch_idx],
                    prev_estimate=None,
                    condition_info=out["condition_info"][hand_batch_idx],
                    batch=batch,
                )
                out["pose_token_hand"] = tokens_output_hand
            else:
                out["pose_token_hand"] = None

            return out

        model.forward_pose_branch = _forward_pose_branch_with_pose_token

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
        from sam_3d_body.data.utils.prepare_batch import prepare_batch  # type: ignore[reportMissingImports]
        from sam_3d_body.utils import recursive_to  # type: ignore[reportMissingImports]

        imgs = read_video_np(video_path, scale=img_ds)
        bbx_xys = torch.as_tensor(bbx_xys).float().cpu().numpy()

        tokens = []
        transls = []
        for i in tqdm(
            range(0, len(imgs), batch_size), desc="SAM3D Body", disable=not self.tqdm_leave
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
                    collated_batch[key] = np.concatenate([d[key] for d in model_batch_list])
                else:
                    collated_batch[key] = [d[key] for d in model_batch_list]

            collated_batch = recursive_to(collated_batch, self.device)
            self.estimator.model._initialize_batch(collated_batch)
            with torch.no_grad():
                output = self.estimator.model.forward_step(collated_batch, decoder_type="body")

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
