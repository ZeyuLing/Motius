"""SAM image/hand inference orchestration, separate from neural network definitions."""

from __future__ import annotations
from typing import Any, Dict
import numpy as np
import roma
import torch
from motius.pipelines.flowhmr.sam_data.utils.prepare_batch import prepare_batch
from motius.models.vision.sam3d_body.modules.mhr_utils import (
    fix_wrist_euler,
    rotation_angle_difference,
)
from motius.utils.sam3d import recursive_to

from motius.motion.conditioning.sam3d_images import prepare_sam3d_images


def _forward_step(model, batch, decoder_type):
    prepare_sam3d_images(model, batch, unit_rgb=batch.get('_unit_rgb', False))
    return model.forward_step(batch, decoder_type=decoder_type)


class _HandDecoderComplete(Exception):
    """A feature-only hand pass has reached its final required operation."""


def run_conditioning_inference(model, img, batch, transform_hand, *, cache=False):
    """Run original body/hand decoders, omitting post-decoder fusion work.

    Decoder-internal intermediate pose predictions remain intact because they
    condition subsequent decoder layers. Body output supplies original hand boxes.
    Decoder capture hooks run before the temporary early-return hook, so the
    conditioning tensors are the same tensors consumed by the full pipeline.
    """
    batch.update(_conditioning_only=cache is not False,
                 _conditioning_cache=cache if isinstance(cache, dict) else None)
    body = _forward_step(model, batch, decoder_type="body")
    left, right = _get_hand_box(model, body, batch)
    width = img.shape[1]
    original_left = left.copy()
    left[:, 0] = width-original_left[:, 2]-1
    left[:, 2] = width-original_left[:, 0]-1

    def hand_complete(_module, _args, _output):
        raise _HandDecoderComplete

    for hand_image, boxes in ((img[:, ::-1], left), (img, right)):
        hand_batch = prepare_batch(hand_image, transform_hand, boxes,
                                   cam_int=batch["cam_int"].clone())
        hand_batch = recursive_to(hand_batch, batch["img"].device)
        hand_batch.update(_conditioning_only=cache is not False,
                          _conditioning_cache=cache if isinstance(cache, dict) else None)
        handle = model.decoder_hand.register_forward_hook(hand_complete)
        try:
            _forward_step(model, hand_batch, decoder_type="hand")
        except _HandDecoderComplete:
            pass
        finally:
            handle.remove()


@torch.no_grad()
def run_official_frame_batch(estimator, images, boxes, cameras, *, cache=False):
    """Batch the official CPU crops and original separate hand decoder calls.

    Each frame keeps its own crop, intrinsic and detected hand boxes.  No GPU
    crop implementation or merged-hand decoder is used in this path.
    """
    model = estimator.model

    def combine(batches):
        keys = [key for key, value in batches[0].items() if isinstance(value, torch.Tensor)]
        merged = {key: torch.cat([batch[key] for batch in batches], dim=0) for key in keys}
        merged['img_ori'] = [item for batch in batches for item in batch['img_ori']]
        return recursive_to(merged, estimator.device)

    bodies = [prepare_batch(image, estimator.transform, np.asarray(box).reshape(1, 4),
                            cam_int=torch.as_tensor(camera).reshape(1, 3, 3))
              for image, box, camera in zip(images, boxes, cameras)]
    batch = combine(bodies)
    batch.update(_conditioning_only=cache is not False,
                 _conditioning_cache=cache if isinstance(cache, dict) else None)
    model._initialize_batch(batch)
    try:
        body = _forward_step(model, batch, decoder_type='body')
        left_boxes, right_boxes = _get_hand_box(model, body, batch)
        for side in ('left', 'right'):
            hand_batches = []
            for index, (image, camera) in enumerate(zip(images, cameras)):
                box = (left_boxes if side == 'left' else right_boxes)[index].copy()
                if side == 'left':
                    box[[0, 2]] = image.shape[1] - box[[2, 0]] - 1
                    image = image[:, ::-1]
                hand_batches.append(prepare_batch(image, estimator.transform_hand, box[None],
                    cam_int=torch.as_tensor(camera).reshape(1, 3, 3)))
            hand_batch = combine(hand_batches)
            hand_batch.update(_conditioning_only=cache is not False,
                              _conditioning_cache=cache if isinstance(cache, dict) else None)
            model._initialize_batch(hand_batch)
            def hand_complete(_module, _args, _output):
                raise _HandDecoderComplete
            handle = model.decoder_hand.register_forward_hook(hand_complete)
            try:
                _forward_step(model, hand_batch, decoder_type='hand')
            except _HandDecoderComplete:
                pass
            finally:
                handle.remove()
    finally:
        model._initialize_batch(batch)


@torch.no_grad()
def run_fast_conditioning_inference(estimator, img, boxes, camera, *, cache=None):
    """Original body first pass, GPU crops and one [left,right] hand pass."""
    from .gpu_preprocess import prepare_gpu_batch, merge_hands
    model = estimator.model
    image = torch.as_tensor(np.ascontiguousarray(img), device=estimator.device)
    batch = prepare_gpu_batch(image, boxes, camera, estimator.transform)
    batch.update(_conditioning_only=cache is not False,
                 _conditioning_cache=cache if isinstance(cache, dict) else None)
    model._initialize_batch(batch)
    try:
        body = _forward_step(model, batch, decoder_type='body')
        left_boxes, right_boxes = _get_hand_box(model, body, batch)
        width = image.shape[1]
        original = left_boxes.copy()
        left_boxes[:, 0] = width-original[:, 2]-1
        left_boxes[:, 2] = width-original[:, 0]-1
        left = prepare_gpu_batch(image.flip(1), left_boxes, camera, estimator.transform_hand)
        right = prepare_gpu_batch(image, right_boxes, camera, estimator.transform_hand)
        hands = merge_hands(left, right)
        hands.update(_conditioning_only=cache is not False,
                     _conditioning_cache=cache if isinstance(cache, dict) else None)
        model._initialize_batch(hands)
        # The token extractor splits the merged decoder result in original order
        # and ends the inference after capturing both actual hand features.
        return _forward_step(model, hands, decoder_type='hand')
    finally:
        model._initialize_batch(batch)


@torch.no_grad()
def run_fast_frame_batch(estimator, images, boxes, cameras, *, cache=None):
    """Independent video frames on the batch axis; every frame keeps its own K."""
    from .gpu_preprocess import prepare_gpu_batch
    model = estimator.model
    gpu_images = [torch.as_tensor(np.ascontiguousarray(image), device=estimator.device) for image in images]
    def combine(batches):
        keys = [k for k, v in batches[0].items() if isinstance(v, torch.Tensor)]
        return {**{k: torch.cat([batch[k] for batch in batches], 0) for k in keys}, '_unit_rgb': True}
    bodies = [prepare_gpu_batch(image, np.asarray(box).reshape(1, 4), camera, estimator.transform)
              for image, box, camera in zip(gpu_images, boxes, cameras)]
    batch = combine(bodies)
    del bodies
    batch.update(_conditioning_only=cache is not False,
                 _conditioning_cache=cache if isinstance(cache, dict) else None)
    model._initialize_batch(batch)
    try:
        body = _forward_step(model, batch, decoder_type='body')
        left_boxes, right_boxes = _get_hand_box(model, body, batch)
        del body
        # Bound the hand decoder workspace while batching left and right hands
        # together for each frame group. Body inference still uses all frames.
        def hand_complete(_module, _args, _output):
            raise _HandDecoderComplete
        handle = model.decoder_hand.register_forward_hook(hand_complete)
        try:
            for start in range(0, len(images), 4):
                end = min(start + 4, len(images))
                left, right = [], []
                for index in range(start, end):
                    image, camera = gpu_images[index], cameras[index]
                    width = image.shape[1]
                    box = left_boxes[index].copy()
                    box[[0, 2]] = width-box[[2, 0]]-1
                    left.append(prepare_gpu_batch(image.flip(1), box[None], camera, estimator.transform_hand))
                    right.append(prepare_gpu_batch(image, right_boxes[index:index+1], camera, estimator.transform_hand))
                    gpu_images[index] = None
                del image
                hands = combine(left+right)
                del left, right
                hands.update(_conditioning_only=cache is not False,
                             _conditioning_cache=cache if isinstance(cache, dict) else None)
                model._initialize_batch(hands)
                try:
                    _forward_step(model, hands, decoder_type='hand')
                except _HandDecoderComplete:
                    pass
        finally:
            handle.remove()
    finally:
        model._initialize_batch(batch)


def run_inference(
    self,
    img,
    batch: Dict,
    inference_type: str = "full",
    transform_hand: Any = None,
    thresh_wrist_angle=1.4,
):
    """
    Run 3DB inference (optionally with hand detector).

    inference_type:
        - full: full-body inference with both body and hand decoders
        - body: inference with body decoder only (still full-body output)
        - hand: inference with hand decoder only (only hand output)
    """

    height, width = img.shape[:2]
    cam_int = batch["cam_int"].clone()

    if inference_type == "body":
        pose_output = _forward_step(self, batch, decoder_type="body")
        return pose_output
    elif inference_type == "hand":
        pose_output = _forward_step(self, batch, decoder_type="hand")
        return pose_output
    elif not inference_type == "full":
        ValueError("Invalid inference type: ", inference_type)

    # Step 1. For full-body inference, we first inference with the body decoder.
    pose_output = _forward_step(self, batch, decoder_type="body")
    left_xyxy, right_xyxy = _get_hand_box(self, pose_output, batch)
    ori_local_wrist_rotmat = roma.euler_to_rotmat(
        "XZY",
        pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(
            1, (2, 3)
        ),
    )

    # Step 2. Re-run with each hand
    ## Left... Flip image & box
    flipped_img = img[:, ::-1]
    tmp = left_xyxy.copy()
    left_xyxy[:, 0] = width - tmp[:, 2] - 1
    left_xyxy[:, 2] = width - tmp[:, 0] - 1

    batch_lhand = prepare_batch(
        flipped_img, transform_hand, left_xyxy, cam_int=cam_int.clone()
    )
    batch_lhand = recursive_to(batch_lhand, batch["img"].device)
    lhand_output = _forward_step(self, batch_lhand, decoder_type="hand")

    # Unflip output
    ## Flip scale
    ### Get MHR values
    scale_r_hands_mean = self.head_pose.scale_mean[8].item()
    scale_l_hands_mean = self.head_pose.scale_mean[9].item()
    scale_r_hands_std = self.head_pose.scale_comps[8, 8].item()
    scale_l_hands_std = self.head_pose.scale_comps[9, 9].item()
    ### Apply
    lhand_output["mhr_hand"]["scale"][:, 9] = (
        (
            scale_r_hands_mean
            + scale_r_hands_std * lhand_output["mhr_hand"]["scale"][:, 8]
        )
        - scale_l_hands_mean
    ) / scale_l_hands_std
    ## Get the right hand global rotation, flip it, put it in as left.
    lhand_output["mhr_hand"]["joint_global_rots"][:, 78] = lhand_output["mhr_hand"][
        "joint_global_rots"
    ][:, 42].clone()
    lhand_output["mhr_hand"]["joint_global_rots"][:, 78, [1, 2], :] *= -1
    ### Flip hand pose
    lhand_output["mhr_hand"]["hand"][:, :54] = lhand_output["mhr_hand"]["hand"][:, 54:]
    ### Unflip box
    batch_lhand["bbox_center"][:, :, 0] = (
        width - batch_lhand["bbox_center"][:, :, 0] - 1
    )

    ## Right...
    batch_rhand = prepare_batch(
        img, transform_hand, right_xyxy, cam_int=cam_int.clone()
    )
    batch_rhand = recursive_to(batch_rhand, batch["img"].device)
    rhand_output = _forward_step(self, batch_rhand, decoder_type="hand")

    # Step 3. replace hand pose estimation from the body decoder.
    ## CRITERIA 1: LOCAL WRIST POSE DIFFERENCE
    joint_rotations = pose_output["mhr"]["joint_global_rots"]
    ### Get lowarm
    lowarm_joint_idxs = torch.tensor([76, 40], device=joint_rotations.device)  # left, right
    lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]  # B x 2 x 3 x 3
    ### Get zero-wrist pose
    wrist_twist_joint_idxs = torch.tensor([77, 41], device=joint_rotations.device)  # left, right
    wrist_zero_rot_pose = (
        lowarm_joint_rotations @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
    )
    ### Get globals from left & right
    left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
    right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
    pred_global_wrist_rotmat = torch.stack(
        [
            left_joint_global_rots[:, 78],
            right_joint_global_rots[:, 42],
        ],
        dim=1,
    )
    ### Get the local poses that lead to the wrist being pred_global_wrist_rotmat
    fused_local_wrist_rotmat = torch.einsum(
        "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
    )
    angle_difference = rotation_angle_difference(
        ori_local_wrist_rotmat, fused_local_wrist_rotmat
    )  # B x 2 x 3 x3
    angle_difference_valid_mask = angle_difference < thresh_wrist_angle

    ## CRITERIA 2: hand box size
    hand_box_size_thresh = 64
    hand_box_size_valid_mask = torch.stack(
        [
            (batch_lhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(dim=1),
            (batch_rhand["bbox_scale"].flatten(0, 1) > hand_box_size_thresh).all(dim=1),
        ],
        dim=1,
    )

    ## CRITERIA 3: all hand 2D KPS (including wrist) inside of box.
    hand_kps2d_thresh = 0.5
    hand_kps2d_valid_mask = torch.stack(
        [
            lhand_output["mhr_hand"]["pred_keypoints_2d_cropped"].abs().amax(dim=(1, 2))
            < hand_kps2d_thresh,
            rhand_output["mhr_hand"]["pred_keypoints_2d_cropped"].abs().amax(dim=(1, 2))
            < hand_kps2d_thresh,
        ],
        dim=1,
    )

    ## CRITERIA 4: 2D wrist distance.
    hand_wrist_kps2d_thresh = 0.25
    kps_right_wrist_idx = 41
    kps_left_wrist_idx = 62
    right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1  # Flip left hand
    body_right_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    body_left_kps_full = pose_output["mhr"]["pred_keypoints_2d"][
        :, [kps_left_wrist_idx]
    ].clone()
    right_kps_dist = (right_kps_full - body_right_kps_full).flatten(0, 1).norm(
        dim=-1
    ) / batch_lhand["bbox_scale"].flatten(0, 1)[:, 0]
    left_kps_dist = (left_kps_full - body_left_kps_full).flatten(0, 1).norm(
        dim=-1
    ) / batch_rhand["bbox_scale"].flatten(0, 1)[:, 0]
    hand_wrist_kps2d_valid_mask = torch.stack(
        [
            left_kps_dist < hand_wrist_kps2d_thresh,
            right_kps_dist < hand_wrist_kps2d_thresh,
        ],
        dim=1,
    )
    ## Left-right
    hand_valid_mask = (
        angle_difference_valid_mask
        & hand_box_size_valid_mask
        & hand_kps2d_valid_mask
        & hand_wrist_kps2d_valid_mask
    )

    # Keypoint prompting with the body decoder.
    # We use the wrist location from the hand decoder and the elbow location
    # from the body decoder as prompts to get an updated body pose estimation.
    batch_size, num_person = batch["img"].shape[:2]
    self.hand_batch_idx = []
    self.body_batch_idx = list(range(batch_size * num_person))

    ## Get right & left wrist keypoints from crops; full image. Each are B x 1 x 2
    kps_right_wrist_idx = 41
    kps_left_wrist_idx = 62
    right_kps_full = rhand_output["mhr_hand"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    left_kps_full = lhand_output["mhr_hand"]["pred_keypoints_2d"][
        :, [kps_right_wrist_idx]
    ].clone()
    left_kps_full[:, :, 0] = width - left_kps_full[:, :, 0] - 1  # Flip left hand

    # Next, get them to crop-normalized space.
    right_kps_crop = self._full_to_crop(batch, right_kps_full)
    left_kps_crop = self._full_to_crop(batch, left_kps_full)

    # Get right & left elbow keypoints from crops; full image. Each are B x 1 x 2
    kps_right_elbow_idx = 8
    kps_left_elbow_idx = 7
    right_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
        :, [kps_right_elbow_idx]
    ].clone()
    left_kps_elbow_full = pose_output["mhr"]["pred_keypoints_2d"][
        :, [kps_left_elbow_idx]
    ].clone()

    # Next, get them to crop-normalized space.
    right_kps_elbow_crop = self._full_to_crop(batch, right_kps_elbow_full)
    left_kps_elbow_crop = self._full_to_crop(batch, left_kps_elbow_full)

    # Assemble them into keypoint prompts
    keypoint_prompt = torch.cat(
        [right_kps_crop, left_kps_crop, right_kps_elbow_crop, left_kps_elbow_crop],
        dim=1,
    )
    keypoint_prompt = torch.cat([keypoint_prompt, keypoint_prompt[..., [-1]]], dim=-1)
    keypoint_prompt[:, 0, -1] = kps_right_wrist_idx
    keypoint_prompt[:, 1, -1] = kps_left_wrist_idx
    keypoint_prompt[:, 2, -1] = kps_right_elbow_idx
    keypoint_prompt[:, 3, -1] = kps_left_elbow_idx

    if keypoint_prompt.shape[0] > 1:
        # Replace invalid keypoints to dummy prompts
        invalid_prompt = (
            (keypoint_prompt[..., 0] < -0.5)
            | (keypoint_prompt[..., 0] > 0.5)
            | (keypoint_prompt[..., 1] < -0.5)
            | (keypoint_prompt[..., 1] > 0.5)
            | (~hand_valid_mask[..., [1, 0, 1, 0]])
        ).unsqueeze(-1)
        dummy_prompt = torch.zeros((1, 1, 3)).to(keypoint_prompt)
        dummy_prompt[:, :, -1] = -2
        keypoint_prompt[:, :, :2] = torch.clamp(
            keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
        )  # [-0.5, 0.5] --> [0, 1]
        keypoint_prompt = torch.where(invalid_prompt, dummy_prompt, keypoint_prompt)
    else:
        # Only keep valid keypoints
        valid_keypoint = (
            torch.all(
                (keypoint_prompt[:, :, :2] > -0.5) & (keypoint_prompt[:, :, :2] < 0.5),
                dim=2,
            )
            & hand_valid_mask[..., [1, 0, 1, 0]]
        ).squeeze()
        keypoint_prompt = keypoint_prompt[:, valid_keypoint]
        keypoint_prompt[:, :, :2] = torch.clamp(
            keypoint_prompt[:, :, :2] + 0.5, min=0.0, max=1.0
        )  # [-0.5, 0.5] --> [0, 1]

    if keypoint_prompt.numel() != 0:
        pose_output, _ = run_keypoint_prompt(self, batch, pose_output, keypoint_prompt)

    ##############################################################################

    # Drop in hand pose
    left_hand_pose_params = lhand_output["mhr_hand"]["hand"][:, :54]
    right_hand_pose_params = rhand_output["mhr_hand"]["hand"][:, 54:]
    updated_hand_pose = torch.cat(
        [left_hand_pose_params, right_hand_pose_params], dim=1
    )

    # Drop in hand scales
    updated_scale = pose_output["mhr"]["scale"].clone()
    updated_scale[:, 9] = lhand_output["mhr_hand"]["scale"][:, 9]
    updated_scale[:, 8] = rhand_output["mhr_hand"]["scale"][:, 8]
    updated_scale[:, 18:] = (
        lhand_output["mhr_hand"]["scale"][:, 18:]
        + rhand_output["mhr_hand"]["scale"][:, 18:]
    ) / 2

    # Update hand shape
    updated_shape = pose_output["mhr"]["shape"].clone()
    updated_shape[:, 40:] = (
        lhand_output["mhr_hand"]["shape"][:, 40:]
        + rhand_output["mhr_hand"]["shape"][:, 40:]
    ) / 2

    ############################ Doing IK ############################

    # First, forward just FK
    joint_rotations = self.head_pose.mhr_forward(
        global_trans=pose_output["mhr"]["global_rot"] * 0,
        global_rot=pose_output["mhr"]["global_rot"],
        body_pose_params=pose_output["mhr"]["body_pose"],
        hand_pose_params=updated_hand_pose,
        scale_params=updated_scale,
        shape_params=updated_shape,
        expr_params=pose_output["mhr"]["face"],
        return_joint_rotations=True,
    )[1]

    # Get lowarm
    lowarm_joint_idxs = torch.tensor([76, 40], device=joint_rotations.device)  # left, right
    lowarm_joint_rotations = joint_rotations[:, lowarm_joint_idxs]  # B x 2 x 3 x 3

    # Get zero-wrist pose
    wrist_twist_joint_idxs = torch.tensor([77, 41], device=joint_rotations.device)  # left, right
    wrist_zero_rot_pose = (
        lowarm_joint_rotations @ self.head_pose.joint_rotation[wrist_twist_joint_idxs]
    )

    # Get globals from left & right
    left_joint_global_rots = lhand_output["mhr_hand"]["joint_global_rots"]
    right_joint_global_rots = rhand_output["mhr_hand"]["joint_global_rots"]
    pred_global_wrist_rotmat = torch.stack(
        [
            left_joint_global_rots[:, 78],
            right_joint_global_rots[:, 42],
        ],
        dim=1,
    )

    # Now we want to get the local poses that lead to the wrist being pred_global_wrist_rotmat
    fused_local_wrist_rotmat = torch.einsum(
        "kabc,kabd->kadc", pred_global_wrist_rotmat, wrist_zero_rot_pose
    )
    wrist_xzy = fix_wrist_euler(roma.rotmat_to_euler("XZY", fused_local_wrist_rotmat))

    # Put it in.
    angle_difference = rotation_angle_difference(
        ori_local_wrist_rotmat, fused_local_wrist_rotmat
    )  # B x 2 x 3 x3
    valid_angle = angle_difference < thresh_wrist_angle
    valid_angle = valid_angle & hand_valid_mask
    valid_angle = valid_angle.unsqueeze(-1)

    body_pose = pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]].unflatten(
        1, (2, 3)
    )
    updated_body_pose = torch.where(valid_angle, wrist_xzy, body_pose)
    pose_output["mhr"]["body_pose"][:, [41, 43, 42, 31, 33, 32]] = (
        updated_body_pose.flatten(1, 2)
    )

    hand_pose = pose_output["mhr"]["hand"].unflatten(1, (2, 54))
    pose_output["mhr"]["hand"] = torch.where(
        valid_angle, updated_hand_pose.unflatten(1, (2, 54)), hand_pose
    ).flatten(1, 2)

    hand_scale = torch.stack(
        [pose_output["mhr"]["scale"][:, 9], pose_output["mhr"]["scale"][:, 8]],
        dim=1,
    )
    updated_hand_scale = torch.stack([updated_scale[:, 9], updated_scale[:, 8]], dim=1)
    masked_hand_scale = torch.where(
        valid_angle.squeeze(-1), updated_hand_scale, hand_scale
    )
    pose_output["mhr"]["scale"][:, 9] = masked_hand_scale[:, 0]
    pose_output["mhr"]["scale"][:, 8] = masked_hand_scale[:, 1]

    # Replace shared shape and scale
    pose_output["mhr"]["scale"][:, 18:] = torch.where(
        valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
        (
            lhand_output["mhr_hand"]["scale"][:, 18:] * valid_angle.squeeze(-1)[:, [0]]
            + rhand_output["mhr_hand"]["scale"][:, 18:]
            * valid_angle.squeeze(-1)[:, [1]]
        )
        / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
        pose_output["mhr"]["scale"][:, 18:],
    )
    pose_output["mhr"]["shape"][:, 40:] = torch.where(
        valid_angle.squeeze(-1).sum(dim=1, keepdim=True) > 0,
        (
            lhand_output["mhr_hand"]["shape"][:, 40:] * valid_angle.squeeze(-1)[:, [0]]
            + rhand_output["mhr_hand"]["shape"][:, 40:]
            * valid_angle.squeeze(-1)[:, [1]]
        )
        / (valid_angle.squeeze(-1).sum(dim=1, keepdim=True) + 1e-8),
        pose_output["mhr"]["shape"][:, 40:],
    )

    ########################################################

    # Re-run forward
    with torch.no_grad():
        verts, j3d, jcoords, mhr_model_params, joint_global_rots = (
            self.head_pose.mhr_forward(
                global_trans=pose_output["mhr"]["global_rot"] * 0,
                global_rot=pose_output["mhr"]["global_rot"],
                body_pose_params=pose_output["mhr"]["body_pose"],
                hand_pose_params=pose_output["mhr"]["hand"],
                scale_params=pose_output["mhr"]["scale"],
                shape_params=pose_output["mhr"]["shape"],
                expr_params=pose_output["mhr"]["face"],
                return_keypoints=True,
                return_joint_coords=True,
                return_model_params=True,
                return_joint_rotations=True,
            )
        )
        j3d = j3d[:, :70]  # 308 --> 70 keypoints
        verts[..., [1, 2]] *= -1  # Camera system difference
        j3d[..., [1, 2]] *= -1  # Camera system difference
        jcoords[..., [1, 2]] *= -1
        pose_output["mhr"]["pred_keypoints_3d"] = j3d
        pose_output["mhr"]["pred_vertices"] = verts
        pose_output["mhr"]["pred_joint_coords"] = jcoords
        pose_output["mhr"]["pred_pose_raw"][
            ...
        ] = 0  # pred_pose_raw is not valid anymore
        pose_output["mhr"]["mhr_model_params"] = mhr_model_params

    ########################################################
    # Project to 2D
    pred_keypoints_3d_proj = (
        pose_output["mhr"]["pred_keypoints_3d"]
        + pose_output["mhr"]["pred_cam_t"][:, None, :]
    )
    pred_keypoints_3d_proj[:, :, [0, 1]] *= pose_output["mhr"]["focal_length"][
        :, None, None
    ]
    pred_keypoints_3d_proj[:, :, [0, 1]] = (
        pred_keypoints_3d_proj[:, :, [0, 1]]
        + torch.FloatTensor([width / 2, height / 2]).to(pred_keypoints_3d_proj)[
            None, None, :
        ]
        * pred_keypoints_3d_proj[:, :, [2]]
    )
    pred_keypoints_3d_proj[:, :, :2] = (
        pred_keypoints_3d_proj[:, :, :2] / pred_keypoints_3d_proj[:, :, [2]]
    )
    pose_output["mhr"]["pred_keypoints_2d"] = pred_keypoints_3d_proj[:, :, :2]

    return pose_output, batch_lhand, batch_rhand, lhand_output, rhand_output


def run_keypoint_prompt(self, batch, output, keypoint_prompt):
    image_embeddings = output["image_embeddings"]
    condition_info = output["condition_info"]
    pose_output = output["mhr"]  # body-only output
    # Use previous estimate as initialization
    prev_estimate = torch.cat(
        [
            pose_output["pred_pose_raw"].detach(),  # (B, 6)
            pose_output["shape"].detach(),
            pose_output["scale"].detach(),
            pose_output["hand"].detach(),
            pose_output["face"].detach(),
        ],
        dim=1,
    ).unsqueeze(dim=1)
    if hasattr(self, "init_camera"):
        prev_estimate = torch.cat(
            [prev_estimate, pose_output["pred_cam"].detach().unsqueeze(1)],
            dim=-1,
        )

    tokens_output, pose_output = self.forward_decoder(
        image_embeddings,
        init_estimate=None,  # not recurring previous estimate
        keypoints=keypoint_prompt,
        prev_estimate=prev_estimate,
        condition_info=condition_info,
        batch=batch,
    )
    pose_output = pose_output[-1]

    output.update({"mhr": pose_output})
    return output, keypoint_prompt


def _get_hand_box(self, pose_output, batch):
    """Get hand bbox from the hand detector"""
    affine = batch["affine_trans"].flatten(0, 1)
    pred_left_hand_box = (
        pose_output["mhr"]["hand_box"][:, 0].detach().cpu().numpy()
        * self.cfg.MODEL.IMAGE_SIZE[0]
    )
    pred_right_hand_box = (
        pose_output["mhr"]["hand_box"][:, 1].detach().cpu().numpy()
        * self.cfg.MODEL.IMAGE_SIZE[0]
    )

    # Change boxes into squares
    batch["left_center"] = pred_left_hand_box[:, :2]
    batch["left_scale"] = (
        pred_left_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
    )
    batch["right_center"] = pred_right_hand_box[:, :2]
    batch["right_scale"] = (
        pred_right_hand_box[:, 2:].max(axis=1, keepdims=True).repeat(2, axis=1)
    )

    # Crop to full. batch["affine_trans"] is full-to-crop, right application
    batch["left_scale"] = (
        batch["left_scale"] / affine[:, 0, 0].cpu().numpy()[:, None]
    )
    batch["right_scale"] = (
        batch["right_scale"] / affine[:, 0, 0].cpu().numpy()[:, None]
    )
    batch["left_center"] = (
        batch["left_center"] - affine[:, [0, 1], [2, 2]].cpu().numpy()
    ) / affine[:, 0, 0].cpu().numpy()[:, None]
    batch["right_center"] = (
        batch["right_center"]
        - affine[:, [0, 1], [2, 2]].cpu().numpy()
    ) / affine[:, 0, 0].cpu().numpy()[:, None]

    left_xyxy = np.concatenate(
        [
            (batch["left_center"][:, 0] - batch["left_scale"][:, 0] * 1 / 2).reshape(
                -1, 1
            ),
            (batch["left_center"][:, 1] - batch["left_scale"][:, 1] * 1 / 2).reshape(
                -1, 1
            ),
            (batch["left_center"][:, 0] + batch["left_scale"][:, 0] * 1 / 2).reshape(
                -1, 1
            ),
            (batch["left_center"][:, 1] + batch["left_scale"][:, 1] * 1 / 2).reshape(
                -1, 1
            ),
        ],
        axis=1,
    )
    right_xyxy = np.concatenate(
        [
            (batch["right_center"][:, 0] - batch["right_scale"][:, 0] * 1 / 2).reshape(
                -1, 1
            ),
            (batch["right_center"][:, 1] - batch["right_scale"][:, 1] * 1 / 2).reshape(
                -1, 1
            ),
            (batch["right_center"][:, 0] + batch["right_scale"][:, 0] * 1 / 2).reshape(
                -1, 1
            ),
            (batch["right_center"][:, 1] + batch["right_scale"][:, 1] * 1 / 2).reshape(
                -1, 1
            ),
        ],
        axis=1,
    )

    return left_xyxy, right_xyxy
