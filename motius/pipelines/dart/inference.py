from typing import Sequence, List, Optional
import random
import numpy as np
import torch

_STANDARD_INITIAL_JOINTS = [
    [0.0012, -0.3668, 0.9377],
    [0.001, -0.4273, 0.8429],
    [-0.0034, -0.3135, 0.829],
]

_MBENCH_COORD_CONVERSION = torch.tensor(
    [[-1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]], dtype=torch.float32
)


class DARTInference:

    def _get_dataset(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size in self._dataset_cache:
            return self._dataset_cache[batch_size]
        from motius.motion.conditioning.dart_seed import DARTSeedData

        self.bundle.load_text_encoder()
        dataset = DARTSeedData(
            cfg_path=self.artifact_dir / self.vae_args.data_args.cfg_path,
            dataset_path=self.artifact_dir / self.vae_args.data_args.data_dir,
            body_type=self.vae_args.data_args.body_type,
            sequence_path=self.artifact_dir / self.seed_sequence,
            batch_size=batch_size,
            device=self.device,
            enforce_gender="male",
            enforce_zero_beta=1,
            clip_model=self.bundle.clip_model,
        )
        self._dataset_cache[batch_size] = dataset
        return dataset

    def _seed_everything(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)

    @staticmethod
    def _to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _motion_sequence_to_motion135(self, sequence: dict) -> np.ndarray:
        global_orient = sequence["global_orient"]
        body_pose = sequence["body_pose"]
        transl = sequence["transl"]
        joints = sequence.get("joints")
        if not isinstance(global_orient, torch.Tensor):
            global_orient = torch.as_tensor(global_orient)
        if not isinstance(body_pose, torch.Tensor):
            body_pose = torch.as_tensor(body_pose)
        if not isinstance(transl, torch.Tensor):
            transl = torch.as_tensor(transl)
        if joints is not None and (not isinstance(joints, torch.Tensor)):
            joints = torch.as_tensor(joints)
        global_orient = global_orient.reshape(-1, 3, 3).float()
        body_pose = body_pose.reshape(-1, 21, 3, 3).float()
        transl = transl.reshape(-1, 3).float()
        if joints is not None:
            joints = joints.reshape(-1, 22, 3).float()
        if self.coord_conversion == "mbench":
            conv = _MBENCH_COORD_CONVERSION.to(
                device=global_orient.device, dtype=global_orient.dtype
            )
            transl = torch.einsum("ij,tj->ti", conv, transl)
            if joints is not None:
                joints = torch.einsum("ij,tkj->tki", conv, joints)
            global_orient = torch.einsum("ij,tjk->tik", conv, global_orient)
        if self.translation_source in {"joints_pelvis", "floor_aligned_joints_pelvis"}:
            if joints is None:
                raise ValueError(
                    f"translation_source={self.translation_source!r} requires sequence['joints']"
                )
            transl = joints[:, 0, :]
        if self.translation_source.startswith("floor_aligned_"):
            if joints is None:
                raise ValueError(
                    f"translation_source={self.translation_source!r} requires sequence['joints']"
                )
            vertical_idx = 1 if self.coord_conversion == "mbench" else 2
            floor = joints[:, :, vertical_idx].amin(dim=1)
            transl = transl.clone()
            transl[:, vertical_idx] = transl[:, vertical_idx] - floor
        rot = torch.cat([global_orient.reshape(-1, 1, 3, 3), body_pose], dim=1)
        rot6d = rot.float()[..., :2, :].clone().reshape(rot.shape[0], 132)
        motion135 = torch.cat([transl.reshape(rot.shape[0], 3), rot6d], dim=-1)
        return motion135.detach().cpu().numpy().astype(np.float32)

    @torch.no_grad()
    def generate_smpl_sequences(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        seed: int = 0,
        sample_offset: int = 0,
        guidance_param: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[dict]:
        if len(captions) != len(lengths):
            raise ValueError("captions and lengths must have equal length")
        out = []
        for i, (caption, length) in enumerate(zip(captions, lengths)):
            self._seed_everything(int(seed) + int(sample_offset) + i)
            out.append(
                self._rollout_one(
                    str(caption),
                    int(length),
                    guidance_param=(
                        self.guidance_param
                        if guidance_param is None
                        else float(guidance_param)
                    ),
                    show_progress=show_progress,
                )
            )
        return out

    @torch.no_grad()
    def generate_motion135(
        self,
        captions: Sequence[str],
        lengths: Sequence[int],
        seed: int = 0,
        sample_offset: int = 0,
        guidance_param: Optional[float] = None,
        show_progress: bool = False,
    ) -> List[np.ndarray]:
        seqs = self.generate_smpl_sequences(
            captions,
            lengths,
            seed=seed,
            sample_offset=sample_offset,
            guidance_param=guidance_param,
            show_progress=show_progress,
        )
        return [self._motion_sequence_to_motion135(seq) for seq in seqs]

    def _rollout_one(
        self, caption: str, length: int, *, guidance_param: float, show_progress: bool
    ) -> dict:
        from tqdm import tqdm
        from motius.motion.conditioning.dart import encode_text
        from motius.motion.representation.dart import tensor_dict_to_device

        dataset = self._get_dataset(batch_size=1)
        # Original FlowMDM evaluation constructs its own SMPL-X utility even
        # for the HumanML3D model's SMPL-H seed dataset.
        if self.rollout_body_type == "dataset":
            primitive_utility = dataset.primitive_utility
        else:
            if self._rollout_utility is None:
                from motius.motion.representation.dart import PrimitiveUtility

                self._rollout_utility = PrimitiveUtility(
                    device=self.device, body_type=self.rollout_body_type
                )
            primitive_utility = self._rollout_utility
        future_length = int(dataset.future_length)
        history_length = int(dataset.history_length)
        primitive_length = history_length + future_length
        sample_fn = (
            self.diffusion.p_sample_loop
            if self.respacing == ""
            else self.diffusion.ddim_sample_loop
        )
        text_embedding = encode_text(
            dataset.clip_model, [caption], force_empty_zero=True
        ).to(dtype=torch.float32, device=self.device)
        batch = dataset.get_batch(batch_size=1)
        input_motions, model_kwargs = (
            batch[0]["motion_tensor_normalized"],
            {"y": batch[0]},
        )
        del model_kwargs["y"]["motion_tensor_normalized"]
        gender = model_kwargs["y"]["gender"][0]
        betas = model_kwargs["y"]["betas"][:, :primitive_length, :].to(self.device)
        pelvis_delta = primitive_utility.calc_calibrate_offset(
            {"betas": betas[:, 0, :], "gender": gender}
        )
        input_motions = input_motions.to(self.device)
        motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)
        history_motion = motion_tensor[:, :history_length, :]
        if self.initial_transform == "standard":
            from motius.motion.representation.dart import get_new_coordinate

            standard_joints = torch.tensor(
                _STANDARD_INITIAL_JOINTS, device=self.device, dtype=torch.float32
            ).reshape(1, 3, 3)
            transf_rotmat, transf_transl = get_new_coordinate(standard_joints)
        else:
            transf_rotmat = torch.eye(
                3, device=self.device, dtype=torch.float32
            ).reshape(1, 3, 3)
            transf_transl = torch.zeros(
                3, device=self.device, dtype=torch.float32
            ).reshape(1, 1, 3)
            if self.initial_transform == "canonical_seed":
                history_feature_dict = primitive_utility.tensor_to_dict(
                    dataset.denormalize(history_motion)
                )
                history_feature_dict.update(
                    {
                        "transf_rotmat": transf_rotmat,
                        "transf_transl": transf_transl,
                        "gender": gender,
                        "betas": betas[:, :history_length, :],
                        "pelvis_delta": pelvis_delta,
                    }
                )
                canonicalized_history, blended_feature_dict = (
                    primitive_utility.get_blended_feature(
                        history_feature_dict,
                        use_predicted_joints=self.use_predicted_joints,
                    )
                )
                transf_rotmat = canonicalized_history["transf_rotmat"]
                transf_transl = canonicalized_history["transf_transl"]
                history_motion = dataset.normalize(
                    primitive_utility.dict_to_tensor(blended_feature_dict)
                )
        motion_sequences = None
        nframes = max(1, int(length))
        num_primitives = int(np.ceil(nframes / future_length))
        iterator = tqdm(range(num_primitives), desc="DART", disable=not show_progress)
        for primitive_id in iterator:
            valid_length = min(future_length, nframes - primitive_id * future_length)
            scale = (
                torch.ones(
                    1, *self.denoiser_args.model_args.noise_shape, device=self.device
                )
                * guidance_param
            )
            y = {
                "text_embedding": text_embedding,
                "history_motion_normalized": history_motion,
                "scale": scale,
            }
            latent = sample_fn(
                self._guided_denoiser,
                (1, *self.denoiser_args.model_args.noise_shape),
                device=self.device,
                clip_denoised=False,
                model_kwargs={"y": y},
                skip_timesteps=0,
                init_image=None,
                progress=False,
                dump_steps=None,
                noise=torch.zeros_like(scale) if self.zero_noise else None,
                const_noise=False,
            )
            latent = latent.permute(1, 0, 2)
            future_motion_pred = self.bundle.decode_motion(
                latent,
                history_motion,
                nfuture=future_length,
                scale_latent=self.denoiser_args.rescale_latent,
            )
            future_frames = dataset.denormalize(future_motion_pred)
            all_frames = torch.cat(
                [dataset.denormalize(history_motion), future_frames], dim=1
            )
            valid_future_frames = future_frames[:, :valid_length, :]
            new_history_end = history_length + valid_length
            new_history_start = new_history_end - history_length
            new_history_frames = all_frames[:, new_history_start:new_history_end, :]
            future_feature_dict = primitive_utility.tensor_to_dict(valid_future_frames)
            future_feature_dict.update(
                {
                    "transf_rotmat": transf_rotmat,
                    "transf_transl": transf_transl,
                    "gender": gender,
                    "betas": betas[:, :valid_length, :],
                    "pelvis_delta": pelvis_delta,
                }
            )
            future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(
                future_feature_dict
            )
            future_primitive_dict = primitive_utility.transform_primitive_to_world(
                future_primitive_dict
            )
            if motion_sequences is None:
                motion_sequences = future_primitive_dict
            else:
                for key in ["transl", "global_orient", "body_pose", "betas", "joints"]:
                    motion_sequences[key] = torch.cat(
                        [motion_sequences[key], future_primitive_dict[key]], dim=1
                    )
            history_feature_dict = primitive_utility.tensor_to_dict(new_history_frames)
            history_feature_dict.update(
                {
                    "transf_rotmat": transf_rotmat,
                    "transf_transl": transf_transl,
                    "gender": gender,
                    "betas": betas[:, new_history_start:new_history_end, :],
                    "pelvis_delta": pelvis_delta,
                }
            )
            if self.fix_floor and primitive_id == num_primitives - 1:
                foot_height = (
                    history_feature_dict["joints"]
                    .reshape(-1, history_length, 22, 3)[:, 0, [10, 11], 2]
                    .amin(dim=-1)
                )
                foot_height_world = (
                    foot_height + history_feature_dict["transf_transl"][:, 0, 2]
                )
                history_feature_dict["transf_transl"][:, 0, 2] -= foot_height_world
            canonicalized_history, blended_feature_dict = (
                primitive_utility.get_blended_feature(
                    history_feature_dict, use_predicted_joints=self.use_predicted_joints
                )
            )
            transf_rotmat = canonicalized_history["transf_rotmat"]
            transf_transl = canonicalized_history["transf_transl"]
            history_motion = dataset.normalize(
                primitive_utility.dict_to_tensor(blended_feature_dict)
            )
        sequence = {
            "texts": [caption],
            "gender": motion_sequences["gender"],
            "betas": motion_sequences["betas"][0],
            "transl": motion_sequences["transl"][0],
            "global_orient": motion_sequences["global_orient"][0],
            "body_pose": motion_sequences["body_pose"][0],
            "joints": motion_sequences["joints"][0],
            "history_length": history_length,
            "future_length": future_length,
        }
        tensor_dict_to_device(sequence, "cpu")
        return sequence
