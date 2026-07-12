"""DART / DartControl ModelBundle.

This wraps the official DART motion-primitive denoiser and MVAE in the
motius model-zoo style. Runtime imports come from the vendored
``motius.models.dart.network`` tree and never from ``ref_repo``.
"""

from __future__ import annotations

import json
import os
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from motius.models.base_model_bundle import ModelBundle
from motius.registry import MODEL_BUNDLES

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NETWORK_DIR = Path(__file__).resolve().parent / "network"
_DEFAULT_ARTIFACT = _REPO_ROOT / "checkpoints" / "dart" / "motius_hml3d"
_DEFAULT_DENOISER = "mld_denoiser/smplh_hml3d_2_8_4/checkpoint_300000.pt"
_DEFAULT_SEED = "data/stand_20fps.pkl"
_STANDARD_INITIAL_JOINTS = [
    [0.0012, -0.3668, 0.9377],
    [0.0010, -0.4273, 0.8429],
    [-0.0034, -0.3135, 0.8290],
]
_MBENCH_COORD_CONVERSION = torch.tensor(
    [
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=torch.float32,
)


def _maybe_download_hub(name_or_path: str, local: Path) -> Path:
    if local.exists():
        return local
    try:
        from huggingface_hub import snapshot_download

        return Path(snapshot_download(repo_id=name_or_path, repo_type="model"))
    except Exception:
        return local


def _resolve_device(device: str | torch.device) -> torch.device:
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return dev


@contextmanager
def _dart_runtime(artifact_dir: Path):
    """Temporarily expose the vendored DART source tree and artifact cwd."""
    old_cwd = os.getcwd()
    network = str(_NETWORK_DIR)
    inserted = False
    if network not in sys.path:
        sys.path.insert(0, network)
        inserted = True
    os.chdir(str(artifact_dir))
    try:
        yield
    finally:
        os.chdir(old_cwd)
        if inserted:
            try:
                sys.path.remove(network)
            except ValueError:
                pass


@MODEL_BUNDLES.register_module()
class DARTBundle(ModelBundle):
    """DART text-conditioned primitive rollout bundle for HumanML3D SMPL-H."""

    motion_dim = 276

    def __init__(
        self,
        artifact_dir: Optional[str] = None,
        denoiser_checkpoint: str = _DEFAULT_DENOISER,
        seed_sequence: str = _DEFAULT_SEED,
        device: str = "cuda",
        guidance_param: float = 5.0,
        respacing: str = "",
        zero_noise: bool = False,
        use_predicted_joints: bool = False,
        fix_floor: bool = False,
        coord_conversion: str = "mbench",
        translation_source: str = "floor_aligned_smpl_transl",
        initial_transform: str = "standard",
        load_dataset: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.artifact_dir = Path(artifact_dir or _DEFAULT_ARTIFACT).resolve()
        self.denoiser_checkpoint = str(denoiser_checkpoint)
        self.seed_sequence = str(seed_sequence)
        self.guidance_param = float(guidance_param)
        self.respacing = str(respacing)
        self.zero_noise = bool(zero_noise)
        self.use_predicted_joints = bool(use_predicted_joints)
        self.fix_floor = bool(fix_floor)
        if coord_conversion not in {"mbench", "none"}:
            raise ValueError(f"coord_conversion must be 'mbench' or 'none', got {coord_conversion!r}")
        self.coord_conversion = coord_conversion
        allowed_translation_sources = {
            "floor_aligned_smpl_transl",
            "floor_aligned_joints_pelvis",
            "joints_pelvis",
            "smpl_transl",
        }
        if translation_source not in allowed_translation_sources:
            raise ValueError(
                "translation_source must be one of "
                "'floor_aligned_smpl_transl', 'floor_aligned_joints_pelvis', "
                "'joints_pelvis', or "
                f"'smpl_transl', got {translation_source!r}"
            )
        self.translation_source = translation_source
        initial_transform = "standard" if initial_transform == "official_flowmdm" else initial_transform
        if initial_transform not in {"standard", "canonical_seed", "identity"}:
            raise ValueError(
                "initial_transform must be one of "
                f"'standard', 'canonical_seed', or 'identity', got {initial_transform!r}"
            )
        self.initial_transform = initial_transform
        self._device = _resolve_device(device)
        self._dataset_cache: Dict[int, object] = {}

        ckpt = self.artifact_dir / self.denoiser_checkpoint
        if not ckpt.exists():
            raise FileNotFoundError(f"DART denoiser checkpoint not found: {ckpt}")
        if not (self.artifact_dir / self.seed_sequence).exists():
            raise FileNotFoundError(f"DART seed sequence not found: {self.artifact_dir / self.seed_sequence}")

        with _dart_runtime(self.artifact_dir):
            from mld.rollout_mld import load_mld
            from mld.train_mld import create_gaussian_diffusion

            (
                self.denoiser_args,
                self.denoiser_model,
                self.vae_args,
                self.vae_model,
            ) = load_mld(self.denoiser_checkpoint, self._device)
            diffusion_args = self.denoiser_args.diffusion_args
            diffusion_args.respacing = self.respacing
            self.diffusion = create_gaussian_diffusion(diffusion_args)

        if load_dataset:
            self._get_dataset(batch_size=1)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs):
        path = Path(pretrained_model_name_or_path)
        expected = path / _DEFAULT_DENOISER
        if not expected.exists():
            path = _maybe_download_hub(str(pretrained_model_name_or_path), path)
        if path.is_dir() and (path / _DEFAULT_DENOISER).exists():
            return cls(artifact_dir=str(path), **kwargs)
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def save_pretrained(self, save_directory: str, **kwargs):
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        index = {
            "model_type": "dart",
            "task": "text-to-motion",
            "motion_representation": "dart276",
            "export_adapters": ["smpl_sequence", "smplh_motion135"],
            "denoiser_checkpoint": self.denoiser_checkpoint,
            "seed_sequence": self.seed_sequence,
            "guidance_param": self.guidance_param,
            "respacing": self.respacing,
            "zero_noise": self.zero_noise,
            "use_predicted_joints": self.use_predicted_joints,
            "fix_floor": self.fix_floor,
            "coord_conversion": self.coord_conversion,
            "translation_source": self.translation_source,
            "initial_transform": self.initial_transform,
        }
        (save_dir / "model_index.json").write_text(json.dumps(index, indent=2) + "\n")

    @property
    def device(self) -> torch.device:
        return self._device

    def to_device(self, device: str | torch.device):
        self._device = _resolve_device(device)
        self.denoiser_model.to(self._device)
        self.vae_model.to(self._device)
        self._dataset_cache.clear()
        return self

    def _get_dataset(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size in self._dataset_cache:
            return self._dataset_cache[batch_size]
        with _dart_runtime(self.artifact_dir):
            from data_loaders.humanml.data.dataset import SinglePrimitiveDataset

            dataset = SinglePrimitiveDataset(
                cfg_path=self.vae_args.data_args.cfg_path,
                dataset_path=self.vae_args.data_args.data_dir,
                body_type=self.vae_args.data_args.body_type,
                sequence_path=self.seed_sequence,
                batch_size=batch_size,
                device=self._device,
                enforce_gender="male",
                enforce_zero_beta=1,
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
        if joints is not None and not isinstance(joints, torch.Tensor):
            joints = torch.as_tensor(joints)
        global_orient = global_orient.reshape(-1, 3, 3).float()
        body_pose = body_pose.reshape(-1, 21, 3, 3).float()
        transl = transl.reshape(-1, 3).float()
        if joints is not None:
            joints = joints.reshape(-1, 22, 3).float()
        if self.coord_conversion == "mbench":
            conv = _MBENCH_COORD_CONVERSION.to(device=global_orient.device, dtype=global_orient.dtype)
            transl = torch.einsum("ij,tj->ti", conv, transl)
            if joints is not None:
                joints = torch.einsum("ij,tkj->tki", conv, joints)
            global_orient = torch.einsum("ij,tjk->tik", conv, global_orient)
        if self.translation_source in {"joints_pelvis", "floor_aligned_joints_pelvis"}:
            if joints is None:
                raise ValueError(f"translation_source={self.translation_source!r} requires sequence['joints']")
            transl = joints[:, 0, :]
        if self.translation_source.startswith("floor_aligned_"):
            if joints is None:
                raise ValueError(f"translation_source={self.translation_source!r} requires sequence['joints']")
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
                    guidance_param=self.guidance_param if guidance_param is None else float(guidance_param),
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
        self,
        caption: str,
        length: int,
        *,
        guidance_param: float,
        show_progress: bool,
    ) -> dict:
        with _dart_runtime(self.artifact_dir):
            from tqdm import tqdm

            from utils.misc_util import encode_text
            from utils.smpl_utils import tensor_dict_to_device

        dataset = self._get_dataset(batch_size=1)
        primitive_utility = dataset.primitive_utility
        future_length = int(dataset.future_length)
        history_length = int(dataset.history_length)
        primitive_length = history_length + future_length
        sample_fn = self.diffusion.p_sample_loop if self.respacing == "" else self.diffusion.ddim_sample_loop

        text_embedding = encode_text(dataset.clip_model, [caption], force_empty_zero=True).to(
            dtype=torch.float32, device=self.device
        )
        batch = dataset.get_batch(batch_size=1)
        input_motions, model_kwargs = batch[0]["motion_tensor_normalized"], {"y": batch[0]}
        del model_kwargs["y"]["motion_tensor_normalized"]
        gender = model_kwargs["y"]["gender"][0]
        betas = model_kwargs["y"]["betas"][:, :primitive_length, :].to(self.device)
        pelvis_delta = primitive_utility.calc_calibrate_offset(
            {
                "betas": betas[:, 0, :],
                "gender": gender,
            }
        )
        input_motions = input_motions.to(self.device)
        motion_tensor = input_motions.squeeze(2).permute(0, 2, 1)
        history_motion = motion_tensor[:, :history_length, :]
        if self.initial_transform == "standard":
            with _dart_runtime(self.artifact_dir):
                from utils.smpl_utils import get_new_coordinate

            standard_joints = torch.tensor(
                _STANDARD_INITIAL_JOINTS,
                device=self.device,
                dtype=torch.float32,
            ).reshape(1, 3, 3)
            transf_rotmat, transf_transl = get_new_coordinate(standard_joints)
        else:
            transf_rotmat = torch.eye(3, device=self.device, dtype=torch.float32).reshape(1, 3, 3)
            transf_transl = torch.zeros(3, device=self.device, dtype=torch.float32).reshape(1, 1, 3)
            if self.initial_transform == "canonical_seed":
                history_feature_dict = primitive_utility.tensor_to_dict(dataset.denormalize(history_motion))
                history_feature_dict.update(
                    {
                        "transf_rotmat": transf_rotmat,
                        "transf_transl": transf_transl,
                        "gender": gender,
                        "betas": betas[:, :history_length, :],
                        "pelvis_delta": pelvis_delta,
                    }
                )
                canonicalized_history, blended_feature_dict = primitive_utility.get_blended_feature(
                    history_feature_dict,
                    use_predicted_joints=self.use_predicted_joints,
                )
                transf_rotmat = canonicalized_history["transf_rotmat"]
                transf_transl = canonicalized_history["transf_transl"]
                history_motion = dataset.normalize(primitive_utility.dict_to_tensor(blended_feature_dict))
        motion_sequences = None

        nframes = max(1, int(length))
        num_primitives = int(np.ceil(nframes / future_length))
        iterator = tqdm(range(num_primitives), desc="DART", disable=not show_progress)
        for primitive_id in iterator:
            valid_length = min(future_length, nframes - primitive_id * future_length)
            scale = torch.ones(1, *self.denoiser_args.model_args.noise_shape, device=self.device) * guidance_param
            y = {
                "text_embedding": text_embedding,
                "history_motion_normalized": history_motion,
                "scale": scale,
            }
            latent = sample_fn(
                self.denoiser_model,
                (1, *self.denoiser_args.model_args.noise_shape),
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
            future_motion_pred = self.vae_model.decode(
                latent,
                history_motion,
                nfuture=future_length,
                scale_latent=self.denoiser_args.rescale_latent,
            )
            future_frames = dataset.denormalize(future_motion_pred)
            all_frames = torch.cat([dataset.denormalize(history_motion), future_frames], dim=1)
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
            future_primitive_dict = primitive_utility.feature_dict_to_smpl_dict(future_feature_dict)
            future_primitive_dict = primitive_utility.transform_primitive_to_world(future_primitive_dict)
            if motion_sequences is None:
                motion_sequences = future_primitive_dict
            else:
                for key in ["transl", "global_orient", "body_pose", "betas", "joints"]:
                    motion_sequences[key] = torch.cat([motion_sequences[key], future_primitive_dict[key]], dim=1)

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
                foot_height_world = foot_height + history_feature_dict["transf_transl"][:, 0, 2]
                history_feature_dict["transf_transl"][:, 0, 2] -= foot_height_world
            canonicalized_history, blended_feature_dict = primitive_utility.get_blended_feature(
                history_feature_dict,
                use_predicted_joints=self.use_predicted_joints,
            )
            transf_rotmat = canonicalized_history["transf_rotmat"]
            transf_transl = canonicalized_history["transf_transl"]
            history_motion = dataset.normalize(primitive_utility.dict_to_tensor(blended_feature_dict))

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

    def forward(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError("Use DARTPipeline.infer_t2m_* for inference.")
