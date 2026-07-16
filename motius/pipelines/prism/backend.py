from diffusers import DiffusionPipeline
from einops import rearrange
import numpy as np
import torch
from transformers import PreTrainedTokenizer, UMT5EncoderModel
from motius.models.prism.autoencoder_kl_2d import AutoencoderKLPrism2DTK
from motius.models.prism.gaussian_distribution import (
    DiagonalGaussianDistributionNd,
)
from motius.models.prism.processor import PRISMMotionProcessor
from motius.models.prism.network import PrismTransformerMotionModel
from diffusers.schedulers import (
    FlowMatchEulerDiscreteScheduler,
)
from typing import Any, Dict, List, Optional, Tuple, Union
from mmengine import print_log
from diffusers.utils.torch_utils import randn_tensor
from motius.motion.representation.rotation import (
    matrix_to_quaternion,
    quaternion_to_axis_angle,
    rotation_6d_to_matrix,
)


def rotation_6d_to_axis_angle(value):
    matrix = rotation_6d_to_matrix(value, convention="column")
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


class PrismARPipeline(DiffusionPipeline):
    """Autoregressive Text-to-Motion Pipeline.

    This pipeline generates long motion sequences by autoregressively generating
    multiple segments. Each segment uses the last frame of the previous segment
    as the first frame condition for the next segment.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLPrism2DTK,
        scheduler: FlowMatchEulerDiscreteScheduler,
        smpl_processor: PRISMMotionProcessor,
        transformer: PrismTransformerMotionModel,
        expand_timesteps: bool = True,
        is_causal: bool = False,
        dtype: Optional[torch.dtype] = None,
    ):
        device = next(transformer.parameters()).device
        super().__init__()

        def _module_dtype(module, fallback: torch.dtype = torch.float32) -> torch.dtype:
            if module is None:
                return fallback
            try:
                return next(module.parameters()).dtype
            except StopIteration:
                return fallback

        if dtype is None:
            vae_dtype = _module_dtype(vae)
            text_dtype = _module_dtype(text_encoder, vae_dtype)
            transformer_dtype = _module_dtype(transformer, vae_dtype)
            processor_dtype = _module_dtype(smpl_processor, torch.float32)
        else:
            vae_dtype = text_dtype = transformer_dtype = processor_dtype = dtype

        self.register_modules(
            vae=vae.to(device, vae_dtype),
            text_encoder=text_encoder.to(device, text_dtype),
            tokenizer=tokenizer,
            transformer=transformer.to(device, transformer_dtype),
            scheduler=scheduler,
        )

        self.register_to_config(
            expand_timesteps=expand_timesteps,
            is_causal=is_causal,
        )

        # ========== SPECTRAL KT-RoPE CONFIG VERIFICATION ==========
        # Verify that spectral RoPE parameters are preserved in transformer config
        if hasattr(transformer, 'config'):
            joint_pos_mode = getattr(transformer.config, 'joint_pos_mode', 'sequential')
            print_log(f"[PRISM Pipeline] Loaded transformer with joint_pos_mode='{joint_pos_mode}'")

            if joint_pos_mode == "spectral":
                num_spectral_modes = getattr(transformer.config, 'num_spectral_modes', 4)
                spectral_scale = getattr(transformer.config, 'spectral_scale', None)
                print_log(f"  ├─ num_spectral_modes={num_spectral_modes}")
                print_log(f"  └─ spectral_scale={spectral_scale}")
                print_log(f"[PRISM Pipeline] Spectral KT-RoPE mode ACTIVE - inference will use kinematic tree embeddings")
            elif joint_pos_mode == "dfs":
                print_log(f"[PRISM Pipeline] DFS mode ACTIVE - inference will use depth-first-search joint ordering")
            else:
                print_log(f"[PRISM Pipeline] Sequential mode (default) - inference will use standard sequential indices")
        else:
            print_log("[PRISM Pipeline] Warning: transformer has no config - cannot verify RoPE mode")
        # ============================================================

        self.smpl_processor: PRISMMotionProcessor = smpl_processor.to(device, processor_dtype)

        self.latents_mean = torch.tensor(
            vae.config.latents_mean, dtype=vae_dtype, device=device
        ).view(1, self.vae.config.z_dim, 1, 1)

        self.latents_std = torch.tensor(
            vae.config.latents_std, dtype=vae_dtype, device=device
        ).view(1, self.vae.config.z_dim, 1, 1)

        self.vae_scale_factor_temporal = vae.config.scale_factor_temporal

        # KAFS-Inference: Per-joint adaptive timestep scaling
        # Shape: [num_joints] with values in range [0.85, 1.15] based on kinematic depth
        self._kafs_alpha_map = None
        self._kafs_mode = "none"  # Tracks which KAFS mode is active

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        num_frames: int = 81,
        num_joints: int = 23,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        first_frame_latents: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare latents for denoising with optional prefix-frame conditioning.

        Args:
            batch_size: Number of samples in the batch.
            num_channels_latents: Number of latent channels.
            num_frames: Number of motion frames.
            num_joints: Number of joints.
            dtype: Data type for tensors.
            device: Device to place tensors on.
            first_frame_latents: Optional encoded condition latents [B, C, T_cond, J].

        Returns:
            latents: Random noise tensor [B, C, T_latent, J].
            condition: Condition tensor with prefix frames encoded [B, C, T_latent, J].
            first_frame_mask: Mask indicating which positions to denoise [B, C, T_latent, J].
                0 for condition positions, 1 for positions to denoise.
        """
        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            num_joints,
        )

        latents = randn_tensor(shape, generator=None, device=device, dtype=dtype)

        # Create condition tensor and mask
        condition = torch.zeros_like(latents)
        first_frame_mask = torch.ones_like(latents)

        if first_frame_latents is not None:
            # first_frame_latents: [B, C, T_cond, J] or [1, C, T_cond, J]
            # Expand batch dimension if needed
            if first_frame_latents.shape[0] == 1 and batch_size > 1:
                first_frame_latents = first_frame_latents.expand(batch_size, -1, -1, -1)
            cond_latent_frames = min(first_frame_latents.shape[2], num_latent_frames)
            condition[:, :, :cond_latent_frames, :] = first_frame_latents[:, :, :cond_latent_frames, :]
            first_frame_mask[:, :, :cond_latent_frames, :] = 0.0

        return latents, condition, first_frame_mask


    def set_kafs_alpha(self, mode: str = "none", alpha_vals: Optional[torch.Tensor] = None, device: Optional[torch.device] = None) -> None:
        """Set KAFS (Kinematic-Adaptive Flow Scheduling) per-joint timestep scaling.

        Args:
            mode: KAFS mode to use. Options:
                - "none": No KAFS scaling (standard baseline)
                - "depth_driven": Per-joint scaling based on kinematic tree depth
                - "uniform": Uniform scaling with all alphas set to a constant
                - "random": Random alphas for ablation control
                - "custom": Use provided alpha_vals tensor
            alpha_vals: Custom alpha values tensor [num_joints] if mode=="custom".
                        If None and mode=="depth_driven", uses hardcoded kinematic-based values.
            device: Device to place alpha_map tensor on. Defaults to VAE device.

        Notes:
            - Alpha values typically range from 0.85 (proximal/root) to 1.15 (distal/wrist)
            - Defines a per-joint monotone time-warp of the shared sigma grid:
              sigma_j(k) = sigma(k) ** (1/alpha_j). alpha<1 => faster descent (root),
              alpha>1 => slower (distal). Endpoints {0,1} are preserved, so every
              joint still denoises fully; only the rate differs. Integrated as a
              consistent per-token Euler step (label == true sigma, per-token dt),
              which keeps KAFS within Diffusion Forcing's valid per-token sampling
              family rather than biasing the velocity field (cf. naive t*alpha).
            - Only applicable when self.config.expand_timesteps is True
        """
        if device is None:
            device = self.vae.device

        if mode == "none":
            self._kafs_alpha_map = None
            self._kafs_mode = "none"
            print_log("KAFS: Disabled (standard baseline)")

        elif mode == "depth_driven":
            # Canonical KAFS: gamma_j = 1/alpha_j warps the shared sigma grid as
            # sigma_j(k) = sigma(k) ** gamma_j. Root/pelvis and spine stay on the
            # baseline schedule (alpha=1.0, gamma=1) so the low-frequency global
            # trajectory is integrated exactly as the baseline; distal joints get
            # alpha<1 (gamma>1) to add low-noise refinement for their high-frequency
            # dynamics. Validated on the overfit model: reconstruction stays on par
            # with the baseline (MPJPE -0.7%). Structure (by kinematic depth):
            # [trans, pelvis, L_hip, R_hip, spine1, L_knee, R_knee, spine2,
            #  L_ankle, R_ankle, spine3, L_foot, R_foot, neck,
            #  L_collar, R_collar, head, L_shoulder, R_shoulder, L_elbow, R_elbow, L_wrist, R_wrist]
            alpha_vals = torch.tensor([
                1.000,         # Translation (global trajectory) -> baseline schedule
                1.000,         # Pelvis (root) -> baseline schedule
                0.975, 0.975,  # L_Hip, R_Hip
                0.975,         # Spine1
                0.950, 0.950,  # L_Knee, R_Knee
                0.950,         # Spine2
                0.925, 0.925,  # L_Ankle, R_Ankle
                0.925,         # Spine3
                0.900, 0.900,  # L_Foot, R_Foot
                0.950,         # Neck
                0.925, 0.925,  # L_Collar, R_Collar
                0.925,         # Head
                0.900, 0.900,  # L_Shoulder, R_Shoulder
                0.875, 0.875,  # L_Elbow, R_Elbow
                0.850, 0.850,  # L_Wrist, R_Wrist (deepest -> most low-noise refinement)
            ], dtype=self.vae.dtype, device=device)

            self._kafs_alpha_map = alpha_vals.view(1, 1, 1, -1)  # [1, 1, 1, 23]
            self._kafs_mode = "depth_driven"
            print_log(f"KAFS: Depth-driven mode enabled. Alpha range: [{alpha_vals.min():.2f}, {alpha_vals.max():.2f}]")

        elif mode == "uniform":
            # All joints get the same alpha (should give similar results to baseline)
            uniform_alpha = 1.0
            alpha_vals = torch.full((23,), uniform_alpha, dtype=self.vae.dtype, device=device)
            self._kafs_alpha_map = alpha_vals.view(1, 1, 1, -1)
            self._kafs_mode = "uniform"
            print_log(f"KAFS: Uniform mode enabled. All alphas = {uniform_alpha}")

        elif mode == "random":
            # Random alphas in [0.85, 1.15] for ablation control
            torch.manual_seed(42)  # Reproducible randomness
            alpha_vals = torch.rand(23, dtype=self.vae.dtype, device=device) * 0.30 + 0.85
            self._kafs_alpha_map = alpha_vals.view(1, 1, 1, -1)
            self._kafs_mode = "random"
            print_log(f"KAFS: Random mode enabled. Alpha range: [{alpha_vals.min():.2f}, {alpha_vals.max():.2f}]")

        elif mode == "custom":
            if alpha_vals is None:
                raise ValueError("alpha_vals must be provided when mode='custom'")
            if isinstance(alpha_vals, list):
                alpha_vals = torch.tensor(alpha_vals, dtype=self.vae.dtype, device=device)
            elif not isinstance(alpha_vals, torch.Tensor):
                raise TypeError("alpha_vals must be a torch.Tensor or list")

            alpha_vals = alpha_vals.to(dtype=self.vae.dtype, device=device)
            if alpha_vals.shape[-1] != 23:
                raise ValueError(f"alpha_vals must have shape [..., 23] for 23 joints, got {alpha_vals.shape}")

            self._kafs_alpha_map = alpha_vals.view(1, 1, 1, -1)
            self._kafs_mode = "custom"
            print_log(f"KAFS: Custom mode enabled. Alpha range: [{alpha_vals.min():.2f}, {alpha_vals.max():.2f}]")

        else:
            raise ValueError(f"Unknown KAFS mode: {mode}. Choose from: none, depth_driven, uniform, random, custom")


    @staticmethod
    def _load_condition_smplx_dict(motion_path: str) -> Dict[str, np.ndarray]:
        """Load a prefix-condition motion from SMPL-X npz or official 272 npy."""
        if str(motion_path).endswith(".npy"):
            motion_272 = np.load(motion_path)
            if motion_272.ndim != 2 or motion_272.shape[-1] != 272:
                raise ValueError(
                    f"Expected official MotionStreamer-272 array for {motion_path}, "
                    f"got shape={motion_272.shape}"
                )
            from motius.motion.representation.motion272 import motion272_to_motion135

            motion135 = motion272_to_motion135(motion_272)
            rot6d = motion135[:, 3:].reshape(-1, 22, 6)
            rot = rotation_6d_to_matrix(rot6d, convention="row")
            axis_angle = quaternion_to_axis_angle(matrix_to_quaternion(rot))
            root = motion135[:, :3]
            return {
                "transl": root.astype(np.float32),
                "global_orient": axis_angle[:, 0].astype(np.float32),
                "body_pose": axis_angle[:, 1:].reshape(rot.shape[0], -1).astype(np.float32),
                "mocap_framerate": np.asarray(30.0, dtype=np.float32),
            }
        return dict(np.load(motion_path))

    def load_condition_pose(self, motion_path: str, condition_num_frames: int = 1) -> torch.Tensor:
        """Load and process a condition pose from SMPL-X npz or official 272 npy.

        Args:
            motion_path: Path to the motion file containing condition data.
            condition_num_frames: Number of observed prefix frames to use.

        Returns:
            Processed motion tensor of shape [1, T_cond, J, C] ready for VAE encoding.
            Where C=6 (6D rotation representation), J=num_joints.
            VAE expects [B, T, K, C] format.
        """
        condition_num_frames = max(1, int(condition_num_frames))
        device = self.vae.device
        dtype = self.vae.dtype

        smplx_dict = self._load_condition_smplx_dict(motion_path)
        # [T, D] where D = J * 6
        motion = (
            self.smpl_processor.smplx_dict_to_motion_vector(smplx_dict)
            .unsqueeze(0)
            .to(device=device, dtype=dtype)
        )

        # Normalize the raw motion vector to the VAE's training distribution.
        # PrismBundle.encode_motion (training) applies smpl_pose_processor.normalize
        # before VAE.encode, so the VAE operates in *normalized* motion space.
        # The pipeline's encode_motion does NOT normalize (it is also reused for the
        # already-normalized last-frame in autoregressive chaining), so the raw GT
        # prefix must be normalized HERE to stay train/inference consistent. Skipping
        # this makes the condition latents out-of-distribution, so the prefix/first
        # frame condition does not take effect (jitter + condition frames diverge
        # from the input).
        motion = self.smpl_processor.normalize(motion)

        # [B, T, D] -> [B, T, J, 6]
        motion = rearrange(motion, "b t (j d) -> b t j d", d=6)

        if motion.shape[1] != condition_num_frames:
            used_frames = min(condition_num_frames, motion.shape[1])
            print_log(
                f"Condition pose: original motion has {motion.shape[1]} frames, "
                f"use the first {used_frames} frame(s)"
            )
            motion = motion[:, :used_frames]  # [B, T_cond, J, 6]

        # Return in VAE expected format: [B, T, J, C]
        return motion.to(device=device, dtype=dtype)

    def extract_last_frame_motion(self, motion_vec: torch.Tensor,
                                  num_frames: int = 1) -> torch.Tensor:
        """Extract the trailing frames from decoded motion for autoregressive
        conditioning.

        Conditioning the next segment on a single frame only pins position, so the
        model is free to pick an arbitrary starting velocity at the junction (a
        visible jerk between segments). Carrying several trailing frames conveys
        velocity/trajectory and matches the multi-frame prefix the model was
        trained with (TP2M cond 5/9), yielding a continuous transition.

        Args:
            motion_vec: Decoded motion tensor of shape [B, T, J, C] from VAE.
            num_frames: Number of trailing frames to carry over.

        Returns:
            Trailing motion tensor of shape [B, num_frames, J, C] ready for VAE
            encoding.
        """
        num_frames = max(1, min(int(num_frames), motion_vec.shape[1]))
        return motion_vec[:, -num_frames:, :, :]  # [B, k, J, C]

    @torch.no_grad()
    def encode_motion(
        self,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        """Encode motion to VAE latent space.

        Args:
            motion: Motion tensor of shape [B, T, J, C] where C=6 (6D rotation).
                This is the format expected by VAE.encode().

        Returns:
            Latent tensor of shape [B, Z_dim, T_latent, J].
        """
        # Encode by SMPL VAE: [B, T, J, C] -> [B, Z_dim*2, T_latent, J]
        # VAE internally permutes to [B, C, T, J] before encoding
        # VAE must run in fp32 — override any global AMP autocast.
        device_type = motion.device.type
        with torch.autocast(device_type, enabled=False):
            z = self.vae.encode(motion.float())

        # Sample from the latent distribution (use mode for deterministic encoding)
        lat = DiagonalGaussianDistributionNd(z)
        z = lat.mode()

        # Normalize latents
        z = (z - self.latents_mean) / self.latents_std

        return z  # [B, Z_dim, T_latent, J]

    @torch.no_grad()
    def generate_single_segment(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        first_frame_motion: Optional[torch.Tensor] = None,
        num_frames: int = 129,
        valid_num_frames: Optional[int] = None,
        num_joints: int = 23,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        max_sequence_length: int = 256,
        attention_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Generate a single motion segment.

        Args:
            prompt: Text prompt for this segment.
            negative_prompt: Negative prompt for classifier-free guidance.
            first_frame_motion: First frame condition tensor [B, 1, J, C] or None.
            num_frames: Number of frames to generate.
            valid_num_frames: Optional pre-pad valid motion length. When set,
                self-attention masks out latent tokens beyond this length while
                still denoising the full generated latent canvas.
            num_joints: Number of joints.
            num_inference_steps: Number of denoising steps.
            guidance_scale: Classifier-free guidance scale.
            max_sequence_length: Maximum sequence length for text encoding.
            attention_kwargs: Additional kwargs for attention.

        Returns:
            Decoded motion tensor of shape [B, T, J, C].
        """
        device = next(self.transformer.parameters()).device
        do_cfg = guidance_scale > 1.0
        batch_size = 1

        # Encode first frame if provided
        first_frame_latents = None
        if first_frame_motion is not None:
            first_frame_latents = self.encode_motion(first_frame_motion)

        # Encode prompt with attention masks
        prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask = self.encode_prompt_with_mask(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_cfg,
            num_motion_per_prompt=1,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # Move masks to correct dtype for transformer
        if prompt_mask is not None:
            prompt_mask = prompt_mask.to(transformer_dtype)
        if negative_prompt_mask is not None:
            negative_prompt_mask = negative_prompt_mask.to(transformer_dtype)

        # No text cross-attention mask: matches the official Wan implementation
        # (text padded with zeros, context_lens=None). The motion transformer
        # attends over the full zero-padded text just like Wan, keeping train and
        # inference consistent. Passing a mask shifts the cross-attention softmax
        # normalization and corrupts conditioning (length-dependent drift).
        prompt_mask = None
        negative_prompt_mask = None

        # Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # KAFS: precompute per-joint warp exponents gamma_j = 1/alpha_j and the
        # base sigma grid. We then run a consistent per-token Euler step so that
        # each joint follows its own monotone schedule sigma_j(k)=sigma(k)**gamma_j
        # (label == true noise level, integrated with its own dt). See set_kafs_alpha.
        kafs_gamma = None
        if self._kafs_alpha_map is not None:
            kafs_gamma = (1.0 / self._kafs_alpha_map).to(device=device)
        kafs_sigmas = self.scheduler.sigmas.to(device=device, dtype=torch.float32)
        kafs_num_train_ts = float(self.scheduler.config.num_train_timesteps)

        # Prepare latents
        num_channels_latents = self.transformer.config.in_channels
        latents, condition, first_frame_mask = self.prepare_latents(
            batch_size=batch_size,
            num_channels_latents=num_channels_latents,
            num_joints=num_joints,
            num_frames=num_frames,
            dtype=transformer_dtype,
            device=device,
            first_frame_latents=first_frame_latents,
        )

        # Create motion padding mask (for attention masking of padded positions).
        # Standard inference treats all generated tokens as valid. Benchmark runs
        # can instead generate a training-length canvas (e.g. 360 frames) while
        # masking latent tokens beyond the official sample length.
        if valid_num_frames is None:
            motion_mask = torch.ones(
                batch_size, latents.shape[2], latents.shape[3], device=latents.device
            )
        else:
            valid_num_frames_t = torch.tensor(
                [int(valid_num_frames)], device=latents.device, dtype=torch.long
            )
            scale = int(self.vae_scale_factor_temporal)
            valid_latent_frames = (valid_num_frames_t + scale - 1) // scale
            valid_latent_frames = torch.clamp(
                valid_latent_frames, min=0, max=latents.shape[2]
            )
            frame_idx = torch.arange(latents.shape[2], device=latents.device).unsqueeze(0)
            motion_mask = frame_idx < valid_latent_frames.unsqueeze(1)
            motion_mask = motion_mask.unsqueeze(-1).expand(
                batch_size, latents.shape[2], latents.shape[3]
            ).float()

        # Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        for i, t in enumerate(timesteps):
            current_model = self.transformer
            current_guidance_scale = guidance_scale

            if self.config.expand_timesteps:
                latent_model_input = (
                    (1 - first_frame_mask) * condition + first_frame_mask * latents
                ).to(transformer_dtype)
                if kafs_gamma is not None:
                    # Per-joint label = true per-joint noise level sigma_j(k)*T.
                    sig_jcur = torch.pow(kafs_sigmas[i], kafs_gamma)  # [1,1,1,J]
                    temp_ts = (
                        first_frame_mask[0][0] * (sig_jcur[0, 0] * kafs_num_train_ts)
                    ).flatten()
                else:
                    temp_ts = (first_frame_mask[0][0] * t).flatten()
                timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
            else:
                latent_model_input = latents.to(transformer_dtype)
                timestep = t.expand(latents.shape[0])

            noise_pred = current_model(
                hidden_states=latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                encoder_hidden_states_mask=prompt_mask,
                attention_kwargs=attention_kwargs,
                is_causal=self.config.is_causal,
                hidden_states_mask=motion_mask,
            )

            if do_cfg:
                noise_uncond = current_model(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=negative_prompt_embeds,
                    encoder_hidden_states_mask=negative_prompt_mask,
                    attention_kwargs=attention_kwargs,
                    is_causal=self.config.is_causal,
                    hidden_states_mask=motion_mask,
                )
                noise_pred = noise_uncond + current_guidance_scale * (
                    noise_pred - noise_uncond
                )

            if kafs_gamma is not None:
                # Consistent per-token Euler: each joint advances by its own dt_j
                # along the warped schedule, matching the per-joint label above.
                sig_jcur = torch.pow(kafs_sigmas[i], kafs_gamma)  # [1,1,1,J]
                sig_jnext = torch.pow(kafs_sigmas[i + 1], kafs_gamma)  # [1,1,1,J]
                dt = (sig_jnext - sig_jcur).to(torch.float32)
                latents = (latents.float() + dt * noise_pred.float()).to(latents.dtype)
            else:
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            # Force-restore condition frame latents after each step
            # so they remain noise-free throughout the entire denoising process.
            if first_frame_latents is not None:
                latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        # Merge condition back for final output (redundant but safe)
        if self.config.expand_timesteps and first_frame_latents is not None:
            latents = (1 - first_frame_mask) * condition + first_frame_mask * latents

        # Decode latents to motion (rot6d space)
        motion_vec = self.decode_motion(latents)

        return motion_vec
    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]],
        negative_prompt: Optional[str] = None,
        first_frame_motion_path: Optional[str] = None,
        condition_num_frames: int = 1,
        num_frames_per_segment: Union[int, List[int]] = 129,
        num_joints: int = 23,
        num_inference_steps: int = 50,
        guidance_scale: float = 2.0,
        use_static: bool = False,
        use_smooth: bool = False,
        normalize: bool = True,
        use_blend: bool = True,
        mocap_framerate: float = 30.0,
        gender: str = "neutral",
        max_sequence_length: int = 256,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        overlap_frames: int = 1,
        ar_condition_frames: int = 5,
        use_rollout_trans: Union[bool, str] = "xz_rollout_y_absolute",
        preserve_segment_lengths: bool = False,
        generation_num_frames_per_segment: Optional[Union[int, List[int]]] = None,
        valid_num_frames_per_segment: Optional[Union[int, List[int]]] = None,
        allow_segment_padding: bool = True,
        align_generation_frames: bool = True,
        return_motion_vec: bool = False,
    ) -> Dict:
        """Generate long motion autoregressively from multiple prompts.

        Args:
            prompts: List of text prompts, one for each segment.
                If a single string is provided, it will be wrapped in a list.
            negative_prompt: Negative prompt for classifier-free guidance.
            first_frame_motion_path: Optional path to npz file for first segment's
                prefix-frame condition.
            condition_num_frames: Number of observed frames to condition on for the
                first segment. Values larger than 1 evaluate TP2M-style prefix conditioning.
            num_frames_per_segment: Number of frames per segment (int for all segments, or list of int per segment).
            num_joints: Number of joints in the output motion.
            num_inference_steps: Number of denoising steps per segment.
            guidance_scale: Classifier-free guidance scale.
            use_static: Whether to use static joint refinement.
            use_smooth: Whether to apply smoothing to output motion.
            normalize: Whether to normalize facing direction and ground plane.
            use_blend: Whether to apply segment boundary blending to smooth transitions.
            mocap_framerate: Frame rate of the output motion.
            gender: Gender for SMPL model ('neutral', 'male', 'female').
            max_sequence_length: Maximum sequence length for text encoding.
            attention_kwargs: Additional kwargs for attention.
            overlap_frames: Number of overlapping frames between segments (default 1).
                The last frame of previous segment becomes the first frame of next segment.
            preserve_segment_lengths: Generate enough VAE-aligned frames for each
                segment to survive autoregressive prefix removal, then trim each
                segment contribution back to its requested length.  Use this for
                fixed-boundary evaluation protocols.
            generation_num_frames_per_segment: Optional generation canvas length
                per segment. Defaults to ``num_frames_per_segment``. Benchmark
                runs use this to generate a 360-frame canvas and crop back to
                official lengths.
            valid_num_frames_per_segment: Optional valid length per segment for
                motion self-attention masking. Defaults to the generation length.
            allow_segment_padding: Whether ``preserve_segment_lengths`` may
                repeat the last generated frame when a segment decodes shorter
                than requested. Benchmark protocols should set this false so
                length bugs fail visibly.
            align_generation_frames: Whether to snap generation lengths to the
                historical PRISM VAE alignment heuristic before preparing
                latents. Benchmark protocols set this false to pass exact
                official/padded frame counts into ``prepare_latents``.
            return_motion_vec: When true, return both the SMPL-X dictionary and
                the decoded normalized motion tensor before translation
                post-processing. This is used by decode-only ablations so the
                same sampled motion can be converted with multiple translation
                policies.

        Returns:
            smplx_dict: Dictionary containing SMPL-X parameters for the full motion.
        """
        # Convert single prompt to list
        if isinstance(prompts, str):
            prompts = [prompts]

        num_segments = len(prompts)
        print_log(f"Generating {num_segments} motion segments autoregressively...")

        # Per-segment frame counts.
        scale = self.vae_scale_factor_temporal

        def _round_frames(n: int) -> int:
            if (n - 1) % scale != 0:
                return (n // scale) * scale + 1
            return max(1, n)

        def _ceil_frames(n: int) -> int:
            n = max(1, int(n))
            if (n - 1) % scale == 0:
                return n
            return ((n - 1 + scale - 1) // scale) * scale + 1

        def _as_segment_list(v: Union[int, List[int]]) -> List[int]:
            if isinstance(v, list):
                if len(v) != num_segments:
                    print_log(
                        f"num_frames_per_segment list length {len(v)} != num_segments {num_segments}; using first value for all."
                    )
                    first = int(v[0] if v else 129)
                    return [first] * num_segments
                return [int(n) for n in v]
            return [int(v)] * num_segments

        requested_segment_lengths = _as_segment_list(num_frames_per_segment)
        generation_requested_lengths = _as_segment_list(
            generation_num_frames_per_segment
            if generation_num_frames_per_segment is not None
            else num_frames_per_segment
        )
        if valid_num_frames_per_segment is None:
            valid_requested_lengths = generation_requested_lengths
        else:
            valid_requested_lengths = _as_segment_list(valid_num_frames_per_segment)

        # Number of trailing frames carried across the autoregressive boundary.
        # Snap to scale*n+1 so the VAE encode/decode of the carried clip is exact
        # (the conditioned region in the next segment is then exactly k frames).
        k_carry = max(1, int(ar_condition_frames))
        k_carry = ((k_carry - 1) // scale) * scale + 1

        if preserve_segment_lengths:
            num_frames_per_segment_list = [
                int(n) + (k_carry if i > 0 else 0)
                for i, n in enumerate(generation_requested_lengths)
            ]
        else:
            num_frames_per_segment_list = [int(n) for n in generation_requested_lengths]
        if align_generation_frames:
            if preserve_segment_lengths:
                num_frames_per_segment_list = [
                    _ceil_frames(n) for n in num_frames_per_segment_list
                ]
            else:
                num_frames_per_segment_list = [
                    _round_frames(n) for n in num_frames_per_segment_list
                ]

        valid_num_frames_list = [
            int(n) + (k_carry if preserve_segment_lengths and i > 0 else 0)
            for i, n in enumerate(valid_requested_lengths)
        ]

        # Load first frame condition if provided
        first_frame_motion = None
        if first_frame_motion_path is not None:
            first_frame_motion = self.load_condition_pose(
                first_frame_motion_path,
                condition_num_frames=condition_num_frames,
            )

        # Store all motion segments
        all_motion_segments = []
        raw_decoded_segment_lengths = []
        pretrim_segment_lengths = []

        # Generate each segment
        with self.progress_bar(total=num_segments) as progress_bar:
            for seg_idx, prompt in enumerate(prompts):
                print_log(
                    f"Generating segment {seg_idx + 1}/{num_segments}: {prompt[:50]}..."
                )

                # Generate single segment
                num_frames_this = num_frames_per_segment_list[seg_idx]
                valid_num_frames_this = valid_num_frames_list[seg_idx]
                motion_vec = self.generate_single_segment(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    first_frame_motion=first_frame_motion,
                    num_frames=num_frames_this,
                    valid_num_frames=valid_num_frames_this,
                    num_joints=num_joints,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    max_sequence_length=max_sequence_length,
                    attention_kwargs=attention_kwargs,
                )
                raw_decoded_segment_lengths.append(int(motion_vec.shape[1]))

                # Store segment. For seg>0 the first k_carry frames reproduce the
                # carried tail of the previous segment (the prefix condition), so
                # drop them to avoid duplication.
                if seg_idx == 0:
                    contrib = motion_vec
                else:
                    skip = min(k_carry, motion_vec.shape[1] - 1)
                    contrib = motion_vec[:, skip:]
                pretrim_segment_lengths.append(int(contrib.shape[1]))

                if preserve_segment_lengths:
                    target_len = int(requested_segment_lengths[seg_idx])
                    if contrib.shape[1] >= target_len:
                        contrib = contrib[:, :target_len]
                    else:
                        pad_n = target_len - int(contrib.shape[1])
                        if not allow_segment_padding:
                            raise ValueError(
                                f"Decoded segment shorter than target: "
                                f"segment={seg_idx}, decoded={contrib.shape[1]}, "
                                f"target={target_len}, generation_frames={num_frames_this}"
                            )
                        pad = contrib[:, -1:].repeat(1, pad_n, 1, 1)
                        contrib = torch.cat([contrib, pad], dim=1)
                all_motion_segments.append(contrib)

                # Carry the trailing k_carry frames from the actually retained
                # contribution.  When preserve_segment_lengths=True we may have
                # generated a few extra VAE-aligned frames; conditioning on those
                # discarded frames would silently shift the next boundary.
                first_frame_motion = self.extract_last_frame_motion(contrib, k_carry)

                progress_bar.update()

        # Concatenate all segments along time dimension
        # motion_vec shape: [B, T, J, C]
        full_motion = torch.cat(all_motion_segments, dim=1)
        # Apply segment boundary blending if requested
        if use_blend and len(all_motion_segments) > 1:
            print_log("Applying segment boundary blending...")
            # Convert torch to numpy for blending
            motion_np = full_motion.squeeze(0).cpu().numpy()  # (T, J, C)
            motion_flat = motion_np.reshape(motion_np.shape[0], -1)  # (T, J*C)

            # Compute boundary positions (accounting for overlaps)
            boundaries = []
            current_pos = all_motion_segments[0].shape[1]
            for seg_idx in range(1, len(all_motion_segments)):
                seg_len = all_motion_segments[seg_idx].shape[1]
                # Boundary is where overlapping region ends
                boundaries.append(current_pos)
                current_pos += seg_len

            # Apply Gaussian blending around each boundary (±5 frames)
            blend_width = 5
            for boundary in boundaries:
                if boundary - blend_width < 0 or boundary + blend_width >= len(motion_flat):
                    continue

                # Create Gaussian blend kernel
                t = np.linspace(-2, 2, 2 * blend_width)
                kernel = np.exp(-0.5 * t**2)
                kernel = kernel / kernel.sum()  # Normalize

                # Apply smoothing in the blend region
                region_start = boundary - blend_width
                region_end = boundary + blend_width
                region = motion_flat[region_start:region_end]

                # Smooth each dimension
                for dim in range(region.shape[1]):
                    region[:, dim] = np.convolve(region[:, dim], kernel, mode='same')

                motion_flat[region_start:region_end] = region

            # Convert back to torch
            motion_blended = motion_flat.reshape(motion_np.shape)
            full_motion = torch.from_numpy(motion_blended).unsqueeze(0).to(full_motion.device).to(full_motion.dtype)
            print_log(f"Blending applied at {len(boundaries)} segment boundaries")

        print_log(f"Total motion frames: {full_motion.shape[1]}")

        # Post-process to SMPL-X format
        smplx_dict = self.post_process_motion(
            full_motion,
            use_static=use_static,
            use_smooth=use_smooth,
            normalize=normalize,
            mocap_framerate=mocap_framerate,
            gender=gender,
            use_rollout_trans=use_rollout_trans,
        )

        smplx_dict["_prism_requested_num_frames"] = np.asarray(
            requested_segment_lengths, dtype=np.int32
        )
        smplx_dict["_prism_generation_num_frames"] = np.asarray(
            num_frames_per_segment_list, dtype=np.int32
        )
        smplx_dict["_prism_valid_num_frames"] = np.asarray(
            valid_num_frames_list, dtype=np.int32
        )
        smplx_dict["_prism_raw_decoded_num_frames"] = np.asarray(
            raw_decoded_segment_lengths, dtype=np.int32
        )
        smplx_dict["_prism_pretrim_num_frames"] = np.asarray(
            pretrim_segment_lengths, dtype=np.int32
        )
        smplx_dict["_prism_final_num_frames"] = np.asarray(
            [smplx_dict["transl"].shape[0]], dtype=np.int32
        )

        if return_motion_vec:
            return {
                "smplx_dict": smplx_dict,
                "motion_vec": full_motion.detach().cpu(),
            }

        return smplx_dict

    def decode_motion(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latents to motion.

        Args:
            latents: Latent tensor of shape [B, C, T_latent, J].

        Returns:
            Motion tensor of shape [B, T, J, C].
        """
        latents = latents * self.latents_std.to(latents.device) + self.latents_mean.to(latents.device)
        # VAE must run in fp32 — override any global AMP autocast.
        device_type = latents.device.type
        with torch.autocast(device_type, enabled=False):
            motion = self.vae.decode(latents.float())

        return motion

    def post_process_motion(
        self,
        x_dec: torch.Tensor,
        use_static: bool = False,
        use_smooth: bool = False,
        normalize: bool = True,
        mocap_framerate: float = 30.0,
        gender: str = "neutral",
        use_rollout_trans: Union[bool, str] = "xz_rollout_y_absolute",
    ) -> Dict:
        """Post-process decoded motion to SMPL-X format.

        Args:
            x_dec: Decoded motion tensor of shape [B, T, J, C].
            use_static: Whether to use post-hoc static joint refinement.
            use_smooth: Whether to apply smoothing.
            normalize: Whether to normalize facing direction and ground plane.
            mocap_framerate: Frame rate of the motion.
            gender: Gender for SMPL model.
            use_rollout_trans: For ``abs_rel`` translation, reconstruct x/z by
                cumulative relative deltas and y from the decoded absolute
                channel by default. Passing true uses full rollout; passing
                false uses decoded absolute translation directly.

        Returns:
            Dictionary containing SMPL-X parameters.
        """
        x_dec = rearrange(x_dec, "b t j d -> b t (j d)")
        x_dec = self.smpl_processor.denormalize(x_dec)
        transl_abs_rel = x_dec[..., :6]
        transl = self.smpl_processor.inv_convert_transl(
            transl_abs_rel,
            use_rollout=use_rollout_trans,
        )
        pred_poses = x_dec[..., 6:]

        pred_poses = rearrange(pred_poses, "b t (j d)-> (b t) j d", d=6)
        # Training data already uses column-major 6D convention [R00,R10,R20,R01,R11,R21]
        # (matrix_to_rotation_6d uses _stack_cols01 → columns of rotation matrix).
        # rotation_6d_to_axis_angle expects column-major input — no permutation needed.
        pred_poses = rotation_6d_to_axis_angle(pred_poses)
        pred_poses = rearrange(pred_poses, "(b t) j d -> b t (j d)", b=1)

        if use_static:
            pred_poses = self.smpl_processor.post_hoc_static_refine(
                transl, pred_poses, rot_type="axis_angle"
            )

        pred_smplx_dict = self.smpl_processor.transl_pose_to_smplx_dict(
            transl.squeeze(0),
            pred_poses.squeeze(0),
            mocap_framerate=mocap_framerate,
            gender=gender,
            rot_type="axis_angle",
        )

        if use_smooth:
            pred_smplx_dict = self.smpl_processor.smooth_smplx_dict(pred_smplx_dict)

        if normalize:
            pred_smplx_dict = self.smpl_processor.normalize_smplx_dict(pred_smplx_dict)

        return pred_smplx_dict

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_motion_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Encodes the prompt into text encoder hidden states."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = 1

        prompt_embeds = self._get_t5_prompt_embeds(
            prompt=prompt,
            num_motion_per_prompt=num_motion_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        negative_prompt_embeds = None

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_motion_per_prompt=num_motion_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    @torch.no_grad()
    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_motion_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # duplicate text embeddings for each generation per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_motion_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_motion_per_prompt, seq_len, -1
        )

        return prompt_embeds
    @torch.no_grad()
    def _get_t5_prompt_embeds_with_mask(
        self,
        prompt: Union[str, List[str]] = None,
        num_motion_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Get T5 prompt embeddings with attention mask.

        Returns:
            Tuple of:
                - prompt_embeds: Text embeddings [B*num_motion, max_seq_len, hidden_dim]
                - prompt_mask: Attention mask [B*num_motion, max_seq_len] (1 for valid, 0 for padding)
        """
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), mask.to(device)
        ).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [
                torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
                for u in prompt_embeds
            ],
            dim=0,
        )

        # Create attention mask: 1 for valid tokens, 0 for padding
        encoder_hidden_states_mask = torch.zeros(
            batch_size, max_sequence_length, dtype=torch.long, device=device
        )
        for i, seq_len in enumerate(seq_lens):
            encoder_hidden_states_mask[i, :seq_len] = 1

        # duplicate text embeddings and mask for each generation per prompt
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_motion_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(
            batch_size * num_motion_per_prompt, seq_len, -1
        )

        # Repeat mask as well
        encoder_hidden_states_mask = encoder_hidden_states_mask.repeat(num_motion_per_prompt, 1)

        return prompt_embeds, encoder_hidden_states_mask

    def encode_prompt_with_mask(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_motion_per_prompt: int = 1,
        max_sequence_length: int = 256,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """Encodes the prompt into text encoder hidden states with attention masks."""
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = 1

        prompt_embeds, prompt_mask = self._get_t5_prompt_embeds_with_mask(
            prompt=prompt,
            num_motion_per_prompt=num_motion_per_prompt,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=dtype,
        )
        negative_prompt_embeds = None
        negative_prompt_mask = None

        if do_classifier_free_guidance:
            negative_prompt = negative_prompt or ""
            negative_prompt = (
                batch_size * [negative_prompt]
                if isinstance(negative_prompt, str)
                else negative_prompt
            )

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds, negative_prompt_mask = self._get_t5_prompt_embeds_with_mask(
                prompt=negative_prompt,
                num_motion_per_prompt=num_motion_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds, prompt_mask, negative_prompt_mask
