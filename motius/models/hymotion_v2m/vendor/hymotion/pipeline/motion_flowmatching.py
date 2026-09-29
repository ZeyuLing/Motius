from __future__ import annotations
import os
import torch
from torchdiffeq import odeint
import torch.nn.functional as F
from .smpl_lite import SmplxLiteJ24
from .metric import compute_jitter
from ..utils.loaders import load_object
from ..datasets.geometry import rotation_matrix_to_rot6d, angle_axis_to_rotation_matrix, rot6d_to_rotation_matrix


def randn_tensor(
    shape,
    generator=None,
    device=None,
    dtype=None,
    layout=None,
):
    """A helper function to create random tensors on the desired `device` with the desired `dtype`. When
    passing a list of generators, you can seed each batch size individually. If CPU generators are passed, the tensor
    is always created on the CPU.
    """
    # device on which tensor is created defaults to device
    rand_device = device
    batch_size = shape[0]

    layout = layout or torch.strided
    device = device or torch.device("cpu")

    if generator is not None:
        gen_device_type = generator.device.type if not isinstance(generator, list) else generator[0].device.type
        if gen_device_type != device.type and gen_device_type == "cpu":
            rand_device = "cpu"
            if device != "mps":
                print(
                    f"The passed generator was created on 'cpu' even though a tensor on {device} was expected."
                    f" Tensors will be created on 'cpu' and then moved to {device}. Note that one can probably"
                    f" slighly speed up this function by passing a generator that was created on the {device} device."
                )
        elif gen_device_type != device.type and gen_device_type == "cuda":
            raise ValueError(f"Cannot generate a {device} tensor from a generator of type {gen_device_type}.")

    # make sure generator list of length 1 is treated like a non-list
    if isinstance(generator, list) and len(generator) == 1:
        generator = generator[0]

    if isinstance(generator, list):
        shape = (1,) + shape[1:]
        latents = [
            torch.randn(shape, generator=generator[i], device=rand_device, dtype=dtype, layout=layout)
            for i in range(batch_size)
        ]
        latents = torch.cat(latents, dim=0).to(device)
    else:
        latents = torch.randn(shape, generator=generator, device=rand_device, dtype=dtype, layout=layout).to(device)

    return latents


def rollout_local_transl_vel(local_transl_vel, global_orient, transl_0=None):
    """
    transl velocity is in local coordinate (or, SMPL-coord)
    Args:
        local_transl_vel: (*, L, 3)
        global_orient: (*, L, 3, 3)
        transl_0: (*, 1, 3), if not provided, the start point is 0
    Returns:
        transl: (*, L, 3)
    """
    transl_vel = torch.einsum("...lij,...lj->...li", global_orient, local_transl_vel)

    # set start point
    if transl_0 is None:
        transl_0 = transl_vel[..., :1, :].clone().detach().zero_()
    transl_ = torch.cat([transl_0, transl_vel[..., :-1, :]], dim=-2)

    # rollout from start point
    transl = torch.cumsum(transl_, dim=-2)
    return transl


class MotionGeneration(torch.nn.Module):
    def __init__(
        self,
        train_frames=300,
        max_text_len=128,
        num_joints=22,
        drop_text_prob=0.1,
        text_encoder_type="t5",
        normalize_type="mean_std_channel",
        network_module=None,
        network_module_args=None,
        **kwargs,
    ):
        super().__init__()
        self.train_frames = train_frames
        self.max_text_len = max_text_len
        print(f"[{self.__class__.__name__}] train_frames: {train_frames}, max_text_len: {max_text_len}")
        self.body_model = SmplxLiteJ24()
        self.num_joints = num_joints
        self.normalize_type = normalize_type
        self.network_module = network_module
        self.network_module_args = network_module_args
        self.motion_transformer = load_object(network_module, network_module_args)
        self.text_encoder = None
        self.drop_text_prob = drop_text_prob
        self.text_encoder_type = text_encoder_type

    def set_epoch(self, epoch):
        self.current_epoch = epoch

    def build_text_encoder(self):
        if self.text_encoder_type == "t5":
            # No hardcoded foreign cache path; use HF's default cache unless the
            # caller overrides it.  (V2M is feature-conditioned and never builds
            # a T5 encoder, so this path is unused for V2M.)
            cache_dir = os.environ.get("HYMOTION_T5_CACHE_DIR") or None
            local_only = cache_dir is not None
            from transformers import T5Tokenizer, T5EncoderModel

            tokenizer = T5Tokenizer.from_pretrained(
                "google/t5-v1_1-xxl", cache_dir=cache_dir, local_files_only=local_only
            )
            print(f">>> Loading T5 encoder...")
            text_encoder = T5EncoderModel.from_pretrained(
                "google/t5-v1_1-xxl", cache_dir=cache_dir, local_files_only=local_only
            )
            print(f">>> T5 encoder loaded...")
            text_encoder = text_encoder.eval().requires_grad_(False)
            print(f">>>  moving T5 encoder to GPU...")
            text_encoder.to(next(self.motion_transformer.parameters()).device)
            print(f">>> T5 encoder moved to GPU...")
            self.tokenizer = tokenizer
            self.text_encoder = text_encoder
        elif self.text_encoder_type == "qwen":
            from ..network.text_encoders.text_encoder import HYTextModel

            self.text_encoder = HYTextModel(
                llm_type="qwen3_embedding",
                max_length_llm=512,
            )
            self.text_encoder.to(next(self.motion_transformer.parameters()).device)
        else:
            raise NotImplementedError(f"text_encoder_type {self.text_encoder_type} not implemented")

    def load_in_demo(self, ckptname, cache_dir, build_text_encoder=True, allow_empty_ckpt=False):
        if not allow_empty_ckpt:
            assert os.path.exists(ckptname), f"{ckptname} not found"
        mean_std_name = os.path.join(cache_dir, "mean_std.pt")
        assert os.path.exists(mean_std_name), f"{mean_std_name} not found"
        if os.path.exists(ckptname):
            self.load_state_dict(torch.load(ckptname))
        self.load_mean_std(mean_std_name)
        if build_text_encoder:
            self.build_text_encoder()
        self.motion_transformer.eval()

    def encode_text(self, text_dict):
        if self.text_encoder is None:
            print(f">>> Text encoder not found, using dummy text encoder...")
            print(f">>> max_text_len: {text_dict}")
            batch_size = len(text_dict["text"])
            return {
                "hidden_state": torch.zeros(
                    batch_size, self.max_text_len, self.motion_transformer.context_in_dim, device=self.rot6d_mean.device
                ),
                "hidden_state_length": torch.zeros(batch_size, device=self.rot6d_mean.device).long()
                + self.max_text_len,
            }

        text = text_dict["text"]
        if self.text_encoder_type == "t5":
            text_tokens = self.tokenizer(
                text,
                # truncation=True,
                max_length=self.max_text_len,
                return_length=False,
                return_overflowing_tokens=False,
                padding="max_length",
                return_tensors="pt",
            )
            input_ids = text_tokens["input_ids"].to(self.rot6d_mean.device)
            attention_mask = text_tokens["attention_mask"].to(self.rot6d_mean.device)

            with torch.no_grad():
                text_tokens = self.text_encoder(input_ids, attention_mask=attention_mask)
                last_hidden_state = text_tokens.last_hidden_state
            length = attention_mask.sum(dim=-1)
            return {
                "hidden_state": last_hidden_state,
                "hidden_state_length": length,
            }
        elif self.text_encoder_type == "qwen":
            with torch.no_grad():
                ctxt_input, ctxt_length = self.text_encoder._encode_llm(text=[text])
                # vtxt_input = self.text_encoder._encode_clip(text=[text])
            # padding
            ctxt_input = self.padding_text_feature(ctxt_input)

            return {
                "hidden_state": ctxt_input,
                "hidden_state_length": ctxt_length,
            }

    def encode_motion(self, batch):
        rot6d = batch["rot6d"]
        if self.normalize_type == "mean_std_channel":
            rot6d_mean, rot6d_std = batch["rot6d_mean"][0], batch["rot6d_std"][0]
            rot6d_std = rot6d_std.clone()
            rot6d_std[rot6d_std < 1e-3] = 1.0
            rot6d_normalize = (rot6d - rot6d_mean[None, None]) / rot6d_std
        else:
            raise NotImplementedError
        # process trans
        trans_vel = batch["trans_vel"]
        if self.normalize_type == "mean_std_channel":
            # trans_mean: (batch_size, 3)
            # trans: (batch_size, seqlen, 3)
            trans_mean, trans_std = batch["trans_vel_mean"][0][None, None], batch["trans_vel_std"][0][None, None]
            trans_std = trans_std.clone()
            trans_std[trans_std < 1e-3] = 1.0
            trans_normalize = (trans_vel - trans_mean) / trans_std
        else:
            raise NotImplementedError

        latent = torch.cat([trans_normalize, rot6d_normalize.reshape(rot6d.shape[0], rot6d.shape[1], -1)], dim=-1)
        return latent

    def decode_motion(self, latent, batch):
        """
        upsample: 是否对latent进行上采样；训练时期计算loss，这里不需要进行上采样
        """
        trans_vel_normalize = latent[:, :, :3]
        rot6d_normalize = latent[:, :, 3:]
        rot6d_normalize = rot6d_normalize.reshape(rot6d_normalize.shape[0], rot6d_normalize.shape[1], -1, 6)

        if self.normalize_type == "mean_std_channel":
            rot6d_mean, rot6d_std = batch["rot6d_mean"][0], batch["rot6d_std"][0]
            rot6d = rot6d_normalize * rot6d_std[None, None] + rot6d_mean[None, None]
            trans_mean, trans_std = batch["trans_vel_mean"][0], batch["trans_vel_std"][0]
            trans_vel = trans_vel_normalize * trans_std[None, None] + trans_mean[None, None]
        else:
            raise NotImplementedError(f"normalize_type {self.normalize_type} not implemented")

        return {
            "trans_vel": trans_vel,
            "rot6d": rot6d,
        }

    @staticmethod
    def noise_from_seeds(latent, seeds, seed_start=0):
        if isinstance(seeds, int):
            seeds = list(range(seeds))
            noise_list = []
            for seed in seeds:
                generator = torch.Generator().manual_seed(seed + seed_start)
                # 为每个seed生成一个噪声样本
                noise_sample = (
                    randn_tensor((1, *latent.shape[1:]), generator=generator).to(latent.device).to(latent.dtype)
                )
                noise_list.append(noise_sample)
            # 拼接所有噪声样本
            noise = torch.cat(noise_list, dim=0)
        else:
            noise_list = []
            for seed in seeds:
                generator = torch.Generator().manual_seed(seed)
                # 为每个seed生成一个噪声样本
                noise_sample = (
                    randn_tensor((1, *latent.shape[1:]), generator=generator).to(latent.device).to(latent.dtype)
                )
                noise_list.append(noise_sample)
            # 拼接所有噪声样本
            noise = torch.cat(noise_list, dim=0)
        return noise

    def load_mean_std(self, mean_std_name):
        mean_std = torch.load(mean_std_name)
        for key, value in mean_std.items():
            self.register_buffer(key, value)


def length_to_mask(lengths, max_len):
    """
        lengths: (B, 1)
        max_len: int
    Returns: (B, max_len)
    """
    assert lengths.max() <= max_len, f"lengths.max()={lengths.max()} > max_len={max_len}"
    if lengths.ndim == 1:
        lengths = lengths.unsqueeze(1)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths
    return mask


class MotionFlowMatching(MotionGeneration):
    def __init__(
        self, loss_type="latent_flow", odeint_kwargs={"method": "euler"}, validation_steps=50, fps=30, **kwargs
    ):
        super().__init__(**kwargs)
        self.odeint_kwargs = odeint_kwargs
        self.validation_steps = validation_steps
        self.fps = fps
        self.loss_type = loss_type

    def compute_loss(self, batch, flow_gt, flow_pred, x1_gt, x1_pred, length):
        mask = length_to_mask(length, flow_gt.shape[1])
        # loss = F.mse_loss(flow_pred[mask], flow_gt[mask], reduction="mean")
        # loss += F.mse_loss(x1_pred[mask], x1_gt[mask], reduction="mean")
        gt_decode = self.decode_motion(x1_gt, batch)
        pred_decode = self.decode_motion(x1_pred, batch)
        loss_rot6d = F.mse_loss(gt_decode["rot6d"][mask], pred_decode["rot6d"][mask], reduction="mean")
        loss_trans = F.mse_loss(gt_decode["trans_vel"][mask], pred_decode["trans_vel"][mask], reduction="mean")
        loss = loss_rot6d * 400 + loss_trans * 100
        return loss

    def forward_in_training(self, batch):
        latent = self.encode_motion(batch)
        dtype, device = latent.dtype, latent.device

        # condition: (B, L, D)
        condition = batch["hidden_state"]
        if self.training:
            flag_drop_condition = torch.rand(condition.shape[0], device=device) < self.drop_text_prob
            condition = (1 - flag_drop_condition[:, None, None].float()) * condition
        # x0 is gaussian noise
        x0 = torch.randn(latent.shape, dtype=dtype).to(device)
        x1 = latent
        # Sample a random timestep for each image
        # time step
        # 1000 与MDM兼容
        time = torch.rand((latent.shape[0],), dtype=dtype).to(device)

        # sample xt (φ_t(x) in the paper)
        t = time.unsqueeze(-1).unsqueeze(-1)
        φ = (1 - t) * x0 + t * x1
        flow = x1 - x0

        pred = self.motion_transformer(φ, condition, time, batch["length"].long(), batch["hidden_state_length"].long())

        # flow matching loss
        mask = length_to_mask(batch["length"], x1.shape[1])
        if self.loss_type == "latent_flow":
            loss = F.mse_loss(pred[mask], flow[mask], reduction="mean")
        elif self.loss_type == "decode_flow":
            pred_decode = self.decode_motion(pred, batch)
            gt_decode = self.decode_motion(x1, batch)
            loss_rot6d = F.mse_loss(pred_decode["rot6d"][mask], gt_decode["rot6d"][mask], reduction="mean")
            loss_trans = F.mse_loss(pred_decode["trans_vel"][mask], gt_decode["trans_vel"][mask], reduction="mean")
            loss = loss_rot6d * 400 + loss_trans * 100

        return {
            "latent": latent,
            "model_output": pred,
            "loss": loss,
        }

    def roll_out_trans(self, trans, fps=30):
        xz_vel = trans[:, :2] / fps
        height = trans[:, 2:3]
        xz = torch.cumsum(xz_vel, dim=0)
        trans = torch.cat(
            [
                xz[:, :1],
                height,
                xz[:, 1:],
            ],
            dim=-1,
        )
        return trans

    @torch.no_grad()
    def validate(self, batch, seeds=[0, 1, 2, 3]):
        latent = self.encode_motion(batch)
        dtype, device = latent.dtype, latent.device
        # neural ode
        hidden_state = batch["hidden_state"]
        hidden_state_length = batch["hidden_state_length"]
        length = batch["length"]

        def fn(t, x):
            # predict flow
            pred = self.motion_transformer(x, hidden_state, t, length, hidden_state_length)
            return pred

        # duplicate test corner for inner time step oberservation
        t = torch.linspace(0, 1, self.validation_steps + 1, device=device, dtype=dtype)
        y0 = self.noise_from_seeds(latent, seeds)
        with torch.no_grad():
            trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        sampled = trajectory[-1]

        model_output = self.decode_motion(sampled, batch)  # BxNxJxD
        body_poses = model_output["rot6d"][:, :, 1:]
        global_orient = model_output["rot6d"][:, :, :1]
        global_orient_rot = rot6d_to_rotation_matrix(model_output["rot6d"][:, :, 0])
        betas = batch["betas"]
        trans_vel = model_output["trans_vel"]

        keypoints3d_all = []
        assert betas.shape[0] == 1, "betas should be (1, 16)"
        for bs in range(body_poses.shape[0]):
            output = self.body_model(
                body_poses[bs],
                betas,
                global_orient[bs],
                rollout_local_transl_vel(trans_vel[bs] / self.fps, global_orient_rot[bs]),
            )
            keypoints3d_all.append(output)
        keypoints3d_all = torch.stack(keypoints3d_all, dim=0)
        global_orient_rot_gt = rot6d_to_rotation_matrix(batch["rot6d"][:, :, 0])
        keypoints3d_all_gt = self.body_model(
            batch["rot6d"][0, :, 1:],
            batch["betas"],
            batch["rot6d"][0, :, :1],
            rollout_local_transl_vel(batch["trans_vel"][0] / self.fps, global_orient_rot_gt[0]),
        )

        model_output["keypoints3d"] = keypoints3d_all
        dist = torch.norm(keypoints3d_all[None] - keypoints3d_all[:, None], dim=-1).mean()
        batch["keypoints3d"] = keypoints3d_all_gt

        return {
            "metrics": {
                "mse_rot6d": F.mse_loss(model_output["rot6d"], batch["rot6d"]).item(),
                "jitter": compute_jitter(model_output["keypoints3d"], self.fps).item(),
                "jitter_gt": compute_jitter(keypoints3d_all_gt, self.fps).item(),
                "inner_dist": dist.item(),
            },
            "model_output": model_output,
            "gt": batch,
        }

    def get_shape_of_noise(self):
        batch = {
            "rot6d": torch.zeros(1, self.train_frames, self.num_joints, 6).to(
                next(self.motion_transformer.parameters()).device
            ),
            "trans_vel": torch.zeros(1, self.train_frames, 3).to(next(self.motion_transformer.parameters()).device),
            "betas": torch.zeros(1, 16).to(next(self.motion_transformer.parameters()).device),
        }
        batch["rot6d_mean"] = self.rot6d_mean[None, : self.num_joints]
        batch["rot6d_std"] = self.rot6d_std[None, : self.num_joints]
        batch["trans_vel_mean"] = self.trans_vel_mean[None]
        batch["trans_vel_std"] = self.trans_vel_std[None]
        latent = self.encode_motion(batch)
        return latent.shape

    def padding_text_feature(self, hidden_state):
        """
        hidden_state: (B, l, D)  => (B, max_text_len, D)
        """
        if hidden_state.shape[1] < self.max_text_len:
            hidden_state = torch.cat(
                [
                    hidden_state,
                    torch.zeros(
                        1, self.max_text_len - hidden_state.shape[1], hidden_state.shape[2], device=hidden_state.device
                    ),
                ],
                dim=1,
            )
        return hidden_state

    def forward(self, noise, noise_length, hidden_state_dict, t, cfg_scale=2.0):
        # noise: (B, train_frames, D)
        # noise_length: (B, 1)
        # hidden_state: (B, max_text_len, D)
        # hidden_state_length: (B, 1)
        # t: (B, 1)
        hidden_state = hidden_state_dict["hidden_state"]
        hidden_state_length = hidden_state_dict["hidden_state_length"]
        hidden_state = self.padding_text_feature(hidden_state)
        if cfg_scale > 1.0:
            pred_uncond = self.motion_transformer(
                noise, torch.zeros_like(hidden_state), t, noise_length, hidden_state_length
            )
            pred_cond = self.motion_transformer(noise, hidden_state, t, noise_length, hidden_state_length)
            pred = pred_uncond + (pred_cond - pred_uncond) * cfg_scale
        else:
            pred = self.motion_transformer(noise, hidden_state, t, noise_length, hidden_state_length)
        return pred

    def decode_motion_from_latent(self, latent):
        mean_std = {
            "rot6d_mean": self.rot6d_mean[None, : self.num_joints],
            "rot6d_std": self.rot6d_std[None, : self.num_joints],
            "trans_vel_mean": self.trans_vel_mean[None],
            "trans_vel_std": self.trans_vel_std[None],
        }
        latent = self.decode_motion(latent, mean_std)
        keypoints3d_all = []
        global_orient_rot = rot6d_to_rotation_matrix(latent["rot6d"][:, :, 0])
        trans_all = []
        body_poses = latent["rot6d"][:, :, 1:]
        global_orient = latent["rot6d"][:, :, :1]
        for bs in range(latent["rot6d"].shape[0]):
            transl_bs = rollout_local_transl_vel(latent["trans_vel"][bs] / self.fps, global_orient_rot[bs])
            trans_all.append(transl_bs)
            if "betas" not in latent:
                latent["betas"] = torch.zeros(1, 16, device=latent["rot6d"].device)
            output = self.body_model(body_poses[bs], latent["betas"], global_orient[bs], transl_bs)
            keypoints3d_all.append(output)
        k3d_all = torch.stack(keypoints3d_all, dim=0)
        latent["transl"] = torch.stack(trans_all, dim=0)
        latent["keypoints3d"] = k3d_all
        return latent

    def generate(self, text, seed_input, duration_slider, cfg_scale=2.0):
        # fps 30
        frames = int(duration_slider * 30)
        device = self.rot6d_mean.device
        batch = {
            "rot6d": torch.zeros(1, self.train_frames, self.num_joints, 6, device=device),
            "trans_vel": torch.zeros(1, self.train_frames, 3, device=device),
            "betas": torch.zeros(1, 16, device=device),
        }
        batch["rot6d_mean"] = self.rot6d_mean[None, : self.num_joints]
        batch["rot6d_std"] = self.rot6d_std[None, : self.num_joints]
        batch["trans_vel_mean"] = self.trans_vel_mean[None]
        batch["trans_vel_std"] = self.trans_vel_std[None]
        latent = self.encode_motion(batch)
        hidden_state_dict = self.encode_text({"text": [text]})
        dtype, device = latent.dtype, latent.device
        # neural ode
        hidden_state_length = hidden_state_dict["hidden_state_length"].long().to(device)
        print(" >>> hidden_state_length: ", hidden_state_length)
        hidden_state = self.padding_text_feature(hidden_state_dict["hidden_state"])
        length = torch.tensor([frames]).to(device)

        def fn(t, x):
            # predict flow
            if cfg_scale >= 0:
                pred_uncond = self.motion_transformer(x, torch.zeros_like(hidden_state), t, length, hidden_state_length)
                pred_cond = self.motion_transformer(x, hidden_state, t, length, hidden_state_length)
                pred = pred_uncond + (pred_cond - pred_uncond) * cfg_scale
            else:
                pred = self.motion_transformer(x, hidden_state, t, length, hidden_state_length)
            return pred

        # duplicate test corner for inner time step oberservation
        t = torch.linspace(0, 1, self.validation_steps + 1, device=device, dtype=dtype)
        y0 = self.noise_from_seeds(latent, seed_input)
        with torch.no_grad():
            trajectory = odeint(fn, y0, t, **self.odeint_kwargs)
        sampled = trajectory[-1]

        model_output = self.decode_motion(sampled, batch)  # BxNxJxD
        body_poses = model_output["rot6d"][:, :, 1:]
        global_orient = model_output["rot6d"][:, :, :1]
        betas = batch["betas"]
        transl = model_output["trans_vel"]

        keypoints3d_all = []
        assert betas.shape[0] == 1, "betas should be (1, 16)"
        global_orient_rot = rot6d_to_rotation_matrix(model_output["rot6d"][:, :, 0])
        global_transl = []
        for bs in range(body_poses.shape[0]):
            trans = rollout_local_transl_vel(transl[bs] / self.fps, global_orient_rot[bs])
            global_transl.append(trans)
            output = self.body_model(body_poses[bs], betas, global_orient[bs], trans)
            keypoints3d_all.append(output)
        keypoints3d_all = torch.stack(keypoints3d_all, dim=0)
        global_transl = torch.stack(global_transl, dim=0)
        keypoints3d_all = keypoints3d_all[:, :frames]
        return {
            "keypoints3d": keypoints3d_all,
            "rot6d": model_output["rot6d"][:, :frames],
            "betas": betas,
            "transl": global_transl[:, :frames],
            "text": text,
        }
