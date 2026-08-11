"""Motius-native MaskControl network and sampling implementation.

The architecture follows MaskControl (ICCV 2025): a retrained MoMask base
transformer, a trainable transformer copy connected through zero-initialized
linear projections (the logits regularizer), and inference-time optimization
of token logits or decoder embeddings.  Runtime code is self-contained and
does not import the upstream repository.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Optional, Sequence

import torch
from einops import repeat
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from motius.models.momask.network.mask_transformer.tools import (
    gumbel_noise,
    lengths_to_mask,
)
from motius.models.momask.network.mask_transformer.transformer import (
    MaskTransformer,
)
from motius.models.momask.network.vq.encdec import Encoder
from motius.motion.representation.humanml import recover_from_ric


# The released "all joints" checkpoint is trained on these six anchors.  The
# paper calls this any-joint control; retaining the exact list is required for
# checkpoint-compatible 36-channel control input (delta + absolute xyz).
CONTROL_JOINT_IDS = (0, 10, 11, 15, 20, 21)
CONTROL_JOINT_NAMES = (
    "pelvis",
    "left_foot",
    "right_foot",
    "head",
    "left_wrist",
    "right_wrist",
)

BODY_PART_JOINTS = {
    "pelvis": (0,),
    "left_foot": (10,),
    "l_foot": (10,),
    "right_foot": (11,),
    "r_foot": (11,),
    "head": (15,),
    "left_wrist": (20,),
    "right_wrist": (21,),
    "lower": (0, 10, 11),
    "upper": (15, 20, 21),
    "left_arm": (20, 18),
    "right_arm": (21, 19),
    "legs": (10, 4, 11, 5),
    "spine": (0,),
    "full_body": CONTROL_JOINT_IDS,
    "all": CONTROL_JOINT_IDS,
}


def _run_all_layers(
    encoder: nn.TransformerEncoder,
    source: torch.Tensor,
    padding_mask: torch.Tensor,
) -> list[torch.Tensor]:
    output = source
    values = []
    for layer in encoder.layers:
        output = layer(output, src_key_padding_mask=padding_mask)
        values.append(output)
    if encoder.norm is not None:
        values[-1] = encoder.norm(values[-1])
    return values


def _forward_with_layer_controls(
    encoder: nn.TransformerEncoder,
    source: torch.Tensor,
    controls: Sequence[torch.Tensor],
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    output = source
    for layer, control in zip(encoder.layers, controls):
        output = layer(output, src_key_padding_mask=padding_mask)
        output = output + control
    if encoder.norm is not None:
        output = encoder.norm(output)
    return output


def relative_hml263_positions(motion: torch.Tensor) -> torch.Tensor:
    """Return MaskControl's differentiable relative-joint proxy.

    The STMC extension in MaskControl optimizes HML263 channels 1:67 as a
    22x3 tensor rather than recovering global joints.  Channel zero is added
    to the proxy root so gradients also reach root angular velocity.
    """

    if motion.ndim != 3 or motion.shape[-1] != 263:
        raise ValueError(f"motion must have shape (B,T,263), got {tuple(motion.shape)}")
    relative = motion[..., 1 : (22 - 1) * 3 + 4].clone()
    relative[..., :3] = relative[..., :3] + motion[..., :1]
    return relative.reshape(*relative.shape[:-1], 22, 3)


class MaskControlTransformer(MaskTransformer):
    """Checkpoint-compatible MaskControl logits regularizer."""

    def __init__(
        self,
        code_dim: int,
        cond_mode: str,
        *,
        vq_model: nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        control_joint_ids: Sequence[int] = CONTROL_JOINT_IDS,
        latent_dim: int = 384,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 6,
        dropout: float = 0.2,
        clip_dim: int = 512,
        cond_drop_prob: float = 0.1,
        clip_version=None,
        opt=None,
        **kwargs,
    ):
        super().__init__(
            code_dim=code_dim,
            cond_mode=cond_mode,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            clip_dim=clip_dim,
            cond_drop_prob=cond_drop_prob,
            clip_version=clip_version,
            opt=opt,
            **kwargs,
        )
        self.num_layers = int(num_layers)
        self.control_joint_ids = tuple(int(value) for value in control_joint_ids)
        if self.control_joint_ids != CONTROL_JOINT_IDS:
            raise ValueError(
                "the released all-joints checkpoint requires control joints "
                f"{CONTROL_JOINT_IDS}, got {self.control_joint_ids}"
            )

        self.first_zero_linear = nn.Linear(self.latent_dim, self.latent_dim)
        self.mid_zero_linear = nn.ModuleList(
            nn.Linear(self.latent_dim, self.latent_dim)
            for _ in range(self.num_layers)
        )
        self.seqTransEncoder_control = copy.deepcopy(self.seqTransEncoder)
        self.encoder_control = Encoder(
            input_emb_width=len(self.control_joint_ids) * 2 * 3,
            output_emb_width=self.latent_dim,
            down_t=2,
            stride_t=2,
            width=512,
            depth=3,
            dilation_growth_rate=3,
            activation="relu",
            norm=None,
        )

        def zero_init(module):
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.zeros_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.first_zero_linear.apply(zero_init)
        self.mid_zero_linear.apply(zero_init)
        self.vq_model = vq_model
        self.register_buffer("mean", torch.as_tensor(mean).float().clone())
        self.register_buffer("std", torch.as_tensor(std).float().clone())
        self.register_buffer(
            "mask_emb_vq",
            vq_model.quantizer.layers[0].codebook.mean(0).detach().clone(),
        )

    def _base_transformer_forward(
        self,
        motion: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        force_mask: bool,
    ) -> torch.Tensor:
        condition = self.mask_cond(condition, force_mask=force_mask)
        embedding = self.token_emb(motion) if motion.ndim == 2 else motion
        hidden = self.position_enc(self.input_process(embedding))
        text = self.cond_emb(condition).unsqueeze(0)
        source = torch.cat((text, hidden), dim=0)
        full_padding = torch.cat(
            (torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1
        )
        output = self.seqTransEncoder(
            source, src_key_padding_mask=full_padding
        )[1:]
        return self.output_process(output)

    def trans_forward(
        self,
        motion: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        force_mask: bool = False,
        control_condition: Optional[torch.Tensor] = None,
        use_control: bool = True,
    ) -> torch.Tensor:
        if not use_control:
            return self._base_transformer_forward(
                motion,
                condition,
                padding_mask,
                force_mask=force_mask,
            )
        if control_condition is None:
            raise ValueError("control_condition is required when use_control=True")

        condition = self.mask_cond(condition, force_mask=force_mask)
        embedding = self.token_emb(motion) if motion.ndim == 2 else motion
        hidden = self.position_enc(self.input_process(embedding))
        text = self.cond_emb(condition).unsqueeze(0)
        source = torch.cat((text, hidden), dim=0)
        full_padding = torch.cat(
            (torch.zeros_like(padding_mask[:, :1]), padding_mask), dim=1
        )

        control = self.encoder_control(control_condition.permute(0, 2, 1))
        control = self.first_zero_linear(control.permute(2, 0, 1))
        control_source = source + torch.cat((torch.zeros_like(text), control), dim=0)
        control_layers = _run_all_layers(
            self.seqTransEncoder_control, control_source, full_padding
        )
        control_layers = [
            projection(value)
            for projection, value in zip(self.mid_zero_linear, control_layers)
        ]

        # This deliberately uses control_source, matching training and the
        # released checkpoint.  Substituting source changes official outputs.
        output = _forward_with_layer_controls(
            self.seqTransEncoder,
            control_source,
            control_layers,
            full_padding,
        )[1:]
        return self.output_process(output)

    def forward_with_cond_scale(
        self,
        motion: torch.Tensor,
        condition: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        cond_scale: float,
        control_condition: Optional[torch.Tensor],
        use_control: bool,
        force_mask: bool = False,
    ) -> torch.Tensor:
        if force_mask:
            return self.trans_forward(
                motion,
                condition,
                padding_mask,
                force_mask=True,
                control_condition=control_condition,
                use_control=use_control,
            )
        logits = self.trans_forward(
            motion,
            condition,
            padding_mask,
            control_condition=control_condition,
            use_control=use_control,
        )
        if cond_scale == 1:
            return logits
        unconditional = self.trans_forward(
            motion,
            condition,
            padding_mask,
            force_mask=True,
            control_condition=control_condition,
            use_control=use_control,
        )
        return unconditional + (logits - unconditional) * float(cond_scale)

    def _decode_embedding(
        self, embedding: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = self.vq_model.forward_decoder(embedding)
        physical = normalized * self.std + self.mean
        joints = recover_from_ric(physical.float(), 22)
        return normalized, joints

    def _control_features(
        self,
        logits: torch.Tensor,
        token_mask: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        embedding = F.softmax(logits / max(float(temperature), 1e-10), dim=-1)
        embedding = embedding @ self.vq_model.quantizer.codebooks[0]
        embedding[token_mask] = self.mask_emb_vq
        _, predicted = self._decode_embedding(embedding)
        delta = (targets - predicted) * target_mask.unsqueeze(-1)
        absolute = targets * target_mask.unsqueeze(-1)
        ids = list(self.control_joint_ids)
        delta = delta[..., ids, :].reshape(*delta.shape[:2], -1)
        absolute = absolute[..., ids, :].reshape(*absolute.shape[:2], -1)
        return torch.cat((delta, absolute), dim=-1)

    @staticmethod
    def _control_loss(
        prediction: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        expanded = target_mask.unsqueeze(-1).expand_as(prediction)
        squared = (prediction - targets).square() * expanded
        count = expanded.sum(dim=(1, 2, 3)).clamp_min(1)
        return (squared.sum(dim=(1, 2, 3)) / count).mean()

    def _prediction_for_loss(
        self,
        embedding: torch.Tensor,
        *,
        relative_control: bool,
    ) -> torch.Tensor:
        normalized = self.vq_model.forward_decoder(embedding)
        if relative_control:
            return relative_hml263_positions(normalized)
        physical = normalized * self.std + self.mean
        return recover_from_ric(physical.float(), 22)

    def sample_base_logits(
        self,
        captions: Optional[Sequence[str]],
        token_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        time_steps: int = 10,
        cond_scale: float = 4.0,
        temperature: float = 1.0,
        force_mask: bool = False,
        use_control: bool = True,
        each_iterations: int = 0,
        each_lr: float = 0.06,
        relative_control: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample the base token layer and return ids plus final logits."""

        device = token_lengths.device
        max_tokens = int(token_lengths.max().item())
        if max_tokens > 98:
            raise ValueError("MaskControl supports at most 392 frames (98 tokens)")
        sequence_length = 49 if max_tokens <= 49 else 98
        batch_size = len(token_lengths)
        if captions is None:
            condition = torch.zeros((batch_size, 512), device=device)
        else:
            if len(captions) != batch_size:
                raise ValueError("captions and token_lengths must have equal length")
            with torch.no_grad():
                condition = self.encode_text(list(captions))

        padding_mask = ~lengths_to_mask(token_lengths, sequence_length)
        ids = torch.where(
            padding_mask,
            torch.full_like(padding_mask, self.pad_id, dtype=torch.long),
            torch.full_like(padding_mask, self.mask_id, dtype=torch.long),
        )
        embedding = self.token_emb(ids)
        mask_embedding = embedding[0, 0].detach().clone()
        scores = torch.where(padding_mask, 1e5, 0.0)
        logits = torch.ones(
            (batch_size, sequence_length, self.opt.num_tokens),
            dtype=embedding.dtype,
            device=device,
        )
        has_controls = bool(target_mask.any().item())

        for step, timestep in enumerate(torch.linspace(0, 1, time_steps, device=device)):
            masked_count = torch.round(
                self.noise_schedule(timestep) * token_lengths
            ).clamp(min=1)
            ranks = scores.argsort(dim=1).argsort(dim=1)
            is_mask = ranks < masked_count.unsqueeze(-1)
            ids = torch.where(is_mask, self.mask_id, ids)
            embedding = torch.where(
                is_mask.unsqueeze(-1), mask_embedding, embedding
            )

            control_condition = None
            if use_control:
                control_condition = self._control_features(
                    logits,
                    is_mask,
                    targets,
                    target_mask,
                    temperature,
                )
            raw_logits = self.forward_with_cond_scale(
                embedding,
                condition,
                padding_mask,
                cond_scale=cond_scale,
                control_condition=control_condition,
                use_control=use_control,
                force_mask=force_mask,
            )
            logits = raw_logits.permute(0, 2, 1).detach()

            iterations = (
                int(each_iterations)
                if each_iterations >= 0
                else (step + 1) * -int(each_iterations)
            )
            if iterations and has_controls:
                logits.requires_grad_(True)
                optimizer = torch.optim.AdamW(
                    [logits],
                    lr=float(each_lr),
                    betas=(0.5, 0.9),
                    weight_decay=1e-6,
                )
                for _ in range(iterations):
                    probabilities = F.softmax(
                        logits / max(float(temperature), 1e-10)
                        + gumbel_noise(logits),
                        dim=-1,
                    )
                    candidate = probabilities @ self.vq_model.quantizer.codebooks[0]
                    candidate = candidate.masked_fill(
                        padding_mask.unsqueeze(-1), 0
                    )
                    prediction = self._prediction_for_loss(
                        candidate, relative_control=relative_control
                    )
                    loss = self._control_loss(prediction, targets, target_mask)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
                logits = logits.detach()

            noisy_logits = (
                logits / max(float(temperature), 1e-10) + gumbel_noise(logits)
            )
            probabilities = F.softmax(noisy_logits, dim=-1)
            predicted_embedding = probabilities @ self.token_emb.weight[
                : self.opt.num_tokens
            ]
            predicted_ids = Categorical(probabilities).sample()
            ids = torch.where(is_mask, predicted_ids, ids)
            embedding = predicted_embedding

            confidence = logits.softmax(dim=-1).gather(
                2, predicted_ids.unsqueeze(-1)
            )[..., 0]
            scores = confidence.masked_fill(~is_mask, 1e5)

        ids = torch.where(padding_mask, -1, ids)
        return ids, logits

    @staticmethod
    def sample_residual_logits(
        residual_model: nn.Module,
        base_ids: torch.Tensor,
        base_logits: torch.Tensor,
        captions: Optional[Sequence[str]],
        token_lengths: torch.Tensor,
        *,
        cond_scale: float = 5.0,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """MaskControl residual sampling, including logits for final DES."""

        residual_model.process_embed_proj_weight()
        batch_size, sequence_length = base_ids.shape
        device = base_ids.device
        if captions is None:
            condition = torch.zeros((batch_size, 512), device=device)
        else:
            with torch.no_grad():
                condition = residual_model.encode_text(list(captions))
        padding_mask = ~lengths_to_mask(token_lengths, sequence_length)
        motion_ids = torch.where(padding_mask, residual_model.pad_id, base_ids)
        history = 0
        logits = torch.cat(
            (base_logits, torch.zeros_like(base_logits[..., :1])), dim=-1
        )
        all_logits = [logits]

        for quantizer in range(1, residual_model.opt.num_quantizers):
            token_embedding = residual_model.token_embed_weight[quantizer - 1]
            # The released implementation uses a near-argmax expectation for
            # the history at every residual layer.
            history = history + F.softmax(logits / 1e-8, dim=-1) @ token_embedding
            raw = residual_model.forward_with_cond_scale(
                history,
                quantizer,
                condition,
                padding_mask,
                cond_scale=cond_scale,
            )
            logits = raw.permute(0, 2, 1)
            ids = (
                logits / max(float(temperature), 1e-10) + gumbel_noise(logits)
            ).argmax(dim=-1)
            motion_ids = torch.where(padding_mask, residual_model.pad_id, ids)
            pad_value = token_embedding[..., -1]
            logits = torch.where(
                motion_ids.unsqueeze(-1) == residual_model.pad_id,
                pad_value,
                logits,
            )
            all_logits.append(logits)
        return torch.stack(all_logits, dim=-1)

    def decode_logits(
        self,
        base_logits: torch.Tensor,
        padding_mask: torch.Tensor,
        *,
        residual_model: Optional[nn.Module],
        residual_logits: Optional[torch.Tensor],
        temperature: float,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        final_iterations: int,
        final_lr: float,
        relative_control: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual_model is None:
            embedding = F.softmax(base_logits, dim=-1)
            embedding = embedding @ self.vq_model.quantizer.codebooks[0]
        else:
            if residual_logits is None:
                raise ValueError("residual_logits are required with residual_model")
            embedding = 0
            for quantizer in range(residual_logits.shape[-1]):
                quantizer_temperature = float(temperature) if quantizer == 0 else 1e-8
                probabilities = F.softmax(
                    residual_logits[..., :-1, quantizer]
                    / max(quantizer_temperature, 1e-10),
                    dim=-1,
                )
                embedding = (
                    embedding
                    + probabilities @ self.vq_model.quantizer.codebooks[quantizer]
                )
        embedding = embedding.masked_fill(padding_mask.unsqueeze(-1), 0)

        if final_iterations and bool(target_mask.any().item()):
            embedding = embedding.detach().clone().requires_grad_(True)
            optimizer = torch.optim.AdamW(
                [embedding],
                lr=float(final_lr),
                betas=(0.5, 0.9),
                weight_decay=1e-6,
            )
            for _ in range(int(final_iterations)):
                prediction = self._prediction_for_loss(
                    embedding, relative_control=relative_control
                )
                loss = self._control_loss(prediction, targets, target_mask)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            embedding = embedding.detach()

        normalized = self.vq_model.forward_decoder(embedding)
        physical = normalized * self.std + self.mean
        return normalized, physical

    def sample_motion(
        self,
        captions: Optional[Sequence[str]],
        frame_lengths: torch.Tensor,
        targets: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        residual_model: Optional[nn.Module],
        time_steps: int = 10,
        cond_scale: float = 4.0,
        residual_cond_scale: float = 5.0,
        temperature: float = 1.0,
        residual_temperature: float = 1.0,
        use_control: bool = True,
        each_iterations: int = 0,
        final_iterations: int = 0,
        each_lr: float = 0.06,
        final_lr: float = 0.06,
        relative_control: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate normalized and physical HML263 sequences."""

        token_lengths = torch.div(frame_lengths, 4, rounding_mode="floor")
        max_tokens = 49 if int(token_lengths.max()) <= 49 else 98
        canvas_frames = max_tokens * 4
        if targets.shape[:2] != (len(frame_lengths), canvas_frames):
            raise ValueError(
                "targets must use the MaskControl canvas shape "
                f"({len(frame_lengths)},{canvas_frames},22,3), got {tuple(targets.shape)}"
            )
        if target_mask.shape != targets.shape[:-1]:
            raise ValueError("target_mask must have shape (B,T,22)")

        grad_context = torch.enable_grad() if (
            each_iterations or final_iterations
        ) else nullcontext()
        with grad_context:
            base_ids, base_logits = self.sample_base_logits(
                captions,
                token_lengths,
                targets,
                target_mask,
                time_steps=time_steps,
                cond_scale=cond_scale,
                temperature=temperature,
                use_control=use_control,
                each_iterations=each_iterations,
                each_lr=each_lr,
                relative_control=relative_control,
            )
            padding_mask = ~lengths_to_mask(token_lengths, max_tokens)
            residual_logits = None
            if residual_model is not None:
                with torch.no_grad():
                    residual_logits = self.sample_residual_logits(
                        residual_model,
                        base_ids,
                        base_logits,
                        captions,
                        token_lengths,
                        cond_scale=residual_cond_scale,
                        temperature=residual_temperature,
                    )
            return self.decode_logits(
                base_logits,
                padding_mask,
                residual_model=residual_model,
                residual_logits=residual_logits,
                temperature=temperature,
                targets=targets,
                target_mask=target_mask,
                final_iterations=final_iterations,
                final_lr=final_lr,
                relative_control=relative_control,
            )


__all__ = [
    "BODY_PART_JOINTS",
    "CONTROL_JOINT_IDS",
    "CONTROL_JOINT_NAMES",
    "MaskControlTransformer",
    "relative_hml263_positions",
]
