"""Checkpoint-compatible MaskControl neural transformer."""
from __future__ import annotations
import copy
from typing import Optional, Sequence
import torch
from torch import nn
from motius.models.momask.network.mask_transformer.transformer import MaskTransformer
from motius.models.momask.network.vq.encdec import Encoder
from motius.motion.conditioning.maskcontrol import CONTROL_JOINT_IDS

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


class MaskControlTransformer(MaskTransformer):
    """Predict logits from token, text, and spatial condition features."""
    def __init__(
        self,
        code_dim: int,
        cond_mode: str,
        *,
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


    def forward(self, motion, condition, padding_mask, *, force_mask=False,
                control_condition=None, use_control=True):
        return self.trans_forward(motion, condition, padding_mask,
            force_mask=force_mask, control_condition=control_condition,
            use_control=use_control)
