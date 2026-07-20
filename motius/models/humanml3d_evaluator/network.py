"""HumanML3D text-motion matching encoders used by the official benchmark."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


def _init_weight(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


class MovementConvEncoder(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(input_size, hidden_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv1d(hidden_size, output_size, 4, 2, 1),
            nn.Dropout(0.2, inplace=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(_init_weight)
        self.out_net.apply(_init_weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.main(inputs.permute(0, 2, 1)).permute(0, 2, 1)
        return self.out_net(outputs)


class TextEncoderBiGRUCo(nn.Module):
    def __init__(
        self,
        word_size: int,
        pos_size: int,
        hidden_size: int,
        output_size: int,
    ) -> None:
        super().__init__()
        self.pos_emb = nn.Linear(pos_size, word_size)
        self.input_emb = nn.Linear(word_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(_init_weight)
        self.pos_emb.apply(_init_weight)
        self.output_net.apply(_init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn(2, 1, hidden_size))

    def forward(
        self,
        word_embeddings: torch.Tensor,
        pos_one_hot: torch.Tensor,
        caption_lengths: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = word_embeddings.shape[0]
        inputs = word_embeddings + self.pos_emb(pos_one_hot)
        inputs = self.input_emb(inputs)
        hidden = self.hidden.repeat(1, batch_size, 1)
        packed = pack_padded_sequence(
            inputs,
            caption_lengths.detach().cpu().tolist(),
            batch_first=True,
        )
        _, final_hidden = self.gru(packed, hidden)
        final_hidden = torch.cat([final_hidden[0], final_hidden[1]], dim=-1)
        return self.output_net(final_hidden)


class MotionEncoderBiGRUCo(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.input_emb = nn.Linear(input_size, hidden_size)
        self.gru = nn.GRU(
            hidden_size, hidden_size, batch_first=True, bidirectional=True
        )
        self.output_net = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(hidden_size, output_size),
        )
        self.input_emb.apply(_init_weight)
        self.output_net.apply(_init_weight)
        self.hidden_size = hidden_size
        self.hidden = nn.Parameter(torch.randn(2, 1, hidden_size))

    def forward(
        self, movements: torch.Tensor, movement_lengths: torch.Tensor
    ) -> torch.Tensor:
        batch_size = movements.shape[0]
        inputs = self.input_emb(movements)
        hidden = self.hidden.repeat(1, batch_size, 1)
        packed = pack_padded_sequence(
            inputs,
            movement_lengths.detach().cpu().tolist(),
            batch_first=True,
        )
        _, final_hidden = self.gru(packed, hidden)
        final_hidden = torch.cat([final_hidden[0], final_hidden[1]], dim=-1)
        return self.output_net(final_hidden)


__all__ = [
    "MotionEncoderBiGRUCo",
    "MovementConvEncoder",
    "TextEncoderBiGRUCo",
]
