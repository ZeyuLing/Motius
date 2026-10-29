"""Selected original TM2T neural architecture components."""

import torch
from torch import nn
from .layers import ResBlock


def init_weight(m):
    if (
        isinstance(m, nn.Conv1d)
        or isinstance(m, nn.Linear)
        or isinstance(m, nn.ConvTranspose1d)
    ):
        nn.init.xavier_normal_(m.weight)
        # m.bias.data.fill_(0.01)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


class VQEncoderV3(nn.Module):
    def __init__(self, input_size, channels, n_down):
        super(VQEncoderV3, self).__init__()
        assert len(channels) == n_down
        layers = [
            nn.Conv1d(input_size, channels[0], 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            ResBlock(channels[0]),
        ]

        for i in range(1, n_down):
            layers += [
                nn.Conv1d(channels[i - 1], channels[i], 4, 2, 1),
                nn.LeakyReLU(0.2, inplace=True),
                ResBlock(channels[i]),
            ]
        self.main = nn.Sequential(*layers)
        # self.out_net = nn.Linear(output_size, output_size)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        # print(outputs.shape)
        return outputs


class VQDecoderV3(nn.Module):
    def __init__(self, input_size, channels, n_resblk, n_up):
        super(VQDecoderV3, self).__init__()
        assert len(channels) == n_up + 1
        if input_size == channels[0]:
            layers = []
        else:
            layers = [
                nn.Conv1d(input_size, channels[0], kernel_size=3, stride=1, padding=1)
            ]

        for i in range(n_resblk):
            layers += [ResBlock(channels[0])]
        # channels = channels
        for i in range(n_up):
            layers += [
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv1d(
                    channels[i], channels[i + 1], kernel_size=3, stride=1, padding=1
                ),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        layers += [
            nn.Conv1d(channels[-1], channels[-1], kernel_size=3, stride=1, padding=1)
        ]
        self.main = nn.Sequential(*layers)
        self.main.apply(init_weight)
        # self.out_net.apply(init_weight)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        outputs = self.main(inputs).permute(0, 2, 1)
        return outputs
