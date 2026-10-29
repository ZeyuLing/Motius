"""SPEC camera distribution network; angle decoding lives in the pipeline."""
import torch
from torch import nn
from .camera_resnet import ResNet,Bottleneck

class CameraRegressorNetwork(nn.Module):

    def __init__(self, backbone='resnet50', num_fc_layers=1, num_fc_channels=1024, num_out_channels=256):
        super(CameraRegressorNetwork, self).__init__()
        if backbone!='resnet50':raise ValueError('Released SPEC uses resnet50')
        self.backbone = ResNet(Bottleneck,[3,4,6,3])
        self.num_out_channels = num_out_channels
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        out_channels = 2048
        assert num_fc_layers > 0, 'Number of FC layers should be more than 0'
        if num_fc_layers == 1:
            self.fc_vfov = nn.Linear(out_channels, num_out_channels)
            self.fc_pitch = nn.Linear(out_channels, num_out_channels)
            self.fc_roll = nn.Linear(out_channels, num_out_channels)
            nn.init.normal_(self.fc_vfov.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_vfov.bias, 0)
            nn.init.normal_(self.fc_pitch.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_pitch.bias, 0)
            nn.init.normal_(self.fc_roll.weight, mean=0, std=0.01)
            nn.init.constant_(self.fc_roll.bias, 0)
        else:
            self.fc_vfov = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)
            self.fc_pitch = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)
            self.fc_roll = self._get_fc_layers(num_fc_layers, num_fc_channels, out_channels)

    def _get_fc_layers(self, num_layers, num_channels, inp_channels):
        modules = []
        for i in range(num_layers):
            if i == 0:
                modules.append(nn.Linear(inp_channels, num_channels))
            elif i == num_layers - 1:
                modules.append(nn.Linear(num_channels, self.num_out_channels))
            else:
                modules.append(nn.Linear(num_channels, num_channels))
        return nn.Sequential(*modules)

    def forward(self, images):
        x = self.backbone(images)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        vfov = self.fc_vfov(x)
        pitch = self.fc_pitch(x)
        roll = self.fc_roll(x)
        return [vfov, pitch, roll]
