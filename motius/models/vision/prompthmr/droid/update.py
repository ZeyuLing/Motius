"""DROID-Calib neural recurrent update and graph aggregation layers."""
import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from .gru import ConvGRU
from .clipping import GradientClip

class GraphAgg(nn.Module):

    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.eta = nn.Sequential(nn.Conv2d(128, 1, 3, padding=1), GradientClip(), nn.Softplus())
        self.upmask = nn.Sequential(nn.Conv2d(128, 8 * 8 * 9, 1, padding=0))

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch * num, ch, ht, wd)
        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))
        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)
        net = self.relu(self.conv2(net))
        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8 * 8 * 9, ht, wd)
        return (0.01 * eta, upmask)

class UpdateModule(nn.Module):

    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2 * 3 + 1) ** 2
        self.corr_encoder = nn.Sequential(nn.Conv2d(cor_planes, 128, 1, padding=0), nn.ReLU(inplace=True), nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True))
        self.flow_encoder = nn.Sequential(nn.Conv2d(4, 128, 7, padding=3), nn.ReLU(inplace=True), nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(inplace=True))
        self.weight = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 2, 3, padding=1), GradientClip(), nn.Sigmoid())
        self.delta = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(128, 2, 3, padding=1), GradientClip())
        self.gru = ConvGRU(128, 128 + 128 + 64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None, mask=None):
        """ RaftSLAM update operator """
        batch, num, ch, ht, wd = net.shape
        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)
        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch * num, -1, ht, wd)
        inp = inp.view(batch * num, -1, ht, wd)
        corr = corr.view(batch * num, -1, ht, wd)
        flow = flow.view(batch * num, -1, ht, wd)
        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)
        delta = delta.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        weight = weight.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        net = net.view(*output_dim)
        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return (net, delta, weight, eta, upmask)
        else:
            return (net, delta, weight)
