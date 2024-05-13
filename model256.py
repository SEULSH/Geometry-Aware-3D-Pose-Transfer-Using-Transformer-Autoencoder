
from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from xcit import XCABlock1D

class PoseFeature(nn.Module):
    def __init__(self, num_points=6890):
        super(PoseFeature, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv11 = XCABlock1D(64, 4, eta=1e-5)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv21 = XCABlock1D(128, 8, eta=1e-5)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.norm1 = torch.nn.InstanceNorm1d(64)
        self.norm2 = torch.nn.InstanceNorm1d(128)
        self.norm3 = torch.nn.InstanceNorm1d(256)

        self.num_points = num_points

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv11(x.transpose(1, 2))
        x = F.relu(self.norm1(x.transpose(1, 2)))
        x = self.conv2(x)
        x = self.conv21(x.transpose(1, 2))
        x = F.relu(self.norm2(x.transpose(1, 2)))
        x = self.conv3(x)
        x = F.relu(self.norm3(x))

        return x


class PoseTransformer(nn.Module):
    def __init__(self, num_points):
        super(PoseTransformer, self).__init__()
        in_channels = 3
        out_channels = [64, 128, 256]
        latent_channels = 256

        self.pose_only = PoseFeature(num_points=num_points)

        eta = 1e-5  # for
        self.pose_mlp = nn.Sequential(
            nn.Linear(in_channels, out_channels[0], bias=True),
            torch.nn.ReLU(),
            nn.Linear(out_channels[0], out_channels[1], bias=True),
            torch.nn.ReLU(),
            nn.Linear(out_channels[1], out_channels[2], bias=True),
        )
        self.shape_enc = nn.ModuleList(
            [XCABlock1D(latent_channels, 16, eta=eta), XCABlock1D(latent_channels, 16, eta=eta)])

        self.conv1 = torch.nn.Conv1d(latent_channels*2, latent_channels, 1)

        self.conv2 = torch.nn.Conv1d(latent_channels, latent_channels // 2, 1)

        self.conv4 = torch.nn.Conv1d(latent_channels // 2, 3, 1)
        self.th = nn.Tanh()

        self.shape_dec1 = nn.ModuleList(
            [XCABlock1D(latent_channels, 16, eta=eta), XCABlock1D(latent_channels, 16, eta=eta)])
        self.shape_dec2 = nn.ModuleList(
            [XCABlock1D(latent_channels // 2, 8, eta=eta), XCABlock1D(latent_channels // 2, 8, eta=eta)])


    def encode(self, x):
        x = self.inference_model(x)
        return x  #

    def inference_model(self, x):
        x = self.pose_mlp(x)
        for enc in self.shape_enc:
            x = enc(x)
        return x

    def decode(self, identity_f, pose):
        logits = self.generative_model(identity_f, pose)
        return logits

    def generative_model(self, identity_f, pose):

        x = torch.cat([identity_f, pose], dim=1)
        x = self.conv1(x).transpose(1, 2)
        for dec in self.shape_dec1:
            x = dec(x)
        x = self.conv2(F.relu(x.transpose(1,2))).transpose(1,2)
        for dec in self.shape_dec2:
            x = dec(x)
        x = 2 * self.th(self.conv4(x.transpose(1, 2)))

        return x


    def forward(self, pose, identity):

        pose = self.pose_only(pose)

        identity_f = self.encode(identity.transpose(1, 2))

        y = self.decode(identity_f.transpose(1, 2), pose)

        return y.transpose(1, 2)