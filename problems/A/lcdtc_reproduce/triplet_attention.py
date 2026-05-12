"""Convolutional Triplet Attention Module (CTAM) from the LCDTC paper.

Captures cross-dimensional interactions via tensor rotation and residual
transformations. Applied to the extra liquid content prediction head.
"""

import torch
import torch.nn as nn


class ZPool(nn.Module):
    """Concatenate max-pool and avg-pool along channel dim -> (2, H, W)."""

    def forward(self, x):
        return torch.cat(
            [torch.max(x, dim=1, keepdim=True)[0], torch.mean(x, dim=1, keepdim=True)],
            dim=1,
        )


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return self.bn(self.conv(x))


class TripletAttention(nn.Module):
    """Triplet Attention as described in the LCDTC paper (Section 4.3).

    Three parallel branches capture C-H, C-W, and H-W interactions.
    Output is the average of all three branches.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        self.zpool = ZPool()

        # Branch 1: C-H interaction (rotate along W axis)
        self.conv1 = BasicConv(2, 1, kernel_size, padding=kernel_size // 2)

        # Branch 2: C-W interaction (rotate along H axis)
        self.conv2 = BasicConv(2, 1, kernel_size, padding=kernel_size // 2)

        # Branch 3: spatial attention (H-W interaction)
        self.conv3 = BasicConv(2, 1, kernel_size, padding=kernel_size // 2)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        # Branch 1: rotate along W -> (B, W, H, C) -> zpool -> conv -> rotate back
        x1 = x.permute(0, 3, 2, 1)  # (B, W, H, C)
        x1 = self.zpool(x1)  # (B, 2, H, C)
        x1 = self.conv1(x1)  # (B, 1, H, C)
        x1 = self.sigmoid(x1)
        x1 = x1 * x.permute(0, 3, 2, 1)  # broadcast multiply
        x1 = x1.permute(0, 3, 2, 1)  # back to (B, C, H, W)

        # Branch 2: rotate along H -> (B, H, W, C) -> zpool -> conv -> rotate back
        x2 = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x2 = self.zpool(x2)  # (B, 2, W, C)
        x2 = self.conv2(x2)  # (B, 1, W, C)
        x2 = self.sigmoid(x2)
        x2 = x2 * x.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 3, 1, 2)  # back to (B, C, H, W)

        # Branch 3: spatial attention
        x3 = self.zpool(x)  # (B, 2, H, W)
        x3 = self.conv3(x3)  # (B, 1, H, W)
        x3 = self.sigmoid(x3)
        x3 = x3 * x

        # Average of three branches
        return (x1 + x2 + x3) / 3.0
