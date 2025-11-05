import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEAttention(nn.Module):
    """
    Squeeze-and-Excitation Attention
    论文: https://arxiv.org/abs/1709.01507
    """

    def __init__(self, dim, reduction_ratio=16):
        super(SEAttention, self).__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio

        # Squeeze: Global Average Pooling
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        # Excitation: Two FC layers with ReLU and Sigmoid
        self.excitation = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Squeeze
        y = self.squeeze(x).view(batch_size, channels)

        # Excitation
        y = self.excitation(y).view(batch_size, channels, 1, 1)

        # Scale
        return x * y.expand_as(x)


class ECAAttention(nn.Module):
    """
    Efficient Channel Attention
    论文: https://arxiv.org/abs/1910.03151
    """

    def __init__(self, dim, kernel_size=3):
        super(ECAAttention, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        # 1D convolution for local cross-channel interaction
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Global Average Pooling
        y = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        # 1D convolution along channel dimension
        y = y.unsqueeze(1)  # [B, 1, C]
        y = self.conv(y)  # [B, 1, C]
        y = self.sigmoid(y)  # [B, 1, C]

        # Reshape and apply attention
        y = y.squeeze(1).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    """

    def __init__(self, dim, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size

        # Spatial attention using convolution
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size,
                              padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Average and max pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # Concatenate and apply convolution
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        y = self.sigmoid(y)

        return x * y


class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    """

    def __init__(self, dim, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.reduction_ratio = reduction_ratio

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, dim)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        # Global Average Pooling and Max Pooling
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(batch_size, channels)
        max_pool = F.adaptive_max_pool2d(x, 1).view(batch_size, channels)

        # Shared MLP
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)

        # Combine and apply sigmoid
        y = self.sigmoid(avg_out + max_out).view(batch_size, channels, 1, 1)

        return x * y.expand_as(x)


class CBAMAttention(nn.Module):
    """
    Convolutional Block Attention Module
    论文: https://arxiv.org/abs/1807.06521
    """

    def __init__(self, dim, reduction_ratio=16, kernel_size=7):
        super(CBAMAttention, self).__init__()
        self.dim = dim

        # Channel attention and spatial attention
        self.channel_attention = ChannelAttention(dim, reduction_ratio)
        self.spatial_attention = SpatialAttention(dim, kernel_size)

    def forward(self, x):
        # Apply channel attention first, then spatial attention
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
