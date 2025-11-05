import torch
import torch.nn as nn
from einops import rearrange
import math

# 论文：Adapt or Perish: Adaptive Sparse Transformer with Attentive Feature Refinement for Image Restoration, CVPR 2024.
# 论文地址：https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_Adapt_or_Perish_Adaptive_Sparse_Transformer_with_Attentive_Feature_Refinement_CVPR_2024_paper.pdf
# 特征细化前馈网络（Feature Refinement Feed-forward Network, FRFN）是一种专门设计的深度学习结构，旨在提高图像处理任务中的特征表示能力
class FRFN(nn.Module):
    def __init__(self, dim=32, hidden_dim=128, act_layer=nn.GELU):
        super().__init__()
        self.linear1 = nn.Sequential(nn.Linear(dim, hidden_dim * 2),
                                     act_layer())
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, groups=hidden_dim, kernel_size=3, stride=1, padding=1),
            act_layer())
        self.linear2 = nn.Sequential(nn.Linear(hidden_dim, dim))
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.dim_conv = self.dim // 4
        self.dim_untouched = self.dim - self.dim_conv
        self.partial_conv3 = nn.Conv2d(self.dim_conv, self.dim_conv, 3, 1, 1, bias=False)

    def forward(self, x):
        x_init = x
        # bs x hw x c
        b, c, h, w = x.size()

        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        x = torch.cat((x1, x2), 1)

        # flaten
        x = rearrange(x, ' b c h w -> b (h w) c', h=h, w=w)

        x = self.linear1(x)
        # gate mechanism
        x_1, x_2 = x.chunk(2, dim=-1)

        x_1 = rearrange(x_1, ' b (h w) c -> b c h w ', h=h, w=w)
        x_1 = self.dwconv(x_1)
        x_1 = rearrange(x_1, ' b c h w -> b (h w) c', h=h, w=w)
        x = x_1 * x_2
        x = self.linear2(x)
        x = rearrange(x, ' b (h w) c -> b c h w ', h=h, w=w)
        x = x + x_init
        return x


if __name__ == '__main__':
    # Instantiate the FRFN class
    dim = 64  # Dimension of input features


    # Create an instance of the FRFN module
    frfn = FRFN(dim)

    # Generate a random input tensor
    B = 1  # Batch size
    H = 256  # Height of the feature map
    W = 256  # Width of the feature map
    C = dim  # Number of channels

    input = torch.randn(B, C, H, W)

    # Forward pass
    output = frfn(input)

    # Print input and output shapes
    print(input.size())
    print(output.size())