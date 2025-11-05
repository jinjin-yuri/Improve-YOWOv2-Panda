import torch
from torch import nn
from torch.nn import init


class SpatialGroupEnhance_3D(nn.Module):

    def __init__(self, groups=8):#jiang 变通道原来64
        super().__init__()
        self.groups = groups
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # Use AdaptiveAvgPool3d for 5D input
        self.weight = nn.Parameter(torch.zeros(1, groups, 1, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, groups, 1, 1, 1))
        self.sig = nn.Sigmoid()
        self.init_weights()
        print("using SGE_3D")

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):  # Adjust for Conv3d
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):  # Adjust for BatchNorm3d
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, t1, h, w = x.shape
        x = x.view(b * self.groups, c // self.groups, t1, h, w)  # bs*g, dim//g, t, h, w
        xn = x * self.avg_pool(x)  # bs*g, dim//g, t, h, w
        xn = xn.sum(dim=1, keepdim=True)  # bs*g, 1, t, h, w
        t = xn.view(b*self.groups, -1)  # bs*g, t*h*w

        t = t - t.mean(dim=1, keepdim=True)  # bs*g, t*h*w
        std = t.std(dim=1, keepdim=True) + 1e-5
        t = t / std  # bs*g, t*h*w
        t = t.view(b, self.groups, t1, h, w)  # bs, g, 1, t*h*w, 1, 1

        t = t * self.weight + self.bias  # bs, g, 1, t*h*w, 1, 1
        t = t.view(b * self.groups, 1, t1, h, w)  # bs*g, 1, t, h, w
        x = x * self.sig(t)
        x = x.view(b, c, t1, h, w)

        return x


# 输入 N C T H W, 输出 N C T H W
if __name__ == '__main__':
    input = torch.randn(1, 256, 16, 64, 64)
    sge = SpatialGroupEnhance_3D()
    output = sge(input)
    print(input.size(), output.size())

