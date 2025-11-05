import torch.nn as nn
import torch


class GAM_Attention_SA(nn.Module):
    def __init__(self, in_channels, rate=4):
        super(GAM_Attention_SA, self).__init__()

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels / rate), kernel_size=7, padding=3),
            nn.BatchNorm2d(int(in_channels / rate)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channels / rate), in_channels, kernel_size=7, padding=3),
            nn.BatchNorm2d(in_channels)
        )
        print("Encoder using GAM_SA")

    def forward(self, x):

        x_spatial_att = self.spatial_attention(x).sigmoid()
        out = x * x_spatial_att

        return out

# 输入 N C H W,  输出 N C H W
if __name__ == '__main__':
    x = torch.randn(1, 64, 20, 20)
    b, c, h, w = x.shape
    net = GAM_Attention_SA(in_channels=c)
    y = net(x)
    print(y.size())