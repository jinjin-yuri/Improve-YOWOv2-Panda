import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor(features))
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        return torch.abs(x) * F.relu(torch.cos(torch.angle(x) + self.b))



class FrequencyHeadMixer(nn.Module):
    def __init__(self, num_heads, expansion=2):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Conv2d(num_heads, num_heads * expansion,
                      kernel_size=(3, 1),  # 沿频率维度卷积
                      padding=(1, 0),
                      groups=num_heads),  # 分组保持独立性
            nn.GELU(),
            nn.Conv2d(num_heads * expansion, num_heads,
                      kernel_size=(3, 1),
                      padding=(1, 0))
        )

    def forward(self, x_fft):
        """输入: [B, H, F, D] 复数张量"""
        # 实部和虚部分别处理
        real = self.mixer(x_fft.real.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        imag = self.mixer(x_fft.imag.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        return torch.complex(real, imag)

class MSFFT(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads=4, dropout=0.3):
        """
        频谱注意力模块,在保持 O(n log n) 计算复杂度的同时,引入额外的非线性和自适应能力。
        参数:
          - embed_dim: 总的嵌入维度。
          - seq_len: 序列长度(例如 Transformer 中 token 的数量,包括类 token)。
          - num_heads: 注意力头的数量。
          - dropout: 逆傅里叶变换(iFFT)后的 dropout 率。
          - adaptive: 是否启用自适应 MLP 以生成乘法和加法的自适应调制参数。
        """
        super().__init__()
        self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        # 频域的 FFT 频率桶数量: (seq_len//2 + 1)
        self.freq_bins = seq_len // 2 + 1
        # 基础乘法滤波器: 每个注意力头和频率桶一个
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))
        # 基础加性偏置: 作为频率幅度的学习偏移
        self.base_bias = nn.Parameter(torch.full((num_heads, self.freq_bins, 1), -0.1))
        self.adaptive_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_heads * self.freq_bins * 2)
        )
        self.dropout = nn.Dropout(dropout)
        # 预归一化层,提高傅里叶变换的稳定性
        self.pre_norm = nn.LayerNorm(embed_dim)
        # 新增模块
        self.head_mixer = FrequencyHeadMixer(num_heads)
        print('MMMMMSSSTTT')


    def complex_activation(self, z):
        """
        对复数张量应用非线性激活函数。该函数计算 z 的幅度,将其传递到 GELU 进行非线性变换,并按比例缩放 z,以保持相位不变。
        参数:
          z: 形状为 (B, num_heads, freq_bins, head_dim) 的复数张量
        返回:
          经过非线性变换的复数张量,形状相同。
        """
        mag = torch.abs(z)
        # 对幅度进行非线性变换,GELU 提供平滑的非线性
        mag_act = F.gelu(mag)
        # 计算缩放因子,防止除零错误
        scale = mag_act / (mag + 1e-6)
        return z * scale

    def forward(self, x):
        B, N, D = x.shape
        x_norm = self.pre_norm(x)
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 原始FFT处理
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')

        # ---- 新增头频率交互 ----
        F_fft = self.head_mixer(F_fft)

        # 自适应调制（保持原有）
        context = x_norm.mean(dim=1)
        adapt_params = self.adaptive_mlp(context).view(B, self.num_heads, self.freq_bins, 2)
        effective_filter = self.base_filter * (1 + adapt_params[..., 0:1])
        effective_bias = self.base_bias + adapt_params[..., 1:2]
        F_fft_mod = F_fft * effective_filter + effective_bias

        F_fft_nl = self.complex_activation(F_fft_mod)

        # 逆变换和输出（保持原有）
        x_filtered = torch.fft.irfft(F_fft_nl, n=self.seq_len, dim=2, norm='ortho')
        #  x_out = x + self.dropout(x_filtered.permute(0, 2, 1, 3).reshape(B, N, D))
        x_out = x_filtered.permute(0, 2, 1, 3).reshape(B, N, D)
        return x_out



class ECAAttention(nn.Module):
    """ECA-Net的跨通道交互（替换SE）"""

    def __init__(self, dim, kernel_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        print(' USE  ECA')

    def forward(self, x):
        B, C, H, W = x.shape
        y = self.avg_pool(x).squeeze(-1).transpose(-1, -2)  # [B, 1, C]
        y = self.conv(y).transpose(-1, -2).unsqueeze(-1)  # [B, C, 1, 1]
        return x * self.sigmoid(y)   # x *


class MLSA(nn.Module):
    def __init__(self, dim, seq_len=None, num_heads=2, dropout=0.2):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        print('MLSA   MLSA')

        # 频谱处理（含内置ModReLU）
        self.low_freq_fft = MSFFT(dim, 49, num_heads=num_heads, dropout=dropout)
        # # ECA注意力
        # self.eca = ECAAttention(dim)

    def spectral_forward(self, x):
        B, C, H, W = x.shape
        # 转换 [B, H*W, C]
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        # 内部已正确处理频域激活
        x_out = self.low_freq_fft(x_flat)
        return x + self.dropout(rearrange(x_out, 'b (h w) c -> b c h w', h=H, w=W))

    def forward(self, x):
        if self.seq_len is None:
            _, _, H, W = x.shape
            self.seq_len = H * W

        spectral_out = self.spectral_forward(x)
        # eca_out = self.eca(spectral_out)
        return spectral_out


if __name__ == '__main__':
    # 初始化
    attn = MLSA(dim=976)

    # 前向计算 - 3D输入测试
    x = torch.randn(8, 976, 7, 7)  # [B,C,H,W]
    out = attn(x)  # 保持输入尺寸
    print("输入形状:", x.shape)
    print("输出形状:", out.shape)
    # 计算量统计
    print(f"参数量: {sum(p.numel() for p in attn.parameters()) / 1e6:.2f}M")