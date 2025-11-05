import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ModReLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.b = nn.Parameter(torch.Tensor(features))
        self.b.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        return torch.abs(x) * F.relu(torch.cos(torch.angle(x) + self.b))


class FFTNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.filter = nn.Linear(dim, dim)
        self.modrelu = ModReLU(dim)

    def forward(self, x):
        # 输入形状: [B, C, H, W]
        B, C, H, W = x.shape
        # 将输入从 [B, C, H, W] 转换为 [B, seq_len, dim]
        x_reshaped = x.view(B, H * W, C)  # seq_len = H * W, dim = C

        # FFTNetBlock 的核心操作
        x_fft = torch.fft.fft(x_reshaped, dim=1)  # FFT along the sequence dimension
        x_filtered = self.filter(x_fft.real) + 1j * self.filter(x_fft.imag)
        x_filtered = self.modrelu(x_filtered)
        x_out = torch.fft.ifft(x_filtered, dim=1).real

        # 将输出从 [B, seq_len, dim] 转换回 [B, C, H, W]
        x_out_reshaped = x_out.reshape(B, C, H, W)  # 使用 reshape 代替 view
        return x_out_reshaped


class MultiHeadSpectralAttention(nn.Module):
    def __init__(self, embed_dim, seq_len, num_heads=4, dropout=0.1, adaptive=True):
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
         # self.embed_dim = embed_dim
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim 必须能被 num_heads 整除")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.seq_len = seq_len
        self.adaptive = adaptive
        # 频域的 FFT 频率桶数量: (seq_len//2 + 1)
        self.freq_bins = seq_len // 2 + 1
        # 基础乘法滤波器: 每个注意力头和频率桶一个
        self.base_filter = nn.Parameter(torch.ones(num_heads, self.freq_bins, 1))
        # 基础加性偏置: 作为频率幅度的学习偏移
        self.base_bias = nn.Parameter(torch.full((num_heads, self.freq_bins, 1), -0.1))
        if adaptive:
            # 自适应 MLP: 每个头部和频率桶生成 2 个值(缩放因子和偏置)
            self.adaptive_mlp = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.GELU(),
                nn.Linear(embed_dim, num_heads * self.freq_bins * 2)
            )
        self.dropout = nn.Dropout(dropout)
        # 预归一化层,提高傅里叶变换的稳定性
        self.pre_norm = nn.LayerNorm(embed_dim)


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
        """
        增强型频谱注意力模块的前向传播。
        参数:
        x: 输入张量,形状为 (B, seq_len, embed_dim)
        返回:经过频谱调制和残差连接的张量,形状仍为 (B, seq_len, embed_dim)
        """
        # 输入形状: [B, C, H, W]
        B, C, H, W = x.shape
        # 将输入从 [B, C, H, W] 转换为 [B, seq_len, dim]
        x_reshaped = x.view(B, H * W, C)  # seq_len = H * W, dim = C
        B, N, D = x_reshaped.shape
        # 预归一化,提高频域变换的稳定性
        x_norm = self.pre_norm(x_reshaped)
        # 重新排列张量以分离不同的注意力头,形状变为 (B, num_heads, seq_len, head_dim)
        x_heads = x_norm.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # 沿着序列维度计算 FFT,结果为复数张量,形状为 (B, num_heads, freq_bins, head_dim)
        F_fft = torch.fft.rfft(x_heads, dim=2, norm='ortho')
        # 计算自适应调制参数(如果启用)
        if self.adaptive:
            # 全局上下文:对 token 维度求均值,形状为 (B, embed_dim)
            context = x_norm.mean(dim=1)
            # 经过 MLP 计算自适应参数,输出形状为 (B, num_heads*freq_bins*2)
            adapt_params = self.adaptive_mlp(context)
            adapt_params = adapt_params.view(B, self.num_heads, self.freq_bins, 2)
            # 划分为乘法缩放因子和加法偏置
            adaptive_scale = adapt_params[..., 0:1]  # 形状: (B, num_heads, freq_bins, 1)
            adaptive_bias = adapt_params[..., 1:2]  # 形状: (B, num_heads, freq_bins, 1)
        else:
            # 如果不使用自适应机制,则缩放因子和偏置设为 0
            adaptive_scale = torch.zeros(B, self.num_heads, self.freq_bins, 1, device=x.device)
            adaptive_bias = torch.zeros(B, self.num_heads, self.freq_bins, 1, device=x.device)
        # 结合基础滤波器和自适应调制参数
        # effective_filter: 影响频谱响应的缩放因子
        effective_filter = self.base_filter * (1 + adaptive_scale)
        # effective_bias: 影响频谱响应的偏置
        effective_bias = self.base_bias + adaptive_bias
        # 在频域进行自适应调制
        # 先进行乘法缩放,再添加偏置(在 head_dim 维度上广播)
        F_fft_mod = F_fft * effective_filter + effective_bias
        # 在频域应用非线性激活
        F_fft_nl = self.complex_activation(F_fft_mod)
        # 逆傅里叶变换(iFFT)还原到时序空间
        # 需要指定 n=self.seq_len 以确保输出长度匹配输入
        x_filtered = torch.fft.irfft(F_fft_nl, dim=2, n=self.seq_len, norm='ortho')
        # 重新排列张量,将注意力头合并回嵌入维度
        x_filtered = x_filtered.permute(0, 2, 1, 3).reshape(B, N, D)
        # 残差连接并应用 Dropout
        x_out = x_reshaped + self.dropout(x_filtered)
        x_out = x_out.reshape(B, C, H, W)  # 使用 reshape 代替 view
        return x_out


    # def profile_module(self, input: Tensor):
    #     """
    #     计算FLOPs和参数数量
    #     """
    #     B, C, H, W = input.shape
    #     seq_len = H * W
    #
    #     # 计算参数数量
    #     params = sum(p.numel() for p in self.parameters())
    #
    #     # 计算FLOPs
    #     flops = 0
    #
    #     # FFT操作
    #     flops += 5 * B * seq_len * int(math.log2(seq_len)) * C  # FFT和IFFT
    #
    #     # 自适应MLP
    #     if self.adaptive:
    #         flops += 2 * (self.embed_dim * self.embed_dim) * B  # 两个线性层
    #
    #     # FDConv的FLOPs
    #     flops += 2 * (B * C * H * W * 3 * 3 * C)  # 两个3x3卷积
    #
    #     # 其他操作(归一化、激活函数等)
    #     flops += B * seq_len * C * 2  # LayerNorm
    #
    #     self.total_ops = torch.tensor(flops)
    #     self.total_params = torch.tensor(params)
    #
    #     return input, params, flops


class pinyu(nn.Module):
    def __init__(self, dim, seq_len=None, num_heads=4, dropout=0.1, adaptive=True):
        """
        Initialize the pinyu module.

        Args:
            dim (int): The input dimension (number of channels).
            seq_len (int, optional): The sequence length (H*W). If None, it will be inferred from input shape.
            num_heads (int): Number of attention heads for MultiHeadSpectralAttention.
            dropout (float): Dropout rate.
            adaptive (bool): Whether to use adaptive modulation in MultiHeadSpectralAttention.
        """
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.dropout = dropout
        self.adaptive = adaptive
        # # 注册缓冲区用于FLOPs计算
        # self.register_buffer('total_ops', torch.tensor(0))
        # self.register_buffer('total_params', torch.tensor(0))
        # Initialize the MultiHeadSpectralAttention module
        self.spectral_attention = None  # Will be initialized in forward if seq_len is None
        print('pinyu6666')

        # If seq_len is provided, we can initialize the module immediately
        if self.seq_len is not None:
            self.spectral_attention = MultiHeadSpectralAttention(
                embed_dim=dim,
                seq_len=seq_len,
                num_heads=num_heads,
                dropout=dropout,
                adaptive=adaptive
            )

    def forward(self, x):
        # Input shape: [B, C, H, W]
        B, C, H, W = x.shape

        # If seq_len wasn't provided during initialization, infer it now
        if self.spectral_attention is None:
            self.seq_len = H * W
            self.spectral_attention = MultiHeadSpectralAttention(
                embed_dim=self.dim,
                seq_len=self.seq_len,
                num_heads=self.num_heads,
                dropout=self.dropout,
                adaptive=self.adaptive
            ).to(x.device)

        # Pass through the spectral attention module
        return self.spectral_attention(x)


if __name__ == '__main__':
    # 参数设置
    batch_size = 1  # 批量大小
    channels = 64  # 通道数
    height = 28  # 高度
    width = 28  # 宽度

    # 创建随机输入张量,形状为 (batch_size, channels, height, width)
    x = torch.randn(batch_size, channels, height, width)

    # 初始化模型
    model = pinyu(64)
    print("微信公众号: AI缝合术!")

    # 前向传播
    output = model(x)
    print("输入形状:", x.shape)
    print("输出形状:", output.shape)

    # 手动调用profile_module计算FLOPs和参数数量
    if hasattr(model.spectral_attention, 'profile_module'):
        _, total_params, total_ops = model.spectral_attention.profile_module(x)
        print(f"\n计算统计:")
        print(f"总参数数量: {total_params/ 1e6:,}")
        print(f"总FLOPs: {total_ops/ 1e9:,}")