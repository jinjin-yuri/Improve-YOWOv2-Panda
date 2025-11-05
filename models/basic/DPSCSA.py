import typing as t
import torch
import torch.nn as nn
from einops import rearrange

from models.basic.wpl import WPL


class DPSCSA(nn.Module):
    def __init__(
            self,
            dim: int,
            head_num: int,
            window_size: int = 7,
            group_kernel_sizes: t.List[int] = [3, 5, 7, 9],
            group_kernel_sizes_m: t.List[int] = [3, 5, 7, 9],
            qkv_bias: bool = False,
            fuse_bn: bool = False,
            down_sample_mode: str = 'avg_pool',
            attn_drop_ratio: float = 0.,
            gate_layer: str = 'sigmoid',
    ):
        super(DPSCSA, self).__init__()
        self.dim = dim
        self.head_num = head_num
        self.head_dim = dim // head_num
        self.scaler = self.head_dim ** -0.5
        self.group_kernel_sizes = group_kernel_sizes
        self.window_size = window_size
        self.qkv_bias = qkv_bias
        self.fuse_bn = fuse_bn
        self.down_sample_mode = down_sample_mode

        assert self.dim % 4 == 0, 'The dimension of input feature should be divisible by 4.'
        self.group_chans = group_chans = self.dim // 4

        # 定义局部和全局深度卷积层
        self.local_dwc = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[0],
                                   padding=group_kernel_sizes[0] // 2, groups=group_chans)
        self.global_dwc_s = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[1],
                                      padding=group_kernel_sizes[1] // 2, groups=group_chans)
        self.global_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[2],
                                      padding=group_kernel_sizes[2] // 2, groups=group_chans)
        self.global_dwc_l = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes[3],
                                      padding=group_kernel_sizes[3] // 2, groups=group_chans)

        # 定义局部和全局深度卷积层（最大池化分支）
        self.local_dwc_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes_m[0],
                                     padding=group_kernel_sizes_m[0] // 2, groups=group_chans)
        self.global_dwc_s_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes_m[1],
                                        padding=group_kernel_sizes_m[1] // 2, groups=group_chans)
        self.global_dwc_m_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes_m[2],
                                        padding=group_kernel_sizes_m[2] // 2, groups=group_chans)
        self.global_dwc_l_m = nn.Conv1d(group_chans, group_chans, kernel_size=group_kernel_sizes_m[3],
                                        padding=group_kernel_sizes_m[3] // 2, groups=group_chans)

        # 注意力门控层平均
        self.sa_gate_a = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h_a = nn.GroupNorm(4, dim)
        self.norm_w_a = nn.GroupNorm(4, dim)
        # 注意力门控层最大
        self.sa_gate_m = nn.Softmax(dim=2) if gate_layer == 'softmax' else nn.Sigmoid()
        self.norm_h_m = nn.GroupNorm(4, dim)
        self.norm_w_m = nn.GroupNorm(4, dim)

        # 通道注意力部分
        # self.conv_d_avg = nn.Identity()  # 平均池化分支
        self.conv_d_max = nn.Identity()  # 最大池化分支
        # self.norm_avg = nn.GroupNorm(1, dim)  # 平均池化分支归一化
        self.norm_max = nn.GroupNorm(1, dim)  # 最大池化分支归一化

        # # 定义查询、键和值的卷积层（平均池化分支）
        # self.q_avg = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        # self.k_avg = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        # self.v_avg = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)

        # 定义查询、键和值的卷积层（最大池化分支）
        self.q_max = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.k_max = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)
        self.v_max = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, bias=qkv_bias, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.ca_gate = nn.Softmax(dim=1) if gate_layer == 'softmax' else nn.Sigmoid()
        # 根据窗口大小和下采样模式选择下采样函数
        if window_size == -1:
            self.down_func_avg = nn.AdaptiveAvgPool2d((1, 1))
            self.down_func_max = nn.AdaptiveMaxPool2d((1, 1))
        else:
            if down_sample_mode == 'recombination':
                self.down_func_avg = self.space_to_chans
                self.down_func_max = self.space_to_chans
                self.conv_d_avg = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1,
                                            bias=False)
                self.conv_d_max = nn.Conv2d(in_channels=dim * window_size ** 2, out_channels=dim, kernel_size=1,
                                            bias=False)
            elif down_sample_mode == 'avg_pool':
                self.down_func_avg = nn.AvgPool2d(kernel_size=(window_size, window_size), stride=window_size)
                self.down_func_max = nn.MaxPool2d(kernel_size=(window_size, window_size), stride=window_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入张量 x 的维度为 (B, C, H, W)
        """
        # 计算空间注意力优先级
        b, c, h_, w_ = x.size()

        # 平均池化分支
        x_h = x.mean(dim=3)
        l_x_h, g_x_h_s, g_x_h_m, g_x_h_l = torch.split(x_h, self.group_chans, dim=1)
        x_w = x.mean(dim=2)
        l_x_w, g_x_w_s, g_x_w_m, g_x_w_l = torch.split(x_w, self.group_chans, dim=1)

        # 最大池化分支
        x_h_m = x.max(dim=3).values
        l_x_h_m, g_x_h_s_m, g_x_h_m_m, g_x_h_l_m = torch.split(x_h_m, self.group_chans, dim=1)
        x_w_m = x.max(dim=2).values
        l_x_w_m, g_x_w_s_m, g_x_w_m_m, g_x_w_l_m = torch.split(x_w_m, self.group_chans, dim=1)

        # 计算水平注意力（平均池化分支）
        x_h_attn = self.sa_gate_a(self.norm_h_a(torch.cat((
            self.local_dwc(l_x_h),
            self.global_dwc_s(g_x_h_s),
            self.global_dwc_m(g_x_h_m),
            self.global_dwc_l(g_x_h_l),
        ), dim=1)))
        x_h_attn = x_h_attn.view(b, c, h_, 1)

        # 计算垂直注意力（平均池化分支）
        x_w_attn = self.sa_gate_a(self.norm_w_a(torch.cat((
            self.local_dwc(l_x_w),
            self.global_dwc_s(g_x_w_s),
            self.global_dwc_m(g_x_w_m),
            self.global_dwc_l(g_x_w_l)
        ), dim=1)))
        x_w_attn = x_w_attn.view(b, c, 1, w_)

        # 计算水平注意力（最大池化分支）
        x_h_attn_m = self.sa_gate_m(self.norm_h_m(torch.cat((
            self.local_dwc_m(l_x_h_m),
            self.global_dwc_s_m(g_x_h_s_m),
            self.global_dwc_m_m(g_x_h_m_m),
            self.global_dwc_l_m(g_x_h_l_m),
        ), dim=1)))
        x_h_attn_m = x_h_attn_m.view(b, c, h_, 1)

        # 计算垂直注意力（最大池化分支）
        x_w_attn_m = self.sa_gate_m(self.norm_w_m(torch.cat((
            self.local_dwc_m(l_x_w_m),
            self.global_dwc_s_m(g_x_w_s_m),
            self.global_dwc_m_m(g_x_w_m_m),
            self.global_dwc_l_m(g_x_w_l_m)
        ), dim=1)))
        x_w_attn_m = x_w_attn_m.view(b, c, 1, w_)

        x = x * x_h_attn * x_w_attn * x_h_attn_m * x_w_attn_m

        # 基于自注意力的通道注意力
        # # 平均池化分支
        # y_avg = self.down_func_avg(x)
        # y_avg = self.conv_d_avg(y_avg)
        # _, _, h_avg, w_avg = y_avg.size()
        # y_avg = self.norm_avg(y_avg)

        # 最大池化分支
        y_max = self.down_func_avg(x)
        y_max = self.conv_d_max(y_max)
        _, _, h_max, w_max = y_max.size()
        y_max = self.norm_max(y_max)

        # # 计算平均池化分支的注意力
        # q_avg = self.q_avg(y_avg)
        # k_avg = self.k_avg(y_avg)
        # v_avg = self.v_avg(y_avg)

        # 计算最大池化分支的注意力
        q_max = self.q_max(y_max)
        k_max = self.k_max(y_max)
        v_max = self.v_max(y_max)

        # # 重塑张量形状
        # q_avg = rearrange(q_avg, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
        #                   head_num=int(self.head_num), head_dim=int(self.head_dim))
        # k_avg = rearrange(k_avg, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
        #                   head_num=int(self.head_num), head_dim=int(self.head_dim))
        # v_avg = rearrange(v_avg, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
        #                   head_num=int(self.head_num), head_dim=int(self.head_dim))

        q_max = rearrange(q_max, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                          head_num=int(self.head_num), head_dim=int(self.head_dim))
        k_max = rearrange(k_max, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                          head_num=int(self.head_num), head_dim=int(self.head_dim))
        v_max = rearrange(v_max, 'b (head_num head_dim) h w -> b head_num head_dim (h w)',
                          head_num=int(self.head_num), head_dim=int(self.head_dim))

        # # 计算注意力（平均池化分支）
        # attn_avg = q_avg @ k_avg.transpose(-2, -1) * self.scaler
        # attn_avg = self.attn_drop(attn_avg.softmax(dim=-1))
        # attn_avg = attn_avg @ v_avg

        # 计算注意力（最大池化分支）
        attn_max = q_max @ k_max.transpose(-2, -1) * self.scaler
        attn_max = self.attn_drop(attn_max.softmax(dim=-1))
        attn_max = attn_max @ v_max

        # # 重塑张量形状
        # attn_avg = rearrange(attn_avg, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_avg),
        #                      w=int(w_avg))
        attn_max = rearrange(attn_max, 'b head_num head_dim (h w) -> b (head_num head_dim) h w', h=int(h_max),
                             w=int(w_max))

        # 合并两个分支的结果
        # attn_avg = attn_avg.mean((2, 3), keepdim=True)
        attn_max = attn_max.mean((2, 3), keepdim=True)
        # attn_a = self.ca_gate(attn_avg)
        attn_m = self.ca_gate(attn_max)

        return attn_m * x


if __name__ == "__main__":
    scsa = DPSCSA(dim=32, head_num=8, window_size=7)
    input_tensor = torch.rand(1, 32, 256, 256)
    print(f"输入张量的形状: {input_tensor.shape}")
    output_tensor = scsa(input_tensor)
    print(f"输出张量的形状: {output_tensor.shape}")